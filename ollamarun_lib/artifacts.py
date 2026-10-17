"""Artifact storage for the serve API: immutable file snapshots behind
opaque ids.

One store backs every artifact. `POST /v1/files` uploads and tool outputs
both register here, and a `file_id` in a message part resolves through the
same lookup. Registration snapshots the bytes; `sha256` and `size` are
computed from what was stored, never from the request, and every lookup
goes through metadata (a validated id picks the manifest, never a path
built from client input).

Layout (under the configured root):

    index.json                  id -> {path, size, ..., pending}
    index.lock                  the flock behind index updates
    meta/<id>.json              authoritative per-artifact manifest
    sessions/<sid>/<turn>/<id>  artifacts a turn produced
    uploads/<id>                uploads (no turn yet)

Writes are atomic (temp file + `os.replace`). The quota is reserved in the
index under a shared lock (in-process `RLock` + `flock`, the
`SessionStore` pattern), so concurrent uploads cannot overshoot it.
`sweep()` drops expired artifacts, stale reservations and debris; it runs
at serve startup and, rate-limited, on registration.

The store raises `ApiError` directly -- it exists for the HTTP layer, which
renders those codes (`artifact_not_found`, `artifact_too_large`,
`artifact_quota_exceeded`).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import mimetypes
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .constants import (ARTIFACTS_DIR, DEFAULT_ARTIFACT_MAX_BYTES,
                        DEFAULT_ARTIFACT_QUOTA_BYTES, DEFAULT_ARTIFACT_TTL)
from .helpers import now_iso
from .server import ApiError

logger = logging.getLogger("ollamarun.artifacts")

try:  # POSIX file locking: two serve processes sharing a store must not
    import fcntl  # interleave their index read-modify-write.
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

#: The one id shape: opaque, URL-safe and filename-safe.
ARTIFACT_ID_RE = re.compile(r"art_[0-9a-f]{32}\Z")
_MEDIA_TYPE_RE = re.compile(
    r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*\Z")
_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_-]+")
_SOURCES = ("user", "tool", "model")
_INDEX_SCHEMA = 1

#: A stale `pending` reservation or leftover temp file older than this is
#: debris from an interrupted registration.
STALE_AFTER = 3600.0
#: The rate limit behind the opportunistic sweep on registration.
SWEEP_INTERVAL = 300.0

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _index_lock(index_path: Path) -> threading.RLock:
    """One reentrant lock per index file, shared by every store in this process."""
    key = str(index_path)
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def sanitize_filename(name: Any) -> str:
    """The display filename: never a path, never a control character."""
    text = str(name or "").replace("\\", "/").split("/")[-1]
    text = "".join(ch for ch in text if ch.isprintable()).strip()
    if not text or set(text) <= {"."}:
        return "file"
    return text[:255]


def sanitize_component(value: Any) -> str:
    """A session/turn id as one filesystem path component."""
    cleaned = _PATH_COMPONENT_RE.sub("_", str(value or ""))[:64]
    return cleaned or "_"


def media_type_of(filename: str) -> str:
    guess, _encoding = mimetypes.guess_type(str(filename or ""))
    return guess or "application/octet-stream"


def _tmp_path(target: Path) -> Path:
    """A per-writer temp name: two processes may write the same target."""
    return target.with_name(
        f"{target.name}.{os.getpid()}-{threading.get_ident()}.tmp")


def _safe_rel(rel: str) -> Optional[str]:
    """A manifest path, accepted only as a relative path inside the root."""
    rel = str(rel or "")
    if not rel or rel.startswith("/") or "\\" in rel:
        return None
    parts = Path(rel).parts
    if any(part in ("..", "") for part in parts):
        return None
    return rel


@dataclass(frozen=True)
class ArtifactRef:
    """One stored artifact: the public record every surface hands around."""

    id: str
    media_type: str
    filename: str
    size: int
    sha256: str
    source: str
    turn_id: Optional[str]
    session_id: Optional[str]
    created_at: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        """Fixed key order: id, media_type, filename, size, sha256, source,
        turn_id, session_id, created_at, url."""
        return {"id": self.id, "media_type": self.media_type,
                "filename": self.filename, "size": self.size,
                "sha256": self.sha256, "source": self.source,
                "turn_id": self.turn_id, "session_id": self.session_id,
                "created_at": self.created_at, "url": self.url}


def _ref_from_manifest(data: dict[str, Any]) -> Optional[ArtifactRef]:
    try:
        ref = ArtifactRef(id=str(data["id"]),
                          media_type=str(data["media_type"]),
                          filename=str(data["filename"]),
                          size=int(data["size"]),
                          sha256=str(data["sha256"]),
                          source=str(data["source"]),
                          turn_id=data.get("turn_id"),
                          session_id=data.get("session_id"),
                          created_at=str(data["created_at"]),
                          url=str(data["url"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not ARTIFACT_ID_RE.fullmatch(ref.id):
        return None
    return ref


class ArtifactStore:
    """The one artifact registry: bytes, metadata, quota and retention."""

    def __init__(self, root: Union[str, Path] = ARTIFACTS_DIR, *,
                 max_bytes: int = DEFAULT_ARTIFACT_MAX_BYTES,
                 quota_bytes: int = DEFAULT_ARTIFACT_QUOTA_BYTES,
                 ttl: float = DEFAULT_ARTIFACT_TTL,
                 sweep_interval: float = SWEEP_INTERVAL,
                 clock: Callable[[], float] = time.time):
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.quota_bytes = int(quota_bytes)
        self.ttl = float(ttl)
        self.sweep_interval = float(sweep_interval)
        self._clock = clock
        self.meta_dir = self.root / "meta"
        self.index_path = self.root / "index.json"
        # `index.json` is a read-modify-write (concurrent uploads reserve
        # quota in it). The in-process lock is shared by every store for the
        # same index; `_index_guard` adds an exclusive flock for processes.
        self._lock = _index_lock(self.index_path)
        self._guard_depth = 0
        self._flock_file = None
        self._last_sweep = 0.0

    # ── index plumbing ──────────────────────────────────────────────────────

    @contextlib.contextmanager
    def _index_guard(self):
        """Serialize an index read-modify-write in this process and on disk."""
        with self._lock:
            if self._guard_depth == 0 and fcntl is not None:
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                self._flock_file = open(f"{self.index_path}.lock", "a+")
                fcntl.flock(self._flock_file.fileno(), fcntl.LOCK_EX)
            self._guard_depth += 1
            try:
                yield
            finally:
                self._guard_depth -= 1
                if self._guard_depth == 0 and self._flock_file is not None:
                    try:
                        fcntl.flock(self._flock_file.fileno(), fcntl.LOCK_UN)
                    finally:
                        self._flock_file.close()
                        self._flock_file = None

    def _read_index(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
        if not isinstance(data, dict) or data.get("schema") != _INDEX_SCHEMA:
            return {}
        entries = data.get("artifacts")
        if not isinstance(entries, dict):
            return {}
        return {str(key): dict(value) for key, value in entries.items()
                if isinstance(value, dict)}

    def _write_index(self, entries: dict[str, dict[str, Any]]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": _INDEX_SCHEMA, "artifacts": entries}
        tmp = _tmp_path(self.index_path)
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        os.replace(tmp, self.index_path)

    # ── manifests and lookups ───────────────────────────────────────────────

    def _manifest_path(self, artifact_id: str) -> Path:
        return self.meta_dir / f"{artifact_id}.json"

    def _write_manifest(self, ref: ArtifactRef, rel_path: str) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        target = self._manifest_path(ref.id)
        payload = {**ref.to_dict(), "path": rel_path,
                   "stored_at": float(self._clock())}
        tmp = _tmp_path(target)
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        os.replace(tmp, target)

    def _read_manifest(self, artifact_id: str) -> Optional[dict[str, Any]]:
        try:
            data = json.loads(
                self._manifest_path(artifact_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _manifest(self, artifact_id: Any) -> Optional[tuple[ArtifactRef, str]]:
        if not isinstance(artifact_id, str) \
                or not ARTIFACT_ID_RE.fullmatch(artifact_id):
            return None
        data = self._read_manifest(artifact_id)
        if not data:
            return None
        ref = _ref_from_manifest(data)
        if ref is None:
            return None
        rel = _safe_rel(data.get("path"))
        if rel is None:
            return None
        return ref, rel

    def resolve(self, artifact_id: str) -> Optional[ArtifactRef]:
        """The ref for one id, or None (unknown, malformed, expired+reaped)."""
        loaded = self._manifest(artifact_id)
        return loaded[0] if loaded else None

    def read(self, artifact_id: str) -> Optional[tuple[ArtifactRef, bytes]]:
        """The ref plus the stored bytes, or None when the object is gone."""
        loaded = self._manifest(artifact_id)
        if loaded is None:
            return None
        ref, rel = loaded
        try:
            data = (self.root / rel).read_bytes()
        except OSError:
            return None
        return ref, data

    # ── registration ────────────────────────────────────────────────────────

    def _reserve(self, artifact_id: str, rel_path: str, size: int) -> None:
        """Check the quota and hold it, before any bytes are written."""
        with self._index_guard():
            entries = self._read_index()
            if self.quota_bytes > 0:
                total = sum(int(item.get("size") or 0)
                            for item in entries.values())
                if total + size > self.quota_bytes:
                    raise ApiError(
                        507, "the artifact store is over its storage quota",
                        code="artifact_quota_exceeded")
            entries[artifact_id] = {"path": rel_path, "size": size,
                                    "pending": True,
                                    "created_at": float(self._clock())}
            self._write_index(entries)

    def _commit_reservation(self, artifact_id: str) -> None:
        with self._index_guard():
            entries = self._read_index()
            entry = entries.get(artifact_id)
            if entry is not None and entry.get("pending"):
                entry["pending"] = False
                self._write_index(entries)

    def _drop_reservation(self, artifact_id: str) -> None:
        with self._index_guard():
            entries = self._read_index()
            if entries.pop(artifact_id, None) is not None:
                self._write_index(entries)

    def register(self, data: bytes, *, filename: str, media_type: str,
                 source: str = "user", session_id: Optional[str] = None,
                 turn_id: Optional[str] = None) -> ArtifactRef:
        """Snapshot bytes as a new artifact; raises `ApiError` on rejection."""
        if source not in _SOURCES:
            raise ApiError(400, f"unknown artifact source: {source!r}",
                           code="unsupported_value")
        media = str(media_type or "").strip().lower()
        if not _MEDIA_TYPE_RE.fullmatch(media):
            raise ApiError(400, f"invalid media type: {media_type!r}",
                           param="media_type", code="unsupported_value")
        payload = bytes(data)
        if 0 < self.max_bytes < len(payload):
            raise ApiError(
                413, f"artifact exceeds the {self.max_bytes} byte limit",
                code="artifact_too_large")
        artifact_id = "art_" + secrets.token_hex(16)
        rel = _safe_rel(
            f"sessions/{sanitize_component(session_id)}"
            f"/{sanitize_component(turn_id)}/{artifact_id}"
            if session_id and turn_id else f"uploads/{artifact_id}")
        if rel is None:  # pragma: no cover - components cannot escape
            raise ApiError(400, "invalid artifact path", code="unsupported_value")
        self._reserve(artifact_id, rel, len(payload))
        target = self.root / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = _tmp_path(target)
            with open(tmp, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
            ref = ArtifactRef(
                id=artifact_id, media_type=media, filename=sanitize_filename(filename),
                size=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                source=source, turn_id=turn_id, session_id=session_id,
                created_at=now_iso(), url=f"/v1/artifacts/{artifact_id}")
            self._write_manifest(ref, rel)
            self._commit_reservation(artifact_id)
        except BaseException:
            self._drop_reservation(artifact_id)
            self._unlink_quiet(target)
            self._unlink_quiet(_tmp_path(target))
            self._unlink_quiet(self._manifest_path(artifact_id))
            raise
        self._maybe_sweep()
        return ref

    def for_turn(self, *, turn_id: str, session_id: str) -> "TurnArtifacts":
        """The publisher facade a turn's tools register through."""
        return TurnArtifacts(self, turn_id=turn_id, session_id=session_id)

    # ── retention ───────────────────────────────────────────────────────────

    @staticmethod
    def _unlink_quiet(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.unlink()

    def _scan_manifests(self) -> dict[str, tuple[ArtifactRef, str, float]]:
        found: dict[str, tuple[ArtifactRef, str, float]] = {}
        if not self.meta_dir.exists():
            return found
        try:
            candidates = sorted(self.meta_dir.glob("art_*.json"))
        except OSError:
            return found
        for path in candidates:
            artifact_id = path.name[:-len(".json")]
            data = self._read_manifest(artifact_id)
            if not data:
                continue
            ref = _ref_from_manifest(data)
            rel = _safe_rel(data.get("path"))
            if ref is None or ref.id != artifact_id or rel is None:
                continue
            try:
                stored_at = float(data.get("stored_at") or 0)
            except (TypeError, ValueError):
                stored_at = 0.0
            found[artifact_id] = (ref, rel, stored_at)
        return found

    def _maybe_sweep(self) -> None:
        now = self._clock()
        if self._last_sweep and now - self._last_sweep < self.sweep_interval:
            return
        try:
            self.sweep(now=now)
        except OSError:
            logger.debug("opportunistic artifact sweep failed", exc_info=True)

    def sweep(self, *, now: Optional[float] = None) -> int:
        """Drop expired artifacts, stale reservations and debris.

        `meta/` is authoritative: a manifest without an index entry is
        re-adopted (so the quota counts it), an index entry without a
        manifest is debris. Returns how many artifacts were removed.
        """
        now = float(self._clock() if now is None else now)
        removed = 0
        changed = False
        with self._index_guard():
            entries = self._read_index()
            manifests = self._scan_manifests()
            for artifact_id in sorted(entries):
                entry = entries[artifact_id]
                manifest = manifests.get(artifact_id)
                if manifest is None:
                    if bool(entry.get("pending")) \
                            and now - float(entry.get("created_at") or 0) \
                            <= STALE_AFTER:
                        continue  # a registration in flight
                    del entries[artifact_id]
                    self._unlink_quiet(self.root / str(entry.get("path") or ""))
                    removed += 1
                    changed = True
                    continue
                ref, rel, stored_at = manifest
                entry["path"] = rel
                entry["size"] = int(ref.size)
                entry["created_at"] = stored_at \
                    or float(entry.get("created_at") or now)
                if bool(entry.get("pending")):
                    entry["pending"] = False  # completed after a crash
                    changed = True
                if self.ttl > 0 and now - float(entry["created_at"]) > self.ttl:
                    del entries[artifact_id]
                    self._unlink_quiet(self.root / rel)
                    self._unlink_quiet(self._manifest_path(artifact_id))
                    removed += 1
                    changed = True
            for artifact_id, (ref, rel, stored_at) in manifests.items():
                if artifact_id in entries:
                    continue
                if not (self.root / rel).is_file():
                    self._unlink_quiet(self._manifest_path(artifact_id))
                    changed = True
                    continue
                entries[artifact_id] = {"path": rel, "size": int(ref.size),
                                        "pending": False,
                                        "created_at": stored_at or now}
                changed = True
            if changed or not self.index_path.exists():
                self._write_index(entries)
        try:
            debris = [path for path in self.root.rglob("*.tmp") if path.is_file()]
        except OSError:  # pragma: no cover - unreadable root
            debris = []
        for path in debris:
            with contextlib.suppress(OSError):
                if now - path.stat().st_mtime > STALE_AFTER:
                    path.unlink()
        self._last_sweep = now
        return removed


@dataclass
class TurnArtifacts:
    """A turn-bound publisher: refs made through it carry the turn's identity.

    Tools receive this through `ToolContext.artifacts` instead of the raw
    store, so a handler never deals with provenance bookkeeping.
    """

    store: ArtifactStore
    turn_id: str
    session_id: str

    def publish_path(self, path: Any, *, source: str = "tool") -> ArtifactRef:
        """Snapshot one regular file; raises `ApiError` if it cannot."""
        target = Path(str(path)).expanduser()
        if not target.is_file():
            raise ApiError(400, f"not a regular file: {path}",
                           code="unsupported_value")
        size = target.stat().st_size
        if 0 < self.store.max_bytes < size:
            raise ApiError(
                413, f"artifact exceeds the {self.store.max_bytes} byte limit",
                code="artifact_too_large")
        return self.store.register(
            target.read_bytes(), filename=target.name,
            media_type=media_type_of(target.name), source=source,
            session_id=self.session_id, turn_id=self.turn_id)
