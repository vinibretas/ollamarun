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

import base64
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union
from urllib.parse import urljoin, urlsplit

import requests

from .constants import (ARTIFACTS_DIR, DEFAULT_ARTIFACT_MAX_BYTES,
                        DEFAULT_ARTIFACT_QUOTA_BYTES, DEFAULT_ARTIFACT_TTL,
                        DEFAULT_SERVE_MAX_CONTENT_PARTS,
                        DEFAULT_SERVE_MAX_IMAGE_BYTES,
                        DEFAULT_SERVE_MAX_IMAGES,
                        DEFAULT_SERVE_MAX_MESSAGES,
                        DEFAULT_SERVE_MAX_MESSAGE_BYTES,
                        DEFAULT_SERVE_MAX_TEXT_BYTES)
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
#: One image part's decoded ceiling (data URI, remote URL or uploaded file).
MAX_IMAGE_BYTES = 10 * 1024 * 1024
#: How much of a text-ish artifact is inlined into a message.
INLINE_TEXT_BYTES = 32 * 1024
#: Remote media fetch: connect/read timeouts and the redirect budget.
REMOTE_TIMEOUT = 10.0
REMOTE_REDIRECTS = 3
#: Media types inlined into messages besides `text/*`.
TEXTUAL_MEDIA_TYPES = frozenset({
    "application/json", "application/xml", "application/yaml",
    "application/x-yaml", "application/toml", "application/x-sh",
    "application/javascript", "application/sql", "application/x-python",
    "application/x-httpd-php", "application/x-perl", "application/x-ruby",
    "application/graphql", "application/ld+json", "application/x-ndjson",
})

_DATA_URI_RE = re.compile(
    r"^data:(?P<type>image/[\w.+-]+);base64,(?P<data>.+)$", re.S)

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
        #: The resolved root: every storage path read back from the index or
        #: a manifest must resolve (symlinks followed) under it.
        self._root_resolved = self.root.resolve()
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

    def _contained(self, rel: str) -> Optional[Path]:
        """The resolved storage path for one manifest rel-path, or None.

        The rel-paths are store-generated, but the index and manifests are
        read back from disk: the resolved path (symlinks followed) must stay
        under the resolved root, else the entry is treated as absent — a
        symlinked directory under the root can never redirect a read or a
        sweep outside it.
        """
        safe = _safe_rel(rel)
        if safe is None:
            return None
        path = (self.root / safe).resolve()
        if path != self._root_resolved \
                and self._root_resolved not in path.parents:
            logger.warning("artifact path escapes the store root: %r", rel)
            return None
        return path

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
        if rel is None or self._contained(rel) is None:
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
        path = self._contained(rel)
        if path is None:
            return None
        try:
            data = path.read_bytes()
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
            if ref is None or ref.id != artifact_id or rel is None \
                    or self._contained(rel) is None:
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
                    stale = self._contained(str(entry.get("path") or ""))
                    if stale is not None:
                        self._unlink_quiet(stale)
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
                    expired = self._contained(rel)
                    if expired is not None:
                        self._unlink_quiet(expired)
                    self._unlink_quiet(self._manifest_path(artifact_id))
                    removed += 1
                    changed = True
            for artifact_id, (ref, rel, stored_at) in manifests.items():
                if artifact_id in entries:
                    continue
                stored = self._contained(rel)
                if stored is None or not stored.is_file():
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


# ── shared content-part resolution ──────────────────────────────────────────
#
# Both API surfaces (OpenAI messages and native agent turns) resolve
# `text`/`image_url`/`file` parts through the functions below, so file
# references, the image capability answer and every error code are identical
# on both wires while each surface keeps its own request schema.


def sniff_media_type(data: bytes) -> str:
    """A best-effort media type from magic bytes (remote fetches)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def fetch_remote_media(url: str, *, max_bytes: int = MAX_IMAGE_BYTES,
                       timeout: float = REMOTE_TIMEOUT,
                       max_redirects: int = REMOTE_REDIRECTS
                       ) -> tuple[bytes, str]:
    """Fetch an http(s) URL with explicit redirect, timeout and size limits.

    One implementation for every remote input, so the OpenAI surface and
    any artifact ingestion answer the same way. The SSRF caveat is the
    locked trusted-network contract: this phase does not filter addresses.
    """
    current = url
    for _attempt in range(max_redirects + 1):
        parsed = urlsplit(current)
        if parsed.scheme not in ("http", "https"):
            raise ApiError(400, "only http(s) URLs are supported",
                           code="unsupported_value")
        try:
            response = requests.get(
                current, stream=True, allow_redirects=False,
                timeout=(timeout, timeout),
                headers={"User-Agent": "ollamarun"})
        except requests.RequestException:
            logger.debug("remote media fetch failed: %s", current, exc_info=True)
            raise ApiError(400, "could not fetch the remote URL",
                           code="remote_fetch_failed")
        try:
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise ApiError(400, "could not fetch the remote URL",
                                   code="remote_fetch_failed")
                current = urljoin(current, location)
                continue
            if response.status_code != 200:
                raise ApiError(
                    400, f"could not fetch the remote URL "
                         f"(HTTP {response.status_code})",
                    code="remote_fetch_failed")
            media_type = str(response.headers.get("Content-Type") or "") \
                .split(";")[0].strip().lower()
            collected = bytearray()
            for chunk in response.iter_content(64 * 1024):
                collected.extend(chunk)
                if len(collected) > max_bytes:
                    raise ApiError(
                        413, f"image exceeds the {max_bytes} byte limit",
                        code="image_too_large")
        finally:
            response.close()
        if not collected:
            raise ApiError(400, "the remote URL returned no data",
                           code="remote_fetch_failed")
        return bytes(collected), media_type or sniff_media_type(bytes(collected))
    raise ApiError(400, "could not fetch the remote URL (too many redirects)",
                   code="remote_fetch_failed")


def require_vision(capabilities: Any) -> None:
    """Refuse image input for a model whose capability list has no `vision`.

    An empty/missing list passes: older backends cannot be verified, and an
    unreachable backend fails the request with its own error.
    """
    caps = {str(capability) for capability in (capabilities or [])}
    if caps and "vision" not in caps:
        raise ApiError(400, "this model does not support image input",
                       code="model_does_not_support_images", param="messages")


def _is_textual(media_type: str) -> bool:
    media = str(media_type or "").lower()
    return (media.startswith("text/")
            or media in TEXTUAL_MEDIA_TYPES
            or media.endswith(("+json", "+xml")))


@dataclass
class ResolvedImage:
    """One image payload, with the artifact id when a file part produced it."""

    b64: str
    artifact_id: Optional[str] = None


@dataclass
class ResolvedContent:
    """One message's resolved content: text, image payloads and refs used."""

    text: str = ""
    images: list[ResolvedImage] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)


def _image_from_url(url: str, *, where: str, image_max: int,
                    fetch: Callable[..., tuple[bytes, str]]) -> str:
    """A data: or http(s) image URL as a base64 payload."""
    match = _DATA_URI_RE.match(url)
    if match is not None:
        data = re.sub(r"\s+", "", match.group("data"))
        if (len(data) * 3) // 4 > image_max:
            raise ApiError(413, f"image exceeds the {image_max} byte limit",
                           param=where, code="image_too_large")
        try:
            base64.b64decode(data, validate=True)
        except (ValueError, TypeError):
            raise ApiError(400, "image data is not valid base64", param=where)
        return data
    if url.lower().startswith(("http://", "https://")):
        data, _media_type = fetch(url, max_bytes=image_max)
        return base64.b64encode(data).decode("ascii")
    raise ApiError(400, "image URLs must be data: URIs or http(s) URLs",
                   param=where, code="unsupported_value")


def _file_chunk(ref: ArtifactRef, store: ArtifactStore, *,
                inline_max: int, image_max: int) -> tuple[str, Optional[ResolvedImage]]:
    """The model-facing text for one file part, plus its image payload."""
    loaded = store.read(ref.id)
    if loaded is None:
        return (f"[attached: {ref.filename} ({ref.media_type}) as {ref.id}; "
                f"the stored contents are unavailable]", None)
    data = loaded[1]
    if ref.media_type.lower().startswith("image/"):
        if len(data) > image_max:
            raise ApiError(413, f"image exceeds the {image_max} byte limit",
                           code="image_too_large")
        note = (f"[attached image: {ref.filename} ({ref.media_type}, "
                f"{ref.size} bytes) as {ref.id}]")
        return note, ResolvedImage(b64=base64.b64encode(data).decode("ascii"),
                                   artifact_id=ref.id)
    if _is_textual(ref.media_type):
        text = data[:inline_max].decode("utf-8", errors="replace")
        note = ""
        if len(data) > inline_max:
            note = (f"\n[… truncated: showing the first {inline_max // 1024} "
                    f"KiB of {ref.size} bytes]")
        return f'<file name="{ref.filename}">\n{text.rstrip()}{note}\n</file>', None
    return (f"[attached: {ref.filename} ({ref.media_type}, {ref.size} bytes) "
            f"as {ref.id}; contents were not extracted]", None)


@dataclass(frozen=True)
class InputLimits:
    """Aggregate per-request input caps, from the validated `serve_max_*`."""

    max_messages: int = DEFAULT_SERVE_MAX_MESSAGES
    max_content_parts: int = DEFAULT_SERVE_MAX_CONTENT_PARTS
    max_message_bytes: int = DEFAULT_SERVE_MAX_MESSAGE_BYTES
    max_text_bytes: int = DEFAULT_SERVE_MAX_TEXT_BYTES
    max_images: int = DEFAULT_SERVE_MAX_IMAGES
    max_image_bytes: int = DEFAULT_SERVE_MAX_IMAGE_BYTES


def input_limits(config: Optional[dict[str, Any]] = None) -> InputLimits:
    """`InputLimits` from a serve config; an omitted key uses the default."""
    config = config or {}

    def _int(key: str, default: int) -> int:
        value = config.get(key)
        return default if value is None else int(value)

    return InputLimits(
        max_messages=_int("serve_max_messages", DEFAULT_SERVE_MAX_MESSAGES),
        max_content_parts=_int("serve_max_content_parts",
                               DEFAULT_SERVE_MAX_CONTENT_PARTS),
        max_message_bytes=_int("serve_max_message_bytes",
                               DEFAULT_SERVE_MAX_MESSAGE_BYTES),
        max_text_bytes=_int("serve_max_text_bytes",
                            DEFAULT_SERVE_MAX_TEXT_BYTES),
        max_images=_int("serve_max_images", DEFAULT_SERVE_MAX_IMAGES),
        max_image_bytes=_int("serve_max_image_bytes",
                             DEFAULT_SERVE_MAX_IMAGE_BYTES))


def _b64_decoded_bytes(payload: str) -> int:
    """The decoded size of a base64 payload (conservative by the padding)."""
    return (len(payload) * 3) // 4


class InputBudget:
    """One request's resolved-input accounting, shared across its messages.

    The JSON body cap bounds the request as *sent*; this bounds what
    resolving content parts (remote image fetches, artifact inlines) may
    assemble on top of it — a small body can name many large inputs. Every
    `note_*` raises the structured 413 as soon as a limit is crossed, before
    the input reaches inference (counts are checked before any fetch).
    """

    def __init__(self, limits: Optional[InputLimits] = None):
        self.limits = limits or InputLimits()
        self.messages = 0
        self.parts = 0
        self.text_bytes = 0
        self.images = 0
        self.image_bytes = 0

    @staticmethod
    def _too_large(param: str, name: str, limit: int) -> ApiError:
        return ApiError(413, f"resolved input exceeds {name} ({limit})",
                        param=param, code="input_too_large",
                        extra={"limit_name": name, "limit_value": limit})

    def note_messages(self, count: int, param: str = "messages") -> None:
        self.messages += count
        if self.messages > self.limits.max_messages:
            raise self._too_large(param, "serve_max_messages",
                                  self.limits.max_messages)

    def note_part(self, param: str) -> None:
        self.parts += 1
        if self.parts > self.limits.max_content_parts:
            raise self._too_large(param, "serve_max_content_parts",
                                  self.limits.max_content_parts)

    def note_text(self, size: int, param: str) -> None:
        if size > self.limits.max_message_bytes:
            raise self._too_large(param, "serve_max_message_bytes",
                                  self.limits.max_message_bytes)
        self.text_bytes += size
        if self.text_bytes > self.limits.max_text_bytes:
            raise self._too_large(param, "serve_max_text_bytes",
                                  self.limits.max_text_bytes)

    def note_image(self, size: int, param: str) -> None:
        self.images += 1
        if self.images > self.limits.max_images:
            raise self._too_large(param, "serve_max_images",
                                  self.limits.max_images)
        self.image_bytes += size
        if self.image_bytes > self.limits.max_image_bytes:
            raise self._too_large(param, "serve_max_image_bytes",
                                  self.limits.max_image_bytes)


def resolve_content_parts(content: Any, *, where: str,
                          store: Optional[ArtifactStore] = None,
                          inline_max: int = INLINE_TEXT_BYTES,
                          image_max: int = MAX_IMAGE_BYTES,
                          fetch: Optional[Callable[..., tuple[bytes, str]]] = None,
                          budget: Optional[InputBudget] = None
                          ) -> ResolvedContent:
    """Resolve one message's content parts into text, images and refs.

    `where` is the caller's parameter path (e.g. `messages[0].content`) so
    both surfaces name the offending part identically. Strings pass through;
    an array accepts `text`, `image_url` (data: or http(s)) and `file`
    (`file_id`), raising `ApiError` for anything else. When a `budget` is
    given, every part, image and text byte is charged to it.
    """
    if content is None:
        return ResolvedContent()
    if isinstance(content, str):
        if budget is not None:
            budget.note_text(len(content.encode("utf-8")), where)
        return ResolvedContent(text=content)
    if not isinstance(content, list):
        raise ApiError(400, "content must be a string or an array of parts",
                       param=where)
    fetch_fn = fetch or fetch_remote_media
    chunks: list[str] = []
    images: list[ResolvedImage] = []
    refs: list[ArtifactRef] = []
    for part_index, part in enumerate(content):
        at = f"{where}[{part_index}]"
        if budget is not None:
            budget.note_part(at)
        if not isinstance(part, dict):
            raise ApiError(400, "content parts must be objects", param=at)
        kind = part.get("type")
        if kind == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise ApiError(400, "text parts need a text string",
                               param=f"{at}.text")
            chunks.append(text)
        elif kind == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else None
            if not isinstance(url, str):
                raise ApiError(400, "image_url parts need a url",
                               param=f"{at}.url")
            b64 = _image_from_url(url, where=f"{at}.url",
                                  image_max=image_max, fetch=fetch_fn)
            if budget is not None:
                budget.note_image(_b64_decoded_bytes(b64), f"{at}.url")
            images.append(ResolvedImage(b64=b64))
        elif kind == "file":
            file_id = part.get("file_id")
            if not isinstance(file_id, str) or not file_id.strip():
                raise ApiError(400, "file parts need a file_id",
                               param=f"{at}.file_id")
            ref = store.resolve(file_id.strip()) if store is not None else None
            if ref is None:
                raise ApiError(
                    404, f"unknown or expired artifact: {file_id.strip()!r}",
                    param=f"{at}.file_id", code="artifact_not_found")
            refs.append(ref)
            chunk, image = _file_chunk(ref, store, inline_max=inline_max,
                                       image_max=image_max)
            chunks.append(chunk)
            if image is not None:
                if budget is not None:
                    budget.note_image(_b64_decoded_bytes(image.b64),
                                      f"{at}.file_id")
                images.append(image)
        else:
            raise ApiError(400, f"unsupported content part type: {kind!r}",
                           param=at, code="unsupported_value")
    text = "\n\n".join(c for c in chunks if c)
    if budget is not None:
        budget.note_text(len(text.encode("utf-8")), where)
    return ResolvedContent(text=text, images=images, artifacts=refs)
