"""Measured-fixture calibration: bounded, host/model/placement-scoped samples."""
from __future__ import annotations

try:  # keep the read-modify-write safe across CLI/serve processes
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

import contextlib
import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from ..helpers import (canonical_model_name, host_is_self, human_bytes,
                       is_local_host)
from ..interactive import model_block_count

try:  # keep the read-modify-write safe across CLI/serve processes
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from . import defaults
from .defaults import (CALIBRATION_MAX_AGE, CALIBRATION_MAX_ENTRIES,
                       CALIBRATION_MAX_SAMPLES, CTX_STEP)
from .machine import MachineGpu
from .models import FitRequest, GpuFit
from .profile import (_arch_key, _arch_prefix, _model_info, kv_bytes_per_token,
                      kv_element_bytes, model_total_layers,
                      trained_context_length)

_CALIBRATION_LOCKS: dict[str, threading.RLock] = {}
_CALIBRATION_LOCKS_GUARD = threading.Lock()


def _calibration_lock(path: Path) -> threading.RLock:
    key = str(path)
    with _CALIBRATION_LOCKS_GUARD:
        return _CALIBRATION_LOCKS.setdefault(key, threading.RLock())


class CtxCalibrationStore:
    """Small, bounded cache of measured memory footprints by context size."""

    def __init__(self, path: Optional[Path] = None):
        self.path = (Path(path) if path is not None
                     else defaults.CTXFIT_CALIBRATION_PATH)
        self.lock = _calibration_lock(self.path)

    @contextlib.contextmanager
    def _guard(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = None
        with self.lock:
            try:
                if fcntl is not None:
                    lock_file = open(str(self.path) + ".lock", "a+")
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                if lock_file is not None:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    finally:
                        lock_file.close()

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {"schema": 1, "entries": {}}
        if not isinstance(data, dict) or data.get("schema") != 1 \
                or not isinstance(data.get("entries"), dict):
            return {"schema": 1, "entries": {}}
        data["entries"] = {str(key): value for key, value in data["entries"].items()
                            if isinstance(value, dict)}
        return data

    def points(self, key: str) -> list[tuple[int, int]]:
        try:
            with self._guard():
                entry = self._read()["entries"].get(key) or {}
        except OSError:
            return []
        try:
            updated = float(entry.get("updated", 0))
        except (TypeError, ValueError):
            return []
        if not updated or time.time() - updated > CALIBRATION_MAX_AGE:
            return []
        points = []
        samples = entry.get("samples", [])
        if not isinstance(samples, list):
            return []
        now = time.time()
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            try:
                ctx, size = int(sample.get("ctx", 0)), int(sample.get("bytes", 0))
                sampled_at = float(sample.get("ts", updated))
            except (TypeError, ValueError):
                continue
            if ctx > 0 and size > 0 and now - sampled_at <= CALIBRATION_MAX_AGE:
                points.append((ctx, size))
        return points

    def digest_points(self, digest: str) -> int:
        """Live measured samples for a model digest, across scopes (a hint).

        Best-effort: used to tell a client that a probe starts from prior
        measurements, never to make a decision.
        """
        digest = str(digest or "")
        if not digest:
            return 0
        try:
            keys = [str(key) for key in self._read().get("entries", {})]
        except (OSError, ValueError, TypeError):
            return 0
        total = 0
        for key in keys:
            if digest in key:
                total += len(self.points(key))
        return total

    def add(self, key: str, ctx: int, size: int) -> None:
        if not key or int(ctx or 0) <= 0 or int(size or 0) <= 0:
            return
        with self._guard():
            data = self._read()
            entries = data["entries"]
            entry = entries.get(key)
            if not isinstance(entry, dict):
                entry = {"samples": []}
            samples = []
            old_samples = entry.get("samples", [])
            if isinstance(old_samples, list):
                for sample in old_samples:
                    if not isinstance(sample, dict):
                        continue
                    try:
                        old_ctx = int(sample.get("ctx", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    if old_ctx != int(ctx):
                        samples.append(sample)
            samples.append({"ctx": int(ctx), "bytes": int(size), "ts": time.time()})
            samples.sort(key=lambda sample: int(sample["ctx"]))
            entry["samples"] = samples[-CALIBRATION_MAX_SAMPLES:]
            entry["updated"] = time.time()
            entries[key] = entry
            if len(entries) > CALIBRATION_MAX_ENTRIES:
                oldest = sorted(entries, key=lambda item: float(
                    (entries[item] or {}).get("updated", 0) or 0))
                for stale in oldest[:len(entries) - CALIBRATION_MAX_ENTRIES]:
                    entries.pop(stale, None)
            self._write(data)

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp",
                                    dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            os.replace(name, self.path)
        finally:
            try:
                os.unlink(name)
            except OSError:
                pass

# llama.cpp kv-cache types -> bytes per element (block-quantized sizes).


def _measured_memory_line(points: list[tuple[int, int]], theoretical_slope: int,
                          minimum_intercept: int, *,
                          conservative: bool = True) -> Optional[tuple[int, int]]:
    """Conservative intercept/slope from measured (context, bytes) samples.

    Pairwise slopes tolerate a noisy /api/ps sample better than least squares.
    The analytic KV slope remains a floor, and the intercept is raised enough
    that the resulting line is never below any of the measurements.
    """
    clean = sorted({(int(ctx), int(size)) for ctx, size in points
                    if int(ctx) > 0 and int(size) > 0})
    if not clean:
        return None
    slopes = []
    for index, (ctx_a, size_a) in enumerate(clean):
        for ctx_b, size_b in clean[index + 1:]:
            if ctx_b - ctx_a < CTX_STEP:
                continue
            slope = (size_b - size_a) / (ctx_b - ctx_a)
            if slope > 0:
                slopes.append(slope)
    if slopes:
        slopes.sort()
        middle = len(slopes) // 2
        measured_slope = (slopes[middle] if len(slopes) % 2 else
                          (slopes[middle - 1] + slopes[middle]) / 2)
    else:
        measured_slope = float(theoretical_slope)
    slope = int(measured_slope + 0.999999)
    if conservative or not slopes:
        slope = max(max(1, int(theoretical_slope)), slope)
    else:
        slope = max(1, slope)
    intercept = max(int(minimum_intercept), max(
        int(size - slope * ctx) for ctx, size in clean))
    return slope, intercept


def _apply_empirical_gpu(fit: GpuFit, points: list[tuple[int, int]]) -> None:
    floor = fit.weights_bytes + fit.overhead_bytes
    line = _measured_memory_line(points, fit.kv_gpu_per_token, floor)
    if line is None:
        return
    slope, intercept = line
    if slope == fit.kv_gpu_per_token and intercept <= floor:
        return
    fit.kv_gpu_per_token = slope
    fit.overhead_bytes = max(fit.overhead_bytes, intercept - fit.weights_bytes)
    fit.calibrated = True
    fit.notes.append(f"calibrated from {len(points)} measured context point(s): "
                     f"{human_bytes(slope)} per token")


def _calibration_scope(req: FitRequest, *, mode: str, model_digest: str,
                       weight_bytes: int, show_data: dict[str, Any],
                       machine: Optional[MachineGpu], pinned_layers: Optional[int],
                       gpu_layers: Optional[int] = None) -> str:
    """Stable cache identity; excludes volatile free memory, includes hardware/config."""
    base = str(getattr(req.client, "base", "") or "")
    if req.ssh_target:
        host = "remote:" + req.ssh_target.strip().lower()
    elif machine and machine.source.startswith("ssh:"):
        host = "remote:" + machine.source[4:].strip().lower()
    elif host_is_self(base) or is_local_host(base):
        host = "local"
    else:
        parsed = urlsplit(base)
        host = f"{(parsed.hostname or base).lower()}:{parsed.port or 11434}"
    env = (machine.env if machine else {}) or {}
    parallel = str(env.get("OLLAMA_NUM_PARALLEL") or "1").strip()
    cache_type = str(env.get("OLLAMA_KV_CACHE_TYPE") or "f16").strip().lower()
    identity = {
        "model": canonical_model_name(req.model),
        "digest": model_digest or f"size:{int(weight_bytes or 0)}",
        "host": host,
        "mode": mode,
        "layers": pinned_layers if pinned_layers is not None else "auto",
        "gpu_layers": gpu_layers if mode == "cpu" else None,
        "gpu_totals_mib": sorted(total for total, _ in (machine.devices if machine else [])),
        "parallel": parallel,
        "kv_type": cache_type,
        "fit_target": str(env.get("LLAMA_ARG_FIT_TARGET") or ""),
        "blocks": model_block_count(show_data),
        "model_max": trained_context_length(show_data),
        "kv_bytes_per_token": kv_bytes_per_token(
            show_data,
            parallel=int(parallel) if parallel.isdigit() and int(parallel) > 0 else 1,
            element_bytes=kv_element_bytes(cache_type)),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record_calibration(req: FitRequest, ctx: int, size: int) -> None:
    if req.calibration_key and size > 0:
        try:
            CtxCalibrationStore(Path(req.calibration_path)
                                if req.calibration_path
                                else defaults.CTXFIT_CALIBRATION_PATH).add(
                                    req.calibration_key, ctx, size)
        except (OSError, TypeError, ValueError):
            pass


def _kv_fraction_on_gpu(show_data: dict[str, Any], pinned_layers: int) -> float:
    """Share of the KV cache that stays on the GPU with `pinned_layers` offloaded.

    llama.cpp fills GPU layers from the top (the output layer first, then the
    last blocks), so with a per-block `head_count_kv` list only the attention
    blocks in that tail keep their cache on the GPU; dense models (a scalar
    head count) scale with the block fraction.
    """
    blocks = model_block_count(show_data)
    total_layers = model_total_layers(show_data)
    if not blocks or not total_layers or pinned_layers >= total_layers:
        return 1.0
    offloaded_blocks = max(0, min(blocks, pinned_layers - 1))
    info = _model_info(show_data)
    arch = _arch_prefix(info)
    heads = _arch_key(info, arch, "attention.head_count_kv")
    if isinstance(heads, list) and len(heads) >= blocks:
        total_heads = sum(value for value in heads
                          if isinstance(value, (int, float)) and value > 0)
        tail_heads = sum(value for value in heads[blocks - offloaded_blocks:]
                         if isinstance(value, (int, float)) and value > 0)
        if total_heads:
            return tail_heads / total_heads
    if not offloaded_blocks:
        return 0.0
    return offloaded_blocks / blocks


