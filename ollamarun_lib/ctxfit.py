"""Context-window sizing: `/num-ctx --max-gpu [--probe]` and
`/num-ctx --max-cpu N [--probe]`.

`--max-gpu` looks for the largest `num_ctx` that still keeps the model's GPU
placement (the whole model, or the pinned `/gpu-layers`), so the next load
runs at the chosen context without a partial CPU offload.

`--max-cpu N` raises the context until the model's host RAM reaches N% of the
DRAM available to it: `/proc/meminfo`'s MemAvailable (or one ssh probe for
`--remote` hosts) plus the RAM the loaded model already holds -- read back
from `/api/ps` as `size - size_vram`, since it is released when the model
reloads at the new context -- less an overhead allowance
(`max_cpu_overhead_mb`). Some layers usually stay on the GPU: a pinned
`/gpu-layers` is honoured, otherwise llama.cpp's fit picks the split, and
only the host side counts toward the target.

The estimate starts from sources that are actually deterministic:

- The KV-cache size per token is exact for standard attention layers, computed
  from `/api/show` GGUF metadata with llama.cpp's own formula
  (`2 * kv_heads * (key_length + value_length) * element_bytes` per layer),
  including per-layer `head_count_kv` arrays from hybrid models that only
  carry a KV cache on some blocks, `key_length`/`value_length` fallbacks to
  `embedding_length / head_count`, `OLLAMA_NUM_PARALLEL` sequence slices and
  `OLLAMA_KV_CACHE_TYPE` element sizes.
- Weights come from `/api/tags` (`size`, the on-disk blob total, so vision
  projectors are counted), cross-checked against the model's parameter count
  (`general.parameter_count`, or `/api/tags` `parameter_size`) and
  quantization (`details.quantization_level` or the GGUF `file_type`): a tags
  size far from `params × bytes/param` is called out and the larger figure
  used. `ModelProfile` also splits the weights into token embeddings, the
  output layer and repeating blocks when the vocabulary/embedding geometry is
  known, so a pinned `/gpu-layers` offloads the output layer and the topmost
  blocks -- not an even fraction.
- Per-layer KV geometry covers more than the dense case: scalar or per-layer
  `head_count_kv` (hybrid attention/recurrent models), per-layer key/value
  lengths with their sliding-window variants, `sliding_window_pattern`/
  `layer_types` (sliding layers' cache grows only to `window + batch`
  cells), and MLA models that store one compressed `kv_lora_rank` latent
  plus the rope tail per layer. Recurrent layers (`ssm.*` state, or
  lfm2-style `shortconv`) get their fixed per-sequence state sized instead.
- Free VRAM comes from `nvidia-smi` -- locally, or one BatchMode ssh probe
  for `--remote` machines -- together with the ollama unit's environment
  (`LLAMA_ARG_FIT_TARGET` is llama.cpp's fit margin, 1 GiB per device by
  default); free DRAM comes from `/proc/meminfo` the same way.

What the estimate cannot see is llama.cpp's measured buffers, so `--probe`
verifies empirically: it loads the model at a candidate context and reads
`/api/ps` back -- for `--max-gpu` the split (`size` vs `size_vram` -> the
real GPU percentage), for `--max-cpu` the host footprint (`size - size_vram`,
which matches what the machine's MemAvailable gave up in steady state).
Candidates are walked up or down -- re-fitting the measured bytes-per-token
line between probes -- until the largest 100%-GPU context is bracketed, or
the largest context whose host RAM stays at or below the fill target is
found. Valid measurements are kept in a bounded cache at
`~/etc/data/ollamarun/ctxfit-calibration.json`, scoped to the model digest,
host, hardware, placement and KV settings. Later estimates use a conservative
measured slope/intercept; current resident measurements can refine the fixed
overhead, and the metadata formula remains the fallback. Cache entries expire
after 180 days.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from rich import box
from rich.markup import escape
from rich.table import Table

from .client import OllamaError
from .constants import DATA_DIR, console
from .helpers import (
    canonical_model_name,
    host_is_self,
    human_bytes,
    is_local_host,
    local_gpu_probe,
    local_ram_probe,
    ssh_gpu_probe,
    ssh_ram_probe,
)
from .interactive import model_block_count, model_parameter_count, resolve_gpu_layers

MIB = 1024 * 1024

DEFAULT_MARGIN_MB = 1024      # llama.cpp's --fit-target default, per device
DEFAULT_OVERHEAD_MB = 0       # extra reserve beyond the fit margin (llama.cpp's
                              # own fit projection already counts its buffers)
DEFAULT_CPU_OVERHEAD_MB = 256  # CPU-side buffers the --max-cpu estimate can't read
DEFAULT_PROBES = 10
DEFAULT_CPU_PROBES = 10
CTX_STEP = 256                # llama.cpp aligns fit-reduced context to 256 * streams
ESTIMATE_STEP = 1024          # the estimate rounds down to readable numbers
SWA_CACHE_BATCH = 1024        # cells llama.cpp adds past a sliding window for batching
CTXFIT_CALIBRATION_PATH = DATA_DIR / "ctxfit-calibration.json"
CALIBRATION_MAX_AGE = 180 * 24 * 60 * 60
CALIBRATION_MAX_ENTRIES = 256
CALIBRATION_MAX_SAMPLES = 16

try:  # keep the read-modify-write safe across CLI/serve processes
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

_CALIBRATION_LOCKS: dict[str, threading.RLock] = {}
_CALIBRATION_LOCKS_GUARD = threading.Lock()


def _calibration_lock(path: Path) -> threading.RLock:
    key = str(path)
    with _CALIBRATION_LOCKS_GUARD:
        return _CALIBRATION_LOCKS.setdefault(key, threading.RLock())


class CtxCalibrationStore:
    """Small, bounded cache of measured memory footprints by context size."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else CTXFIT_CALIBRATION_PATH
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
KV_ELEMENT_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,   # 34 bytes / 32 values
    "q5_1": 0.75,     # 24 / 32
    "q5_0": 0.6875,   # 22 / 32
    "q4_1": 0.625,    # 20 / 32
    "q4_0": 0.5625,   # 18 / 32
}

# `general.file_type` (llama.cpp's llama_ftype enum) -> quantization label.
FILE_TYPE_LABELS = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4", 39: "NVFP4",
}

# Typical stored bytes per parameter: block-quantized tensors plus overhead.
QUANT_BYTES_PER_PARAM = {
    "F32": 4.0, "F16": 2.0, "BF16": 2.0, "FP8": 1.0, "FP16": 2.0,
    "Q8_0": 1.0625, "Q6_K": 0.8203, "Q5_K_M": 0.70, "Q5_K_S": 0.68,
    "Q5_1": 0.75, "Q5_0": 0.6875, "Q4_K_M": 0.61, "Q4_K_S": 0.58,
    "Q4_1": 0.625, "Q4_0": 0.5625, "Q3_K_L": 0.54, "Q3_K_M": 0.50,
    "Q3_K_S": 0.44, "Q2_K": 0.33, "IQ4_XS": 0.545, "IQ4_NL": 0.5625,
    "IQ3_M": 0.50, "IQ3_S": 0.44, "IQ3_XS": 0.42, "IQ3_XXS": 0.36,
    "IQ2_M": 0.33, "IQ2_S": 0.30, "IQ2_XS": 0.28, "IQ2_XXS": 0.25,
    "IQ1_M": 0.23, "IQ1_S": 0.19, "MXFP4": 0.53, "NVFP4": 0.53,
    "Q2_K_S": 0.30, "TQ1_0": 0.19, "TQ2_0": 0.21,
}


def quantization_label(show_data: dict[str, Any],
                       entry: Optional[dict[str, Any]] = None) -> str:
    """Human label for a model's quantization (tags first, GGUF enum after)."""
    details = (entry or {}).get("details") or {}
    label = str(details.get("quantization_level") or "").strip().upper()
    if label:
        return label
    file_type = _model_info(show_data).get("general.file_type")
    if isinstance(file_type, int):
        return FILE_TYPE_LABELS.get(file_type, f"file_type {file_type}")
    return ""


def quantization_bytes_per_param(label: str) -> Optional[float]:
    """Typical stored bytes per parameter for a quantization label, or None."""
    text = (label or "").strip().upper().replace("-", "_")
    if text in QUANT_BYTES_PER_PARAM:
        return QUANT_BYTES_PER_PARAM[text]
    for prefix in sorted(QUANT_BYTES_PER_PARAM, key=len, reverse=True):
        if text.startswith(prefix):
            return QUANT_BYTES_PER_PARAM[prefix]
    return None


NUM_CTX_HELP = """usage: /num-ctx N | --max-gpu [--probe] | --max-cpu N [--probe]
  N          set the context window (capped to the model's max)
  --max-gpu  use the largest context that still runs the model on the GPU:
             per-layer KV geometry from /api/show (hybrid layers, sliding
             windows, MLA compression and recurrent state included), the
             on-disk size (/api/tags) cross-checked against the model's
             parameter count and quantization, and the machine's free VRAM
             (an nvidia-smi probe, locally or over ssh for --remote hosts),
             minus llama.cpp's fit margin (LLAMA_ARG_FIT_TARGET, or
             max_gpu_margin_mb; it already covers the buffers the fit
             projects, so the extra max_gpu_overhead_mb reserve defaults to
             0). Models already resident are credited
             back -- their VRAM is released when they reload at the new
             context; max_gpu_credit_residents=false limits that credit to
             the model being sized. A pinned /gpu-layers is honoured, with
             the output layer and the topmost blocks (not an even split)
             counted as the offloaded share.
  --max-cpu N  raise the context until the model's host RAM reaches N% (1-100)
             of the DRAM available to it: /proc/meminfo's MemAvailable -- one
             ssh probe for --remote hosts -- plus the RAM the loaded model
             already holds (/api/ps size - size_vram; a reload returns it),
             less an overhead allowance (max_cpu_overhead_mb). Some layers
             stay on the GPU unless /gpu-layers is cpu; a pinned /gpu-layers
             is honoured and only the host side counts toward the target.
             The per-layer KV and recurrent-state model above applies here
             too.
  --probe    verify by loading: starting from the estimate, the model is
             warmed at candidate contexts and the real reading is taken back
             from /api/ps -- the GPU split for --max-gpu, the host RAM for
             --max-cpu -- until the boundary is bracketed
             (max_gpu_probes / max_cpu_probes, default 10). Successful probes
             calibrate later estimates; compatible cached measurements are
             reused, otherwise the profile estimate is used."""


# ── machine state (free VRAM + the ollama unit env) ─────────────────────────

@dataclass
class MachineGpu:
    """GPU totals/free (MiB per device), the ollama unit env, and where they came from."""
    devices: list[tuple[int, int]] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    source: str = ""            # "local" or "ssh:HOST"
    error: str = ""


def probe_machine_gpu(*, ssh_target: Optional[str] = None,
                      ssh_user: Optional[str] = None) -> MachineGpu:
    """Read free VRAM from this machine or an ssh host (never raises)."""
    if ssh_target:
        got = ssh_gpu_probe(ssh_target, user=ssh_user)
        if got is None:
            return MachineGpu(source=f"ssh:{ssh_target}",
                              error=f"no GPU data from ssh {ssh_target}")
        devices, env = got
        return MachineGpu(devices=devices, env=env, source=f"ssh:{ssh_target}")
    devices, env = local_gpu_probe()
    if not devices:
        return MachineGpu(env=env, source="local", error="no nvidia-smi GPU data")
    return MachineGpu(devices=devices, env=env, source="local")


def machine_state_for(client_base: str, *, ssh_target: Optional[str] = None,
                      ssh_user: Optional[str] = None) -> Optional[MachineGpu]:
    """MachineGpu for a client's base URL: local when it is this machine, ssh
    for --remote hosts, and None when neither applies."""
    if client_base and host_is_self(client_base):
        return probe_machine_gpu()
    if ssh_target:
        return probe_machine_gpu(ssh_target=ssh_target, ssh_user=ssh_user)
    if is_local_host(client_base or ""):
        return probe_machine_gpu()
    return None


# ── machine state (available DRAM for the --max-cpu fill) ───────────────────

@dataclass
class MachineRam:
    """DRAM totals/available in bytes, and where they came from."""
    total_bytes: int = 0
    available_bytes: int = 0    # /proc/meminfo's MemAvailable
    source: str = ""            # "local" or "ssh:HOST"
    error: str = ""


def probe_machine_ram(*, ssh_target: Optional[str] = None,
                      ssh_user: Optional[str] = None) -> MachineRam:
    """Read DRAM from this machine or an ssh host (never raises)."""
    if ssh_target:
        got = ssh_ram_probe(ssh_target, user=ssh_user)
        if got is None:
            return MachineRam(source=f"ssh:{ssh_target}",
                              error=f"no memory data from ssh {ssh_target}")
        total, available = got
        return MachineRam(total_bytes=total, available_bytes=available,
                          source=f"ssh:{ssh_target}")
    got = local_ram_probe()
    if got is None:
        return MachineRam(source="local", error="cannot read /proc/meminfo")
    total, available = got
    return MachineRam(total_bytes=total, available_bytes=available, source="local")


def machine_ram_for(client_base: str, *, ssh_target: Optional[str] = None,
                    ssh_user: Optional[str] = None) -> Optional[MachineRam]:
    """MachineRam for a client's base URL: local when it is this machine, ssh
    for --remote hosts, and None when neither applies."""
    if client_base and host_is_self(client_base):
        return probe_machine_ram()
    if ssh_target:
        return probe_machine_ram(ssh_target=ssh_target, ssh_user=ssh_user)
    if is_local_host(client_base or ""):
        return probe_machine_ram()
    return None


# ── GGUF metadata -> KV cache size ──────────────────────────────────────────

def _model_info(show_data: dict[str, Any]) -> dict[str, Any]:
    return show_data.get("model_info") or {}


def _arch_prefix(info: dict[str, Any]) -> str:
    """The architecture prefix of the GGUF metadata keys ('' for bare names)."""
    for key in info:
        if key == "block_count":
            return ""
        if key.endswith(".block_count"):
            return key[: -len(".block_count")]
    return ""


def _arch_key(info: dict[str, Any], arch: str, suffix: str) -> Any:
    return info.get(f"{arch}.{suffix}" if arch else suffix)


def trained_context_length(show_data: dict[str, Any]) -> Optional[int]:
    """The model's trained context window (the real cap), ignoring parameter overrides."""
    for key, value in _model_info(show_data).items():
        if key.endswith(".context_length") and isinstance(value, int) and value > 0:
            return value
    return None


def model_total_layers(show_data: dict[str, Any]) -> Optional[int]:
    """Layers as `num_gpu` counts them: the repeating blocks plus the output layer."""
    blocks = model_block_count(show_data)
    return blocks + 1 if blocks else None


def kv_element_bytes(cache_type: Optional[str]) -> float:
    """Bytes per KV element for an OLLAMA_KV_CACHE_TYPE (f16 when unrecognized/unset)."""
    return KV_ELEMENT_BYTES.get((cache_type or "").strip().lower(), 2.0)


@dataclass
class KvLayerSpec:
    """One repeating block's cache geometry."""
    heads: int
    key_length: float
    value_length: float
    window: Optional[int] = None
    mla: bool = False

    def bytes_per_token(self, element_bytes: float) -> int:
        if self.mla:
            return max(1, int(round((self.key_length + self.value_length)
                                    * element_bytes)))
        if self.heads <= 0:
            return 0
        return max(1, int(round(self.heads * (self.key_length + self.value_length)
                                * element_bytes)))

    def cells(self, ctx: int, batch: int = SWA_CACHE_BATCH) -> int:
        if not self.window:
            return ctx
        return min(ctx, self.window + max(1, batch))


def _as_list(value: Any, blocks: int) -> Optional[list[Any]]:
    """A per-layer metadata list (or a scalar repeated for every block)."""
    if isinstance(value, list):
        if not value:
            return None
        return [value[index] if index < len(value) else value[-1]
                for index in range(blocks)]
    if isinstance(value, (int, float, bool)):
        return [value] * blocks
    return None


def _swa_flags(info: dict[str, Any], arch: str, blocks: int) -> tuple[list[bool], str]:
    """Per-block sliding-window flags plus a short description of the source."""
    layer_types = _arch_key(info, arch, "attention.layer_types")
    if isinstance(layer_types, list) and layer_types:
        flags = []
        for index in range(blocks):
            value = str(layer_types[index] if index < len(layer_types)
                        else layer_types[-1]).lower()
            flags.append("slid" in value)
        return flags, "layer_types"
    pattern = _arch_key(info, arch, "attention.sliding_window_pattern")
    if isinstance(pattern, list) and pattern:
        flags = []
        for index in range(blocks):
            value = pattern[index] if index < len(pattern) else pattern[-1]
            if isinstance(value, str):
                flags.append("slid" in value.lower())
            else:
                flags.append(bool(value))
        return flags, "sliding_window_pattern"
    if isinstance(pattern, int) and pattern > 1:
        # Every Nth layer is full attention; the rest use the window.
        return [((index + 1) % pattern != 0) for index in range(blocks)], \
            f"every {pattern}th layer full"
    if arch == "gemma3":
        return [((index + 1) % 6 != 0) for index in range(blocks)], \
            "gemma3 every 6th layer full"
    if arch == "gemma2":
        return [((index + 1) % 2 != 0) for index in range(blocks)], \
            "gemma2 every 2nd layer full"
    window = _arch_key(info, arch, "attention.sliding_window")
    return ([bool(window)] * blocks if _positive(window) else [False] * blocks), \
        ("all layers" if _positive(window) else "")


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def _vocab_size(info: dict[str, Any], arch: str) -> Optional[int]:
    """Vocabulary size from the GGUF: the explicit key, tokens, or token ids.

    Ollama elides the tokenizer arrays from `/api/show`, so models without an
    explicit `<arch>.vocab_size` still expose their special-token ids; the
    largest id + 1 is a close lower bound for the embedding/output rows.
    """
    explicit = _arch_key(info, arch, "vocab_size")
    if _positive(explicit):
        return int(explicit)
    tokens = info.get("tokenizer.ggml.tokens")
    if isinstance(tokens, list) and tokens:
        return len(tokens)
    ids = [int(value) for key, value in info.items()
           if key.startswith("tokenizer.ggml.") and key.endswith("_token_id")
           and isinstance(value, (int, float)) and not isinstance(value, bool)
           and value >= 0]
    if not ids:
        return None
    # sentencepiece models expose only tiny special ids here; a real
    # vocabulary starts well above them.
    guess = max(ids) + 1
    return guess if guess >= 1024 else None


def _layer_kv_specs(show_data: dict[str, Any]) -> tuple[Optional[list[KvLayerSpec]], str]:
    """Per-block KV geometry, or (None, reason) when the metadata cannot size it."""
    info = _model_info(show_data)
    arch = _arch_prefix(info)
    blocks = model_block_count(show_data)
    if not blocks or not arch:
        return None, "block count or architecture missing"
    if _arch_key(info, arch, "attention.causal") is False:
        return None, "the model has no causal attention KV cache"
    heads_raw = _arch_key(info, arch, "attention.head_count_kv")
    if heads_raw is None:
        # A missing GQA field means plain multi-head attention: every head
        # keeps a K/V pair. More useful than refusing to size the model.
        heads_raw = _arch_key(info, arch, "attention.head_count")
    if heads_raw is None and (
            _positive(_arch_key(info, arch, "ssm.state_size"))
            or _positive(_arch_key(info, arch, "shortconv.l_cache"))):
        heads_raw = 0  # pure recurrent/convolution model: no KV cache
    heads = _as_list(heads_raw, blocks)
    if heads is None:
        return None, "attention head count missing"
    try:
        heads = [max(0, int(value)) for value in heads
                 if isinstance(value, (int, float))]
    except (TypeError, ValueError):
        return None, "attention head count is not numeric"
    if len(heads) < blocks:
        return None, "attention head count list is short"
    heads = heads[:blocks]

    embedding = _arch_key(info, arch, "embedding_length")
    head_count = _arch_key(info, arch, "attention.head_count")
    key = _arch_key(info, arch, "attention.key_length")
    value = _arch_key(info, arch, "attention.value_length")
    key_swa = _arch_key(info, arch, "attention.key_length_swa")
    value_swa = _arch_key(info, arch, "attention.value_length_swa")
    key_default = key if _positive(key) else None
    value_default = value if _positive(value) else key_default
    mla = False
    if not key_default:
        # MLA caches one compressed latent plus a rope tail per layer; this
        # must win over the embedding/head-count fallback.
        lora = _arch_key(info, arch, "attention.kv_lora_rank")
        if _positive(lora):
            key_default = lora
            value_default = _arch_key(info, arch, "attention.qk_rope_head_dim")
            if not _positive(value_default):
                value_default = 0
            mla = True
    if not key_default and _positive(embedding) and _positive(head_count):
        key_default = value_default = embedding / head_count
    if not key_default:
        return None, "attention key/value dimensions missing"
    if not mla and not _positive(value_default):
        value_default = key_default

    window = _arch_key(info, arch, "attention.sliding_window")
    window = int(window) if _positive(window) else None
    swa, swa_source = _swa_flags(info, arch, blocks)
    key_swa = key_swa if _positive(key_swa) else key_default
    value_swa = value_swa if _positive(value_swa) else value_default

    specs: list[KvLayerSpec] = []
    for index in range(blocks):
        sliding = bool(window) and swa[index]
        specs.append(KvLayerSpec(
            heads=heads[index],
            key_length=float(key_swa if sliding else key_default),
            value_length=float(value_swa if sliding else value_default),
            window=window if sliding else None,
            mla=mla,
        ))
    note = ""
    if window:
        count = sum(1 for spec in specs if spec.window)
        note = (f"{count}/{blocks} layers use a {window:,}-token sliding window"
                f" ({swa_source}); the cache keeps window + batch cells")
    return specs, note


def kv_bytes_per_token(show_data: dict[str, Any], *, parallel: int = 1,
                       element_bytes: float = 2.0) -> Optional[int]:
    """KV-cache bytes per context token at full context, or None when unknown."""
    specs, _ = _layer_kv_specs(show_data)
    if not specs:
        return None
    per_token = sum(spec.bytes_per_token(element_bytes) for spec in specs)
    if not per_token:
        return None
    return max(1, per_token) * max(1, int(parallel))


def kv_bytes_at(show_data: dict[str, Any], ctx: int, *, parallel: int = 1,
                element_bytes: float = 2.0, batch: int = SWA_CACHE_BATCH,
                block_range: Optional[tuple[int, int]] = None) -> Optional[int]:
    """Exact KV-cache bytes at `ctx`, accounting for saturated sliding windows."""
    specs, _ = _layer_kv_specs(show_data)
    if not specs:
        return None
    start, end = block_range or (0, len(specs))
    start = max(0, min(len(specs), start))
    end = max(start, min(len(specs), end))
    total = sum(spec.cells(max(0, int(ctx)), batch)
                * spec.bytes_per_token(element_bytes)
                for spec in specs[start:end])
    return max(0, total) * max(1, int(parallel))


def _recurrent_bytes(show_data: dict[str, Any], *, parallel: int = 1) -> tuple[int, int, str]:
    """(bytes per sequence, recurrent layer count, note) for SSM/hybrid models."""
    info = _model_info(show_data)
    arch = _arch_prefix(info)
    blocks = model_block_count(show_data) or 0
    specs, _ = _layer_kv_specs(show_data)
    if not specs:
        return 0, 0, ""
    recurrent_layers = sum(1 for spec in specs if spec.heads <= 0)
    if not recurrent_layers:
        return 0, 0, ""
    inner = _arch_key(info, arch, "ssm.inner_size")
    state = _arch_key(info, arch, "ssm.state_size")
    conv = _arch_key(info, arch, "ssm.conv_kernel")
    if _positive(inner) and _positive(state):
        conv = int(conv) if _positive(conv) else 1
        # f32 recurrent state, plus the short-convolution state llama.cpp
        # keeps alongside it. The observed qwen35 layout is S = inner * state
        # per layer and R = inner * kernel per layer (rounded up by runtime).
        per_layer = int(inner) * int(state) * 4 + int(inner) * max(1, conv) * 4
        total = per_layer * recurrent_layers * max(1, int(parallel))
        return total, recurrent_layers, (
            f"{recurrent_layers}/{blocks} recurrent layers "
            f"(SSM state {human_bytes(total)})")
    shortconv = _arch_key(info, arch, "shortconv.l_cache")
    embedding = _arch_key(info, arch, "embedding_length")
    if _positive(shortconv) and _positive(embedding):
        # lfm2-style short convolutions: one f32 cache per channel.
        per_layer = int(embedding) * max(1, int(shortconv)) * 4
        total = per_layer * recurrent_layers * max(1, int(parallel))
        return total, recurrent_layers, (
            f"{recurrent_layers}/{blocks} short-convolution layers "
            f"(state {human_bytes(total)})")
    return 0, recurrent_layers, (
        f"{recurrent_layers} non-attention layers have no state metadata; "
        f"their fixed state is not sized")


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
                                if req.calibration_path else CTXFIT_CALIBRATION_PATH).add(
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


# ── the model profile (GGUF + /api/tags facts behind the estimate) ──────────

@dataclass
class ModelProfile:
    """Everything the estimate can learn about the model before sizing it."""
    architecture: str = ""
    blocks: Optional[int] = None
    embedding_length: Optional[int] = None
    feed_forward_length: Optional[int] = None
    context_length: Optional[int] = None
    vocab_size: Optional[int] = None
    expert_count: Optional[int] = None
    expert_used: Optional[int] = None
    params: Optional[int] = None
    params_source: str = ""
    quantization: str = ""
    weights_bytes: int = 0
    weights_source: str = ""
    projector_bytes: int = 0
    projector_params: Optional[int] = None
    tied_embeddings: bool = False
    embedding_bytes: int = 0
    output_bytes: int = 0
    block_bytes: int = 0
    split_estimate: bool = False
    kv_layers: list[KvLayerSpec] = field(default_factory=list)
    kv_note: str = ""
    recurrent_bytes: int = 0          # per sequence, all recurrent layers
    recurrent_layers: int = 0
    recurrent_note: str = ""
    notes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def total_layers(self) -> Optional[int]:
        return self.blocks + 1 if self.blocks else None

    def _recurrent_per_layer(self) -> int:
        return self.recurrent_bytes // self.recurrent_layers \
            if self.recurrent_layers else 0

    def kv_bytes(self, ctx: int, *, element_bytes: float = 2.0, parallel: int = 1,
                 batch: int = SWA_CACHE_BATCH,
                 block_range: Optional[tuple[int, int]] = None) -> int:
        if not self.kv_layers:
            return 0
        start, end = block_range or (0, len(self.kv_layers))
        start = max(0, min(len(self.kv_layers), start))
        end = max(start, min(len(self.kv_layers), end))
        total = sum(spec.cells(max(0, int(ctx)), batch)
                    * spec.bytes_per_token(element_bytes)
                    for spec in self.kv_layers[start:end])
        return max(0, total) * max(1, int(parallel))

    def kv_slope(self, *, element_bytes: float = 2.0, parallel: int = 1,
                 block_range: Optional[tuple[int, int]] = None,
                 asymptotic: bool = True) -> int:
        """Bytes per extra token: every layer, or only the non-window ones."""
        if not self.kv_layers:
            return 0
        start, end = block_range or (0, len(self.kv_layers))
        start = max(0, min(len(self.kv_layers), start))
        end = max(start, min(len(self.kv_layers), end))
        total = sum(spec.bytes_per_token(element_bytes)
                    for spec in self.kv_layers[start:end]
                    if not asymptotic or not spec.window)
        return max(0, total) * max(1, int(parallel))

    def gpu_weight_share(self, layers: int) -> tuple[int, int]:
        """(GPU, CPU) weight bytes for `layers` offloaded (output layer counted)."""
        total_layers = self.total_layers or 0
        layers = max(0, min(total_layers, int(layers)))
        if layers >= total_layers:
            return self.weights_bytes, 0   # full offload keeps every weight
        if not self.split_estimate or not self.blocks:
            gpu = self.weights_bytes * layers // total_layers if total_layers else 0
            return gpu, self.weights_bytes - gpu
        # The output layer is offloaded first, then blocks from the top down.
        offloaded_blocks = max(0, min(self.blocks, layers - 1))
        gpu = (self.output_bytes if layers >= 1 else 0) \
            + int(self.block_bytes * offloaded_blocks / self.blocks)
        return min(gpu, self.weights_bytes), max(0, self.weights_bytes - gpu)

    def recurrent_share(self, layers: int) -> tuple[int, int]:
        """(GPU, CPU) recurrent-state bytes for `layers` offloaded (output counted)."""
        total_layers = self.total_layers or 0
        layers = max(0, min(total_layers, int(layers)))
        if not self.recurrent_layers or not self.blocks:
            return 0, self.recurrent_bytes
        offloaded_blocks = max(0, min(self.blocks, layers - 1))
        per_layer = self._recurrent_per_layer()
        gpu_layers = sum(1 for spec in self.kv_layers[-offloaded_blocks:]
                         if spec.heads <= 0) if offloaded_blocks else 0
        gpu = per_layer * gpu_layers
        return gpu, self.recurrent_bytes - gpu

    def kv_block_range(self, layers: int) -> Optional[tuple[int, int]]:
        """The repeating blocks on the GPU for `layers` offloaded layers."""
        if not self.blocks:
            return None
        if layers >= self.blocks + 1:
            return (0, self.blocks)
        offloaded = max(0, min(self.blocks, layers - 1))
        return (self.blocks - offloaded, self.blocks)

    def summary(self) -> str:
        bits = []
        if self.params:
            bits.append(f"{self.params / 1e9:.2f}B params")
        if self.quantization:
            bits.append(self.quantization)
        if self.architecture:
            bits.append(self.architecture)
        if self.weights_bytes:
            bits.append(human_bytes(self.weights_bytes))
        return " · ".join(bits)


def _quantization_for_size(size_bytes: int, params: Optional[int]) -> str:
    """The quantization label whose bytes/param best explains a file size."""
    if not size_bytes or not params:
        return ""
    ratio = size_bytes / params
    best = min(QUANT_BYTES_PER_PARAM.items(),
               key=lambda item: abs(item[1] - ratio))
    return best[0] if abs(best[1] - ratio) / max(ratio, 1e-9) < 0.2 else ""


def model_profile(show_data: dict[str, Any],
                  entry: Optional[dict[str, Any]] = None) -> ModelProfile:
    """Fold GGUF metadata and the /api/tags entry into one sizing profile."""
    info = _model_info(show_data)
    arch = _arch_prefix(info)
    details = (entry or {}).get("details") or {}
    profile = ModelProfile(
        architecture=arch,
        blocks=model_block_count(show_data),
        embedding_length=_int_or_none(_arch_key(info, arch, "embedding_length")
                                      or details.get("embedding_length")),
        feed_forward_length=_int_or_none(_arch_key(info, arch, "feed_forward_length")),
        context_length=_int_or_none(trained_context_length(show_data)
                                    or details.get("context_length")),
        vocab_size=_vocab_size(info, arch),
        expert_count=_int_or_none(_arch_key(info, arch, "expert_count")),
        expert_used=_int_or_none(_arch_key(info, arch, "expert_used_count")),
    )
    params = _int_or_none(info.get("general.parameter_count"))
    if params:
        profile.params_source = "GGUF general.parameter_count"
    else:
        billions = model_parameter_count(entry or {})
        if billions:
            params = int(billions * 1e9)
            profile.params_source = "/api/tags parameter_size"
    profile.params = params

    profile.quantization = quantization_label(show_data, entry)
    per_param = quantization_bytes_per_param(profile.quantization)
    size = _int_or_none((entry or {}).get("size")) or 0
    expected = int(params * per_param) if params and per_param else 0
    if entry and (entry.get("remote_model") or entry.get("remote_host")):
        profile.notes.append("the tag points at a remote/cloud model — "
                             "weights cannot be sized locally")
    elif size and expected and size < expected * 3 // 4:
        implied = _quantization_for_size(size, params)
        detail = f" (the file size matches {implied})" if implied else ""
        profile.notes.append(
            f"tags size {human_bytes(size)} is well under the "
            f"{human_bytes(expected)} implied by {profile.quantization} × "
            f"{params / 1e9:.2f}B params{detail} — using the larger figure")
        profile.weights_bytes = expected
        profile.weights_source = "parameters × quantization"
    elif size:
        profile.weights_bytes = size
        profile.weights_source = "/api/tags size"
        if expected and size > expected * 3 // 2:
            implied = _quantization_for_size(size, params)
            detail = f" (the file size matches {implied})" if implied else ""
            profile.notes.append(
                f"tags size {human_bytes(size)} is well above the "
                f"{human_bytes(expected)} implied by {profile.quantization} × "
                f"{params / 1e9:.2f}B params{detail} — using the disk size")
    elif expected:
        profile.weights_bytes = expected
        profile.weights_source = f"{params / 1e9:.2f}B params × {profile.quantization}"

    projector = show_data.get("projector_info") or {}
    projector_params = _int_or_none(projector.get("general.parameter_count"))
    if projector_params:
        profile.projector_params = projector_params
        file_type = projector.get("general.file_type")
        label = FILE_TYPE_LABELS.get(file_type) if isinstance(file_type, int) else ""
        projector_bpp = quantization_bytes_per_param(label) or per_param
        if projector_bpp:
            profile.projector_bytes = min(profile.weights_bytes,
                                          int(projector_params * projector_bpp))
        profile.notes.append(
            f"vision projector: {projector_params / 1e6:.0f}M params "
            f"≈ {human_bytes(profile.projector_bytes)}")

    tied = _arch_key(info, arch, "tie_word_embeddings")
    if isinstance(tied, bool):
        profile.tied_embeddings = tied
    main_params = params - (profile.projector_params or 0) if params else None
    main_bytes = max(0, profile.weights_bytes - profile.projector_bytes)
    if (main_params and main_params > 0 and main_bytes
            and profile.vocab_size and profile.embedding_length):
        bpp = main_bytes / main_params
        embed_params = profile.vocab_size * profile.embedding_length
        output_params = 0 if profile.tied_embeddings else embed_params
        if embed_params + output_params < main_params:
            profile.embedding_bytes = int(embed_params * bpp)
            profile.output_bytes = int(output_params * bpp)
            profile.block_bytes = max(
                0, main_bytes - profile.embedding_bytes - profile.output_bytes)
            profile.split_estimate = True
            shape = "tied" if profile.tied_embeddings else "separate"
            profile.notes.append(
                f"weights split: {human_bytes(profile.embedding_bytes)} token "
                f"embeddings + {human_bytes(profile.output_bytes)} output layer "
                f"({shape}) + {human_bytes(profile.block_bytes)} blocks")

    specs, kv_note = _layer_kv_specs(show_data)
    if specs:
        profile.kv_layers = specs
        profile.kv_note = kv_note
    else:
        profile.error = kv_note or "cannot size the KV cache"
    recurrent, recurrent_layers, recurrent_note = _recurrent_bytes(show_data)
    profile.recurrent_bytes = recurrent
    profile.recurrent_layers = recurrent_layers
    profile.recurrent_note = recurrent_note
    return profile


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


# ── the estimate ────────────────────────────────────────────────────────────

@dataclass
class GpuFit:
    """The estimate: inputs, what goes on the GPU, and the resulting context."""
    ctx: int = 0
    feasible: bool = False
    kv_per_token: int = 0
    kv_gpu_per_token: int = 0
    weights_bytes: int = 0          # GPU share of the weights
    weights_total: int = 0
    usable_bytes: int = 0
    free_bytes: int = 0
    margin_bytes: int = 0           # per device
    overhead_bytes: int = 0
    credited_bytes: int = 0         # resident models' VRAM credited back (size_vram sum)
    residents: list[tuple[str, int]] = field(default_factory=list)
    ctx_no_margin: Optional[int] = None     # what would fit without the margin
    model_max: Optional[int] = None
    capped_by_model: bool = False
    pinned_layers: Optional[int] = None
    total_layers: Optional[int] = None
    parallel: int = 1
    element_bytes: float = 2.0
    devices: list[tuple[int, int]] = field(default_factory=list)
    source: str = ""
    profile: Optional[ModelProfile] = None
    kv_bytes_at_ctx: int = 0
    kv_bytes_total_at_ctx: int = 0
    recurrent_bytes: int = 0
    recurrent_gpu_bytes: int = 0
    weights_source: str = ""
    calibrated: bool = False
    notes: list[str] = field(default_factory=list)
    error: str = ""


def _config_int(config: dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(config.get(key))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def estimate_max_gpu_ctx(show_data: dict[str, Any], weight_bytes: int,
                         machine: MachineGpu, *, pinned_layers: Optional[int] = None,
                         config: Optional[dict[str, Any]] = None,
                         residents: Optional[list[tuple[str, int]]] = None,
                         empirical_points: Optional[list[tuple[int, int]]] = None,
                         profile: Optional[ModelProfile] = None) -> GpuFit:
    """Largest context that keeps the placement on the GPU (no loads, pure math).

    `residents` are the models already holding VRAM as (name, size_vram): the
    free-VRAM reading then excludes what their own instances hold, but that
    memory is released when they reload at the new context, so it is credited
    back into the pool instead of being counted against the placement.

    The per-token cost comes from the model profile: per-layer KV geometry
    (hybrid layers included), saturated sliding windows, MLA compression and
    recurrent state, cross-checked against the model's parameters and
    quantization. `empirical_points` (measured `/api/ps` footprints) still win
    when they exist, because they carry llama.cpp's measured buffers.
    """
    config = config or {}
    profile = profile or model_profile(show_data)
    weights_total = max(0, int(weight_bytes or profile.weights_bytes or 0))
    fit = GpuFit(weights_total=weights_total,
                 devices=list(machine.devices), source=machine.source,
                 pinned_layers=pinned_layers,
                 total_layers=profile.total_layers or model_total_layers(show_data),
                 model_max=profile.context_length or trained_context_length(show_data),
                 profile=profile, weights_source=profile.weights_source,
                 recurrent_bytes=profile.recurrent_bytes)
    if residents:
        fit.residents = [(str(name), int(size)) for name, size in residents
                         if int(size or 0) > 0]
        fit.credited_bytes = sum(size for _, size in fit.residents)
    if not machine.devices:
        fit.error = machine.error or "no GPU memory data"
        return fit

    env = machine.env or {}
    parallel = str(env.get("OLLAMA_NUM_PARALLEL") or "").strip()
    fit.parallel = int(parallel) if parallel.isdigit() and int(parallel) > 1 else 1
    fit.element_bytes = kv_element_bytes(env.get("OLLAMA_KV_CACHE_TYPE"))
    if fit.element_bytes != 2.0:
        fit.notes.append(f"KV cache type {env.get('OLLAMA_KV_CACHE_TYPE')} "
                         f"({fit.element_bytes:g} B per element)")
    if fit.parallel > 1:
        fit.notes.append(f"OLLAMA_NUM_PARALLEL={fit.parallel}: the KV cache holds "
                         f"{fit.parallel} sequences")

    margin_mb = _config_int(config, "max_gpu_margin_mb", DEFAULT_MARGIN_MB)
    target = str(env.get("LLAMA_ARG_FIT_TARGET") or "").strip()
    if target.isdigit():
        margin_mb = int(target)
        fit.notes.append(f"fit margin {margin_mb} MiB per device (LLAMA_ARG_FIT_TARGET)")
    fit.margin_bytes = margin_mb * MIB
    fit.overhead_bytes = _config_int(config, "max_gpu_overhead_mb",
                                     DEFAULT_OVERHEAD_MB) * MIB
    fit.free_bytes = sum(free for _, free in machine.devices) * MIB
    fit.usable_bytes = sum(max(0, free * MIB - fit.margin_bytes)
                           for _, free in machine.devices) + fit.credited_bytes
    for name, size in fit.residents:
        fit.notes.append(f"crediting {name} ({human_bytes(size)} on the GPU) — "
                         f"it reloads at the new context and releases that memory")

    fit.kv_per_token = profile.kv_slope(element_bytes=fit.element_bytes,
                                        parallel=fit.parallel) or 0
    if not fit.kv_per_token:
        fit.kv_per_token = kv_bytes_per_token(show_data, parallel=fit.parallel,
                                              element_bytes=fit.element_bytes) or 0
    if not fit.kv_per_token and not profile.recurrent_bytes:
        fit.error = ("cannot size the KV cache from /api/show "
                     "(attention metadata missing for this architecture)")
        return fit

    if pinned_layers is None:
        fit.weights_bytes = weights_total
        gpu_layers = fit.total_layers or 0
        block_range = (0, profile.blocks) if profile.blocks else None
        fit.kv_gpu_per_token = fit.kv_per_token
        fit.notes.append("all layers are placed on the GPU")
    else:
        total = fit.total_layers or 0
        gpu_layers = max(0, min(total, pinned_layers))
        if profile.split_estimate:
            fit.weights_bytes, _ = profile.gpu_weight_share(gpu_layers)
            fit.weights_source = "non-block weights split by layer"
        else:
            fit.weights_bytes = weights_total * gpu_layers // total if total else weights_total
        block_range = profile.kv_block_range(gpu_layers)
        tail = profile.kv_slope(element_bytes=fit.element_bytes,
                                parallel=fit.parallel, block_range=block_range)
        fit.kv_gpu_per_token = max(1, tail) if tail else raw_kv_share(
            fit.kv_per_token, gpu_layers, total)
        where = f"gpu-layers {gpu_layers}" + (f" of {total}" if total else "")
        fit.notes.append(f"{where} pinned: the KV cache for the offloaded tail "
                         f"stays on the GPU")
    fit.recurrent_gpu_bytes, _ = profile.recurrent_share(gpu_layers)
    fit.kv_bytes_total_at_ctx = profile.kv_bytes(
        0, element_bytes=fit.element_bytes, parallel=fit.parallel)

    if empirical_points:
        _apply_empirical_gpu(fit, empirical_points)

    def kv_gpu_bytes(at_ctx: int) -> int:
        if fit.calibrated:
            return max(0, fit.kv_gpu_per_token * at_ctx)
        measured = profile.kv_bytes(at_ctx, element_bytes=fit.element_bytes,
                                    parallel=fit.parallel, block_range=block_range)
        return measured if measured is not None else max(0, fit.kv_gpu_per_token * at_ctx)

    def solve_ctx(budget: int) -> tuple[int, bool]:
        """Largest context <= the model max whose KV fits `budget` (capped flag)."""
        if budget <= 0:
            return 0, False
        cap = fit.model_max or (1 << 20)
        if kv_gpu_bytes(cap) <= budget:
            return cap, True
        lo, hi = 0, max(CTX_STEP, _align_down(cap, CTX_STEP))
        while hi - lo > CTX_STEP:
            mid = _align_down((lo + hi) // 2, CTX_STEP)
            if mid <= lo:
                break
            if kv_gpu_bytes(mid) <= budget:
                lo = mid
            else:
                hi = mid
        return lo, False

    fixed = fit.weights_bytes + fit.overhead_bytes + fit.recurrent_gpu_bytes
    ctx, capped = solve_ctx(fit.usable_bytes - fixed)
    fit.capped_by_model = bool(fit.model_max) and capped
    if ctx >= ESTIMATE_STEP:
        ctx = _align_down(ctx, ESTIMATE_STEP)
    fit.ctx = max(0, int(ctx))
    fit.feasible = fit.ctx > 0
    fit.kv_bytes_at_ctx = kv_gpu_bytes(fit.ctx)
    fit.kv_bytes_total_at_ctx = profile.kv_bytes(
        fit.ctx, element_bytes=fit.element_bytes,
        parallel=fit.parallel) or fit.kv_bytes_at_ctx
    # What the same placement would fit with no margin at all: printed as a
    # hint when the per-device reserve is the only thing making it infeasible.
    usable_unc = (sum(max(0, free * MIB) for _, free in machine.devices)
                  + fit.credited_bytes)
    unc, _ = solve_ctx(usable_unc - fixed)
    if unc >= ESTIMATE_STEP:
        unc = _align_down(unc, ESTIMATE_STEP)
    fit.ctx_no_margin = int(unc) if unc else None
    fit.notes.extend(profile.notes)
    if profile.kv_note:
        fit.notes.append(profile.kv_note)
    if profile.recurrent_note:
        fit.notes.append(profile.recurrent_note)
    if not fit.feasible:
        usable = human_bytes(fit.usable_bytes) if fit.usable_bytes else "0 B"
        if fit.credited_bytes:
            usable += (f" ({human_bytes(fit.free_bytes)} free + "
                       f"{human_bytes(fit.credited_bytes)} resident)")
        extra = ""
        if fit.recurrent_gpu_bytes:
            extra += f" plus {human_bytes(fit.recurrent_gpu_bytes)} of recurrent state"
        if fit.overhead_bytes:
            extra += f" plus {human_bytes(fit.overhead_bytes)} of buffers"
        fit.error = (f"the GPU placement does not fit: {human_bytes(fit.weights_bytes)} "
                     f"of weights{extra} vs {usable} usable after the "
                     f"{human_bytes(fit.margin_bytes)} margin")
    return fit


def raw_kv_share(kv_per_token: int, gpu_layers: int, total_layers: int) -> int:
    """Fallback KV share for a pinned placement without per-layer metadata."""
    if not kv_per_token or not total_layers:
        return kv_per_token
    offloaded = max(0, min(total_layers, gpu_layers) - 1)
    return max(1, kv_per_token * offloaded // max(1, total_layers - 1))


# ── the CPU-side estimate (fill N% of the DRAM available to the model) ──────

@dataclass
class CpuFit:
    """The estimate: the RAM budget, the placement, and the resulting context."""
    ctx: int = 0
    feasible: bool = False
    percent: int = 0
    total_bytes: int = 0                # MemTotal
    available_bytes: int = 0            # MemAvailable as measured
    resident_bytes: int = 0             # the model's host RAM already in /api/ps
    budget_bytes: int = 0               # available + resident
    target_bytes: int = 0               # percent% of the budget
    footprint_bytes: int = 0            # the estimate's host RAM at ctx
    weights_total: int = 0
    cpu_weights: int = 0
    kv_per_token: int = 0
    cpu_kv_per_token: int = 0
    overhead_bytes: int = 0             # CPU buffers the estimate cannot read
    gpu_overhead_bytes: int = 0         # the GPU reserve the placement fit uses
    gpu_layers: Optional[int] = None    # layers llama.cpp's fit should keep on the GPU
    total_layers: Optional[int] = None
    pinned_layers: Optional[int] = None
    model_max: Optional[int] = None
    budget_ctx: Optional[int] = None    # where the analytic footprint hits the budget
    capped_by_model: bool = False
    devices: list[tuple[int, int]] = field(default_factory=list)
    free_bytes: int = 0
    margin_bytes: int = 0
    usable_bytes: int = 0
    source: str = ""
    profile: Optional[ModelProfile] = None
    cpu_kv_bytes: int = 0
    recurrent_bytes: int = 0
    recurrent_cpu_bytes: int = 0
    weights_source: str = ""
    calibrated: bool = False
    notes: list[str] = field(default_factory=list)
    error: str = ""


def estimate_max_cpu_ctx(show_data: dict[str, Any], weight_bytes: int, ram: MachineRam, *,
                         percent: int, resident_bytes: int = 0,
                         machine: Optional[MachineGpu] = None,
                         pinned_layers: Optional[int] = None,
                         config: Optional[dict[str, Any]] = None,
                         empirical_points: Optional[list[tuple[int, int]]] = None,
                         profile: Optional[ModelProfile] = None) -> CpuFit:
    """The context whose host RAM reaches `percent`% of the DRAM available to it.

    The budget is MemAvailable plus the RAM the loaded model already holds
    (it comes back when the model reloads at the new context); other
    programs and models keep theirs. The weights and the per-token KV of the
    CPU-side layers are the deterministic inputs -- what llama.cpp's fit
    keeps on the GPU is simulated from free VRAM unless `/gpu-layers` is
    pinned.
    """
    config = config or {}
    profile = profile or model_profile(show_data)
    weights_total = max(0, int(weight_bytes or profile.weights_bytes or 0))
    fit = CpuFit(
        percent=max(1, min(100, int(percent))),
        total_bytes=max(0, int(ram.total_bytes or 0)),
        available_bytes=max(0, int(ram.available_bytes or 0)),
        resident_bytes=max(0, int(resident_bytes or 0)),
        weights_total=weights_total,
        pinned_layers=pinned_layers,
        total_layers=profile.total_layers or model_total_layers(show_data),
        model_max=profile.context_length or trained_context_length(show_data),
        source=ram.source,
        profile=profile,
        weights_source=profile.weights_source,
        recurrent_bytes=profile.recurrent_bytes,
    )
    fit.budget_bytes = fit.available_bytes + fit.resident_bytes
    fit.target_bytes = fit.budget_bytes * fit.percent // 100
    if not fit.available_bytes:
        fit.error = ram.error or "cannot read the machine's free memory"
        return fit
    if not fit.weights_total:
        fit.error = "cannot size the model (/api/tags has no entry)"
        return fit
    if not fit.total_layers:
        fit.error = "cannot size the model's layers (GGUF block_count missing)"
        return fit

    env = (machine.env if machine else {}) or {}
    parallel = str(env.get("OLLAMA_NUM_PARALLEL") or "").strip()
    sequences = int(parallel) if parallel.isdigit() and int(parallel) > 1 else 1
    element = kv_element_bytes(env.get("OLLAMA_KV_CACHE_TYPE"))
    if element != 2.0:
        fit.notes.append(f"KV cache type {env.get('OLLAMA_KV_CACHE_TYPE')} "
                         f"({element:g} B per element)")
    if sequences > 1:
        fit.notes.append(f"OLLAMA_NUM_PARALLEL={sequences}: the KV cache holds "
                         f"{sequences} sequences")
    fit.kv_per_token = profile.kv_slope(element_bytes=element,
                                        parallel=sequences) or 0
    if not fit.kv_per_token:
        fit.kv_per_token = kv_bytes_per_token(show_data, parallel=sequences,
                                              element_bytes=element) or 0
    if not fit.kv_per_token and not profile.recurrent_bytes:
        fit.error = ("cannot size the KV cache from /api/show "
                     "(attention metadata missing for this architecture)")
        return fit

    fit.overhead_bytes = _config_int(config, "max_cpu_overhead_mb",
                                     DEFAULT_CPU_OVERHEAD_MB) * MIB
    fit.gpu_overhead_bytes = _config_int(config, "max_gpu_overhead_mb",
                                         DEFAULT_OVERHEAD_MB) * MIB
    margin_mb = _config_int(config, "max_gpu_margin_mb", DEFAULT_MARGIN_MB)
    margin_env = str(env.get("LLAMA_ARG_FIT_TARGET") or "").strip()
    if margin_env.isdigit():
        margin_mb = int(margin_env)
        fit.notes.append(f"fit margin {margin_mb} MiB per device (LLAMA_ARG_FIT_TARGET)")
    fit.margin_bytes = margin_mb * MIB
    if machine and machine.devices:
        fit.devices = list(machine.devices)
        fit.free_bytes = sum(free for _, free in machine.devices) * MIB
        fit.usable_bytes = sum(max(0, free * MIB - fit.margin_bytes)
                               for _, free in machine.devices)
    else:
        fit.notes.append("no GPU memory data: assuming every layer stays on the CPU")
    if fit.resident_bytes:
        fit.notes.append(f"the loaded model holds {human_bytes(fit.resident_bytes)} of "
                         f"host RAM -- counted back into the budget, a reload returns it")

    empirical_cpu_kv: Optional[int] = None

    def placement(at_ctx: int) -> tuple[Optional[int], int, int, int, int]:
        """(gpu layers, CPU weights, CPU KV bytes, CPU KV per token, CPU SSM).

        llama.cpp fills GPU layers from the top (the output layer first, then
        the last blocks) while weights, the tail's per-layer KV and its
        recurrent state fit the usable VRAM; the rest stays on the CPU.
        """
        total = fit.total_layers or 0

        def gpu_kv_bytes(layers: int) -> int:
            if not profile.kv_layers:
                return int(fit.kv_per_token * _kv_fraction_on_gpu(show_data, layers)
                           * at_ctx)
            return profile.kv_bytes(at_ctx, element_bytes=element,
                                    parallel=sequences,
                                    block_range=profile.kv_block_range(layers))

        def gpu_usage(layers: int) -> int:
            if profile.split_estimate:
                weights, _ = profile.gpu_weight_share(layers)
            else:
                weights = fit.weights_total * layers // total if total else 0
            recurrent_gpu, _ = profile.recurrent_share(layers)
            return (weights + gpu_kv_bytes(layers) + recurrent_gpu
                    + fit.gpu_overhead_bytes)

        if fit.pinned_layers is not None:
            layers = max(0, min(total, fit.pinned_layers))
        elif fit.usable_bytes:
            layers = total
            while layers > 0 and gpu_usage(layers) > fit.usable_bytes:
                layers -= 1
        else:
            layers = 0
        if profile.split_estimate:
            _, cpu_weights = profile.gpu_weight_share(layers)
        else:
            cpu_weights = fit.weights_total * (total - layers) // total if total else 0
        gpu_range = profile.kv_block_range(layers) if profile.blocks else None
        if profile.kv_layers and profile.blocks:
            start = gpu_range[0] if gpu_range else profile.blocks
            cpu_kv = profile.kv_bytes(at_ctx, element_bytes=element,
                                      parallel=sequences,
                                      block_range=(0, start))
            cpu_slope = profile.kv_slope(element_bytes=element,
                                         parallel=sequences,
                                         block_range=(0, start))
        else:
            gpu_kv = gpu_kv_bytes(layers)
            cpu_kv = max(0, fit.kv_per_token * at_ctx - gpu_kv)
            cpu_slope = max(0, fit.kv_per_token - (
                gpu_kv // at_ctx if at_ctx else 0))
        if empirical_cpu_kv is not None:
            cpu_kv = max(cpu_kv, empirical_cpu_kv * at_ctx)
            cpu_slope = max(cpu_slope, empirical_cpu_kv)
        _, recurrent_cpu = profile.recurrent_share(layers)
        return layers, cpu_weights, cpu_kv, cpu_slope, recurrent_cpu

    if empirical_points:
        _, base_cpu_weights, _, base_cpu_kv, _ = placement(CTX_STEP)
        line = _measured_memory_line(
            empirical_points, base_cpu_kv,
            base_cpu_weights + fit.overhead_bytes)
        if line is not None:
            empirical_cpu_kv, intercept = line
            fit.overhead_bytes = max(fit.overhead_bytes,
                                     intercept - base_cpu_weights)
            fit.calibrated = True
            fit.notes.append(f"calibrated from {len(empirical_points)} measured "
                             f"context point(s): {human_bytes(empirical_cpu_kv)} per token")


    def footprint(at_ctx: int) -> int:
        _, weights, kv, _, recurrent = placement(at_ctx)
        return weights + kv + recurrent + fit.overhead_bytes

    def solve(ceiling: int, limit: int) -> Optional[int]:
        """Largest context <= ceiling whose host footprint stays <= limit."""
        if footprint(CTX_STEP) > limit:
            return None
        if ceiling <= CTX_STEP or footprint(ceiling) <= limit:
            return max(1, int(ceiling))
        lo = CTX_STEP
        hi = max(lo, _align_down(int(ceiling), CTX_STEP))
        while hi - lo > CTX_STEP:
            mid = _align_down((lo + hi) // 2, CTX_STEP)
            if mid <= lo:
                break
            if footprint(mid) <= limit:
                lo = mid
            else:
                hi = mid
        return lo

    ceiling: Optional[int] = fit.model_max
    if ceiling is None:
        ceiling = CTX_STEP
        while footprint(ceiling) <= fit.target_bytes and ceiling < (1 << 20):
            ceiling *= 2
        ceiling = min(ceiling, 1 << 20)
        fit.notes.append(f"the model's max context is unknown -- search capped at "
                         f"{ceiling:,} tokens")

    ctx = solve(ceiling, fit.target_bytes)
    if ctx is None:
        floor_fp = footprint(CTX_STEP)
        (fit.gpu_layers, fit.cpu_weights, fit.cpu_kv_bytes,
         fit.cpu_kv_per_token, fit.recurrent_cpu_bytes) = placement(CTX_STEP)
        fit.footprint_bytes = floor_fp
        fit.error = (f"the model already holds {human_bytes(floor_fp)} at num_ctx "
                     f"{CTX_STEP:,}, past the {human_bytes(fit.target_bytes)} "
                     f"{fit.percent}% target -- raise the percentage")
        return fit
    fit.capped_by_model = bool(fit.model_max) and ctx >= fit.model_max
    if ctx >= ESTIMATE_STEP:
        ctx = _align_down(ctx, ESTIMATE_STEP)
    fit.ctx = max(0, int(ctx))
    fit.feasible = fit.ctx > 0
    (fit.gpu_layers, fit.cpu_weights, fit.cpu_kv_bytes,
     fit.cpu_kv_per_token, fit.recurrent_cpu_bytes) = placement(fit.ctx)
    fit.footprint_bytes = footprint(fit.ctx)
    fit.budget_ctx = solve(ceiling, fit.budget_bytes) if fit.budget_bytes else None
    fit.notes.extend(profile.notes)
    if profile.kv_note:
        fit.notes.append(profile.kv_note)
    if profile.recurrent_note:
        fit.notes.append(profile.recurrent_note)
    if fit.capped_by_model and not fit.cpu_kv_per_token:
        fit.notes.append("the whole model fits on the GPU: the host footprint cannot "
                         "grow -- lower /gpu-layers to fill DRAM")
    return fit


# ── reports ─────────────────────────────────────────────────────────────────

def print_fit_report(fit: GpuFit, *, model: str, host_label: Optional[str] = None,
                     probe: bool = False) -> None:
    where = f" on {host_label}" if host_label else ""
    table = Table(title=f"context fit · {escape(model)}{escape(where)}",
                  box=box.ROUNDED, header_style="bold magenta", show_header=False)
    table.add_column("field", style="dim")
    table.add_column("value")
    if len(fit.devices) > 1:
        each = ", ".join(f"{human_bytes(free * MIB)} free" for _, free in fit.devices)
        memory = f"{human_bytes(fit.free_bytes)} free ({len(fit.devices)} GPUs: {each})"
    else:
        memory = f"{human_bytes(fit.free_bytes)} free"
    if fit.credited_bytes:
        memory += (f" + {human_bytes(fit.credited_bytes)} resident → "
                   f"{human_bytes(fit.free_bytes + fit.credited_bytes)} pool")
    if fit.profile and fit.profile.summary():
        table.add_row("model", escape(fit.profile.summary()))
    table.add_row("gpu memory", memory)
    table.add_row("margin", f"{human_bytes(fit.margin_bytes)} per device → "
                            f"{human_bytes(fit.usable_bytes)} usable")
    weights = human_bytes(fit.weights_bytes)
    if fit.pinned_layers is not None:
        weights += f" of {human_bytes(fit.weights_total)} (pinned layers)"
    if fit.weights_source and fit.weights_source != "/api/tags size":
        weights += f" [dim]({escape(fit.weights_source)})[/dim]"
    table.add_row("weights", weights)
    kv_line = f"{human_bytes(fit.kv_gpu_per_token)} per token"
    if fit.ctx:
        kv_bytes = fit.kv_bytes_at_ctx or fit.kv_gpu_per_token * fit.ctx
        kv_line += f" · {human_bytes(kv_bytes)} at {fit.ctx:,}"
    if fit.kv_gpu_per_token != fit.kv_per_token:
        kv_line += f" (of {human_bytes(fit.kv_per_token)} total)"
    table.add_row("kv cache", kv_line)
    if fit.recurrent_gpu_bytes:
        table.add_row("recurrent", f"{human_bytes(fit.recurrent_gpu_bytes)} "
                                   f"SSM state on the GPU")
    if fit.overhead_bytes:
        table.add_row("overhead", f"{human_bytes(fit.overhead_bytes)} reserved")
    if fit.model_max:
        table.add_row("model max", f"{fit.model_max:,} tokens")
    console.print(table)
    for note in fit.notes:
        console.print(f"[dim]· {escape(note)}[/dim]")
    if not fit.feasible:
        return
    capped = " [dim](capped by the model's max context)[/dim]" if fit.capped_by_model else ""
    console.print(f"[ok]num_ctx {fit.ctx:,}[/ok]{capped}")
    if not probe:
        console.print("[dim]estimate only — /num-ctx --max-gpu --probe loads the model "
                      "and verifies the real GPU split[/dim]")


def print_cpu_fit_report(fit: CpuFit, *, model: str, host_label: Optional[str] = None,
                         probe: bool = False) -> None:
    where = f" on {host_label}" if host_label else ""
    table = Table(title=f"context fit · {escape(model)}{escape(where)}",
                  box=box.ROUNDED, header_style="bold magenta", show_header=False)
    table.add_column("field", style="dim")
    table.add_column("value")
    if fit.profile and fit.profile.summary():
        table.add_row("model", escape(fit.profile.summary()))
    table.add_row("machine memory", f"{human_bytes(fit.total_bytes)} total · "
                                    f"{human_bytes(fit.available_bytes)} available")
    if fit.resident_bytes:
        table.add_row("model now", f"{human_bytes(fit.resident_bytes)} of host RAM "
                                   f"(back on reload)")
        table.add_row("budget", f"{human_bytes(fit.budget_bytes)} = available + the "
                                f"resident model")
    else:
        table.add_row("budget", human_bytes(fit.budget_bytes))
    table.add_row("target", f"{human_bytes(fit.target_bytes)} = {fit.percent}% of the budget")
    if fit.pinned_layers is None and fit.gpu_layers is None:
        placement = "every layer on the CPU (no GPU data)"
    elif fit.pinned_layers is None:
        placement = f"gpu-layers ~{fit.gpu_layers} of {fit.total_layers} (fitted)"
    else:
        placement = f"gpu-layers {fit.pinned_layers} of {fit.total_layers} (pinned)"
    table.add_row("placement", placement)
    cpu_weights = f"{human_bytes(fit.cpu_weights)} of {human_bytes(fit.weights_total)}"
    if fit.weights_source and fit.weights_source != "/api/tags size":
        cpu_weights += f" [dim]({escape(fit.weights_source)})[/dim]"
    table.add_row("cpu weights", cpu_weights)
    kv_line = f"{human_bytes(fit.cpu_kv_per_token)} per token"
    if fit.ctx:
        kv_bytes = fit.cpu_kv_bytes or fit.cpu_kv_per_token * fit.ctx
        kv_line += f" · {human_bytes(kv_bytes)} at {fit.ctx:,}"
    if fit.cpu_kv_per_token != fit.kv_per_token:
        kv_line += f" (of {human_bytes(fit.kv_per_token)} total)"
    table.add_row("kv cache", kv_line)
    if fit.recurrent_cpu_bytes:
        table.add_row("recurrent", f"{human_bytes(fit.recurrent_cpu_bytes)} "
                                   f"SSM state on the host")
    table.add_row("overhead", f"{human_bytes(fit.overhead_bytes)} reserved")
    if fit.model_max:
        table.add_row("model max", f"{fit.model_max:,} tokens")
    console.print(table)
    for note in fit.notes:
        console.print(f"[dim]· {escape(note)}[/dim]")
    if not fit.feasible:
        return
    capped = " [dim](capped by the model's max context)[/dim]" if fit.capped_by_model else ""
    console.print(f"[ok]num_ctx {fit.ctx:,}[/ok]{capped}")
    if fit.capped_by_model and fit.budget_bytes:
        reached = fit.footprint_bytes * 100 // fit.budget_bytes
        console.print(f"[dim]· at the model's max the host holds "
                      f"{human_bytes(fit.footprint_bytes)} = {reached}% of the budget "
                      f"({fit.percent}% requested)[/dim]")
    if not probe:
        console.print("[dim]estimate only — /num-ctx --max-cpu --probe loads the model "
                      "and verifies the real host RAM[/dim]")


# ── the probe (verify by loading) ───────────────────────────────────────────

@dataclass
class Probe:
    ctx: int
    total: int = 0
    vram: int = 0
    pct: int = 0
    host: int = 0                       # size - size_vram (the --max-cpu measure)
    failed: str = ""


@dataclass
class ProbeResult:
    ctx: Optional[int] = None
    probes: list[Probe] = field(default_factory=list)
    capped_by_model: bool = False
    reason: str = ""
    target_bytes: int = 0               # --max-cpu: the host-RAM fill target
    budget_bytes: int = 0               # --max-cpu: the budget it is a share of


@dataclass
class FitRequest:
    """Everything the estimate/probe needs about the target machine and model."""
    client: Any
    model: str
    config: dict[str, Any] = field(default_factory=dict)
    host_label: Optional[str] = None
    ssh_target: Optional[str] = None
    ssh_user: Optional[str] = None
    pinned_spec: Optional[str] = None       # session gpu-layers ("50%", "all", ...)
    num_gpu: Optional[int] = None           # absolute num_gpu fallback
    base_options: Optional[dict[str, Any]] = None
    keepalive: Optional[str] = None
    machine: Optional[MachineGpu] = None    # injectable (tests / reuse)
    refresh_gpu: Optional[Callable[[], Optional[MachineGpu]]] = None
    ram: Optional[MachineRam] = None        # injectable (tests / reuse)
    calibration_key: str = ""
    calibration_path: Optional[str] = None


def _align_down(value: int, step: int) -> int:
    return (value // step) * step


def _device_budget(machine: Optional[MachineGpu], margin_bytes: int) -> Optional[int]:
    """Usable bytes across devices (free minus the per-device margin); None when unknown."""
    if not machine or not machine.devices:
        return None
    return sum(max(0, free * MIB - margin_bytes) for _, free in machine.devices)


def _refresh_machine(req: FitRequest, fit: GpuFit) -> None:
    if req.refresh_gpu is None:
        return
    fresh = req.refresh_gpu()
    if fresh is not None and fresh.devices:
        req.machine = fresh


def _wait_for_ps(client: Any, model: str, ctx: int,
                 *, timeout: float = 8.0) -> Optional[dict[str, Any]]:
    """The /api/ps entry for `model` once it reports the requested context."""
    wanted = canonical_model_name(model)
    deadline = time.monotonic() + timeout
    while True:
        try:
            for entry in client.ps():
                name = entry.get("name") or entry.get("model") or ""
                if canonical_model_name(name) != wanted:
                    continue
                seen = entry.get("context_length")
                if seen in (None, ctx):
                    return entry
        except OllamaError:
            return None
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


def probe_max_gpu_ctx(req: FitRequest, fit: GpuFit) -> ProbeResult:
    """Load the model at candidate contexts until the 100%-GPU boundary is bracketed.

    Candidates are placed in *footprint space*: the size a full offload would
    report is affine in the context (llama.cpp's own fit interpolates exactly
    this way), so every 100% probe pins the line to `footprint = a + b * ctx`
    and every partial-load or OOM probe caps the capacity below that point's
    footprint. The next candidate is *solved* from that line for the current
    capacity ceiling -- not a bisection midpoint -- so a good first estimate
    needs only a probe or two to bracket the boundary. Only the *sign* of a
    partial probe is trusted: with layers on the CPU the reported size
    double-counts buffers, so its byte counts would mislead the search.
    """
    result = ProbeResult()
    base_options = dict(req.base_options or {})
    if fit.pinned_layers is not None:
        base_options["num_gpu"] = fit.pinned_layers
    probes_max = max(1, _config_int(req.config, "max_gpu_probes", DEFAULT_PROBES))
    step = CTX_STEP
    slope = float(fit.kv_gpu_per_token or fit.kv_per_token or step)
    ctx = max(step, fit.ctx)
    best: Optional[int] = None
    low_bad: Optional[int] = None
    # The capacity in footprint bytes: everything <= k_lo fit, nothing > k_hi did.
    # Both bounds start empty and only measured, full-offload probes move them --
    # the analytic sizes can overshoot the real footprint (vision projectors),
    # which would poison a byte-space solve.
    k_lo = 0.0
    k_hi: Optional[float] = None
    store = CtxCalibrationStore(Path(req.calibration_path)
                                if req.calibration_path else None)
    good: list[tuple[int, int]] = (store.points(req.calibration_key)
                                   if req.calibration_key else [])
    # Cached points seed the measured line, but never the live success/failure
    # bracket: other resident models may have changed the available VRAM.
    tested = 0
    seen: set[int] = set()
    last_partial = False

    def measured_line() -> Optional[tuple[float, float]]:
        """(intercept, bytes per token) through the measured full-offload points.

        A conservative pairwise slope through cached and live measurements
        gives the next candidate directly; the budget bracket still moves only
        from live probes because resident VRAM can change between runs.
        """
        if not good:
            return None
        line = _measured_memory_line(good, int(max(1, slope)), 0,
                                     conservative=False)
        return (float(line[1]), float(line[0])) if line else None

    def footprint(at_ctx: int) -> Optional[float]:
        fitted = measured_line()
        return None if fitted is None else fitted[0] + fitted[1] * at_ctx

    def solved_ctx(k_bytes: float) -> Optional[int]:
        """The context whose predicted footprint hits `k_bytes`, or None."""
        fitted = measured_line()
        if fitted is None or fitted[1] <= 0:
            return None
        return _align_down(max(step, int((k_bytes - fitted[0]) / fitted[1])), step)

    def tighten_ceiling(k_bytes: Optional[float] = None) -> None:
        nonlocal k_hi
        if k_bytes is not None and k_bytes > k_lo:
            k_hi = k_bytes if k_hi is None else min(k_hi, k_bytes)
        budget = _device_budget(req.machine, fit.margin_bytes)
        if budget is not None and budget > k_lo:
            k_hi = float(budget) if k_hi is None else min(k_hi, float(budget))

    while tested < probes_max:
        # Unload first so the free-VRAM reading is not taken while the model
        # still holds its previous context (a reload would evict it anyway),
        # and give the driver a moment to actually release the memory.
        try:
            req.client.unload(req.model)
        except OllamaError:
            pass
        if req.refresh_gpu is not None:
            time.sleep(0.4)  # let the driver release the previous VRAM
            _refresh_machine(req, fit)
        seen.add(ctx)
        console.print(f"[dim]probe {tested + 1}: {escape(req.model)} at "
                      f"num_ctx {ctx:,}…[/dim]")
        try:
            req.client.load(req.model, keepalive=req.keepalive,
                            options={**base_options, "num_ctx": ctx})
        except OllamaError as exc:
            result.probes.append(Probe(ctx=ctx, failed=str(exc)[:200]))
            low_bad = ctx if low_bad is None else min(low_bad, ctx)
            tighten_ceiling(footprint(ctx))
            last_partial = False
            tested += 1
        else:
            entry = _wait_for_ps(req.client, req.model, ctx)
            if entry is None or not int(entry.get("size") or 0):
                result.reason = ("could not read the loaded model back from "
                                 "/api/ps (stale or missing entry)")
                break
            total = int(entry.get("size") or 0)
            vram = int(entry.get("size_vram") or 0)
            pct = vram * 100 // total if total else 0
            result.probes.append(Probe(ctx=ctx, total=total, vram=vram, pct=pct))
            last_partial = pct < 100
            tested += 1
            if pct >= 100:
                _record_calibration(req, ctx, vram)
                if best is None or ctx > best:
                    best = ctx
                good.append((ctx, total))
                k_lo = max(k_lo, float(total))
                fitted = measured_line() if len(good) >= 2 else None
                if fitted is not None and fitted[1] > 0:
                    slope = fitted[1]
                # Nothing failed yet? Use the estimated ceiling (free VRAM
                # minus the fit margin) as the first bound and jump at it.
                tighten_ceiling()
                if fit.model_max and ctx >= fit.model_max:
                    result.capped_by_model = True
                    break
            else:
                low_bad = ctx if low_bad is None else min(low_bad, ctx)
                tighten_ceiling(footprint(ctx))
        # ── next candidate: solve the measured line for the ceiling ─────────
        if best is not None and low_bad is not None and low_bad - best <= step:
            break
        if best is None:
            nxt = _align_down(int(ctx * 0.5), step)      # nothing fit yet: halve
        elif k_hi is None:
            nxt = ctx + step                              # no ceiling known
        else:
            if k_hi - k_lo <= slope * step:
                break                                     # converged in bytes
            nxt = solved_ctx(k_hi)
            if nxt is None or nxt == ctx \
                    or (low_bad is not None and nxt >= low_bad - step):
                # A partial load means the boundary is just below it: try
                # that step before halving. A hard load failure says nothing
                # about where the boundary is, so bisect the byte interval.
                near = (low_bad - step
                        if last_partial and low_bad is not None else None)
                if near is not None and near > (best or 0) and near not in seen:
                    nxt = _align_down(near, step)
                else:
                    nxt = solved_ctx((k_lo + k_hi) / 2)
            if nxt is None or nxt == ctx:
                nxt = _align_down(int((best + (low_bad or fit.model_max
                                                or ctx + 2 * step)) / 2), step)
        if nxt is None:
            break
        if best is not None:
            nxt = max(nxt, best + step)
        if low_bad is not None:
            nxt = min(nxt, low_bad - step)
        if fit.model_max and nxt > fit.model_max:
            nxt = fit.model_max
        if nxt < step:
            nxt = step
        if nxt == ctx or (fit.model_max and ctx >= fit.model_max) or nxt <= 0:
            break
        ctx = nxt
    result.ctx = best
    if best is not None:
        successes = [probe for probe in result.probes if not probe.failed and probe.pct >= 100]
        if successes and successes[-1].ctx != best:
            try:  # leave the model warm at the chosen context
                req.client.load(req.model, keepalive=req.keepalive,
                                options={**base_options, "num_ctx": best})
            except OllamaError:
                pass
    elif not result.reason:
        measured = [probe for probe in result.probes if not probe.failed and probe.total]
        if measured:
            top = max(measured, key=lambda probe: probe.pct)
            result.reason = (f"no probed context stayed fully on the GPU "
                             f"(best {top.pct}% at num_ctx {top.ctx:,})")
        else:
            result.reason = "the model could not be loaded for probing"
    return result


def print_probe_report(result: ProbeResult, *, model: str) -> None:
    table = Table(title=f"context probe · {escape(model)}", box=box.ROUNDED,
                  header_style="bold magenta")
    table.add_column("num_ctx", justify="right")
    table.add_column("size", justify="right")
    table.add_column("vram", justify="right")
    table.add_column("gpu", justify="right")
    table.add_column("result")
    for probe in result.probes:
        if probe.failed:
            table.add_row(f"{probe.ctx:,}", "—", "—", "—",
                          f"[err]load failed[/err] [dim]{escape(probe.failed)}[/dim]")
            continue
        style = "ok" if probe.pct >= 100 else "warn"
        table.add_row(f"{probe.ctx:,}", human_bytes(probe.total), human_bytes(probe.vram),
                      f"[{style}]{probe.pct}%[/{style}]",
                      "fully on GPU" if probe.pct >= 100 else "partial offload")
    console.print(table)
    if result.ctx is None:
        console.print(f"[err]{escape(result.reason or 'no context kept the model fully on the GPU')}[/err]")
        return
    bits = [f"[ok]num_ctx {result.ctx:,}[/ok]", "100% GPU"]
    if result.capped_by_model:
        bits.append("[dim]capped by the model's max context[/dim]")
    console.print(" · ".join(bits))


def probe_max_cpu_ctx(req: FitRequest, fit: CpuFit) -> ProbeResult:
    """Load the model at candidate contexts until its host RAM reaches the target.

    The measured footprint is `/api/ps`'s `size - size_vram` for the loaded
    model: the RAM it holds (weights plus the CPU-side KV), which in steady
    state matches what the machine's MemAvailable gave up when it loaded.
    Candidates are solved from the measured footprint line -- the per-token
    cost is affine in the context -- and only contexts at or below the
    target are eligible, so the chosen context never overshoots the fill.
    """
    result = ProbeResult()
    base_options = dict(req.base_options or {})
    if fit.pinned_layers is not None:
        base_options["num_gpu"] = fit.pinned_layers
    probes_max = max(1, _config_int(req.config, "max_cpu_probes", DEFAULT_CPU_PROBES))
    step = CTX_STEP
    target = max(1, fit.target_bytes)
    slope = float(fit.cpu_kv_per_token or step)
    ctx = max(step, fit.ctx or min(4096, fit.model_max or 4096))
    best: Optional[int] = None
    low_bad: Optional[int] = None
    store = CtxCalibrationStore(Path(req.calibration_path)
                                if req.calibration_path else None)
    history = store.points(req.calibration_key) if req.calibration_key else []
    observations: list[tuple[int, int]] = list(history)
    ceiling = fit.budget_ctx or fit.model_max
    tested = 0
    seen: set[int] = set()

    def measured_line() -> Optional[tuple[float, float]]:
        """(intercept, bytes per token) through the measured points.

        A conservative pairwise slope uses both target-side and over-target
        samples, solving the next candidate rather than stepping toward it.
        """
        points = observations
        if not points:
            return None
        line = _measured_memory_line(points, int(max(1, slope)), 0,
                                     conservative=False)
        return (float(line[1]), float(line[0])) if line else None

    def solved_ctx(k_bytes: float) -> Optional[int]:
        """The context whose predicted footprint hits `k_bytes`, or None."""
        fitted = measured_line()
        if fitted is None or fitted[1] <= 0:
            return None
        return _align_down(max(step, int((k_bytes - fitted[0]) / fitted[1])), step)

    while tested < probes_max:
        # Unload first so the next load starts from a released context: the
        # footprint read back is then that context's own, not the old one's.
        try:
            req.client.unload(req.model)
        except OllamaError:
            pass
        seen.add(ctx)
        console.print(f"[dim]probe {tested + 1}: {escape(req.model)} at "
                      f"num_ctx {ctx:,}…[/dim]")
        try:
            req.client.load(req.model, keepalive=req.keepalive,
                            options={**base_options, "num_ctx": ctx})
        except OllamaError as exc:
            result.probes.append(Probe(ctx=ctx, failed=str(exc)[:200]))
            low_bad = ctx if low_bad is None else min(low_bad, ctx)
            tested += 1
        else:
            entry = _wait_for_ps(req.client, req.model, ctx)
            if entry is None or not int(entry.get("size") or 0):
                result.reason = ("could not read the loaded model back from "
                                 "/api/ps (stale or missing entry)")
                break
            total = int(entry.get("size") or 0)
            vram = int(entry.get("size_vram") or 0)
            host = max(0, total - vram)
            pct = vram * 100 // total if total else 0
            result.probes.append(Probe(ctx=ctx, total=total, vram=vram,
                                       pct=pct, host=host))
            _record_calibration(req, ctx, host)
            tested += 1
            observations.append((ctx, host))
            if host <= target:
                if best is None or ctx > best:
                    best = ctx
                if fit.model_max and ctx >= fit.model_max:
                    result.capped_by_model = True
                    break
            else:
                low_bad = ctx if low_bad is None else min(low_bad, ctx)
        # ── next candidate: solve the measured line for the target ──────────
        if best is not None and low_bad is not None and low_bad - best <= step:
            break
        if best is None:
            nxt = _align_down(int(ctx * 0.5), step)    # over the target: back off
        else:
            nxt = solved_ctx(target)
            if nxt is None or nxt <= best or (low_bad is not None and nxt >= low_bad):
                nxt = None
            if nxt is None:
                fitted = measured_line()
                projected = (fitted[0] + fitted[1] * (best + step)) if fitted else None
                if low_bad is None and projected is not None and projected > target:
                    break                                    # the boundary is at `best`
                near = low_bad - step if low_bad is not None else None
                if near is not None and near > best and near not in seen:
                    nxt = _align_down(near, step)
                elif low_bad is not None:
                    nxt = _align_down((best + low_bad) // 2, step)
                else:
                    nxt = best + step
        if nxt is None:
            break
        if best is not None:
            nxt = max(nxt, best + step)
        if low_bad is not None:
            nxt = min(nxt, low_bad - step)
        if ceiling:
            nxt = min(nxt, ceiling)
        if fit.model_max:
            nxt = min(nxt, fit.model_max)
        if nxt < step:
            nxt = step
        if nxt == ctx or nxt <= 0:
            break
        ctx = nxt
    result.ctx = best
    result.target_bytes = target
    result.budget_bytes = fit.budget_bytes
    if best is not None:
        successes = [probe for probe in result.probes
                     if not probe.failed and probe.host <= target]
        if successes and successes[-1].ctx != best:
            try:  # leave the model warm at the chosen context
                req.client.load(req.model, keepalive=req.keepalive,
                                options={**base_options, "num_ctx": best})
            except OllamaError:
                pass
    elif not result.reason:
        over = [probe for probe in result.probes if not probe.failed]
        if over:
            top = min(over, key=lambda probe: probe.host)
            result.reason = (f"every probed context was already past the target "
                             f"(smallest {human_bytes(top.host)} at num_ctx "
                             f"{top.ctx:,})")
        else:
            result.reason = "the model could not be loaded for probing"
    return result


def print_cpu_probe_report(result: ProbeResult, *, model: str, percent: int) -> None:
    title = f"context probe · {escape(model)}"
    if result.budget_bytes:
        title += (f" · target {percent}% of {human_bytes(result.budget_bytes)}"
                  f" = {human_bytes(result.target_bytes)}")
    table = Table(title=title, box=box.ROUNDED, header_style="bold magenta")
    table.add_column("num_ctx", justify="right")
    table.add_column("size", justify="right")
    table.add_column("vram", justify="right")
    table.add_column("host RAM", justify="right")
    table.add_column("of target", justify="right")
    table.add_column("result")
    for probe in result.probes:
        if probe.failed:
            table.add_row(f"{probe.ctx:,}", "—", "—", "—", "—",
                          f"[err]load failed[/err] [dim]{escape(probe.failed)}[/dim]")
            continue
        share = probe.host * 100 // result.target_bytes if result.target_bytes else 0
        within = probe.host <= result.target_bytes
        style = "ok" if within else "warn"
        table.add_row(f"{probe.ctx:,}", human_bytes(probe.total), human_bytes(probe.vram),
                      human_bytes(probe.host), f"[{style}]{share}%[/{style}]",
                      "within target" if within else "past the target")
    console.print(table)
    if result.ctx is None:
        console.print(f"[err]{escape(result.reason or 'no probed context stayed within the target')}[/err]")
        return
    chosen = max((probe for probe in result.probes
                  if not probe.failed and probe.ctx == result.ctx),
                 key=lambda probe: probe.ctx, default=None)
    host = chosen.host if chosen else 0
    bits = [f"[ok]num_ctx {result.ctx:,}[/ok]", f"host RAM {human_bytes(host)}"]
    if result.budget_bytes:
        bits.append(f"{host * 100 / result.budget_bytes:.1f}% of the budget")
    if result.capped_by_model:
        bits.append("[dim]capped by the model's max context[/dim]")
    elif result.target_bytes and host * 100 < result.target_bytes * 99:
        bits.append(f"[dim]under the {percent}% target[/dim]")
    console.print(" · ".join(bits))


# ── orchestration used by the REPL command and the CLI flag ────────────────

def parse_num_ctx_arg(text: str) -> tuple[str, Optional[int]]:
    """('value', N) | ('max-gpu', None) | ('error', None) for a --num-ctx argument."""
    token = (text or "").strip().lower()
    if token in {"max-gpu", "--max-gpu", "maxgpu"}:
        return "max-gpu", None
    try:
        return "value", int(token)
    except ValueError:
        return "error", None


def max_gpu_context(req: FitRequest, *, probe: bool = False) -> Optional[int]:
    """Estimate (and optionally verify) the largest 100%-GPU context.

    Prints the whole report; returns the chosen num_ctx, or None when it could
    not be determined (the caller then changes nothing).
    """
    if not req.model:
        console.print("[err]no model selected[/err]")
        return None
    try:
        show_data = req.client.show(req.model)
    except OllamaError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return None
    weight_bytes = 0
    model_digest = ""
    model_entry: dict[str, Any] = {}
    try:
        wanted = canonical_model_name(req.model)
        for entry in req.client.tags():
            if canonical_model_name(entry.get("name") or "") == wanted:
                model_entry = entry
                weight_bytes = int(entry.get("size") or 0)
                model_digest = str(entry.get("digest") or "")
                break
    except (OllamaError, TypeError, ValueError):
        weight_bytes = 0
    profile = model_profile(show_data, model_entry)
    if not weight_bytes and not profile.weights_bytes:
        console.print(f"[err]cannot size {escape(req.model)} on this machine "
                      f"(/api/tags has no entry)[/err]")
        return None

    # Models already resident hold VRAM that the free reading excludes but a
    # reload releases: credit it back (always for the model being sized, other
    # residents unless max_gpu_credit_residents is off).
    credited: list[tuple[str, int]] = []
    skipped_bytes = 0
    resident_observation: Optional[tuple[int, int]] = None
    credit_all = bool(req.config.get("max_gpu_credit_residents", True))
    wanted = canonical_model_name(req.model)
    try:
        for entry in req.client.ps():
            name = str(entry.get("name") or entry.get("model") or "")
            vram = int(entry.get("size_vram") or 0)
            if canonical_model_name(name) == wanted:
                ctx = int(entry.get("context_length") or 0)
                size = int(entry.get("size") or 0)
                if ctx > 0 and size > 0 and vram >= size:
                    resident_observation = (ctx, vram)
            if vram <= 0:
                continue
            if credit_all or canonical_model_name(name) == wanted:
                credited.append((name or req.model, vram))
            else:
                skipped_bytes += vram
    except (OllamaError, TypeError, ValueError):
        credited = []
        skipped_bytes = 0

    machine = req.machine
    if machine is None:
        machine = machine_state_for(getattr(req.client, "base", "") or "",
                                    ssh_target=req.ssh_target, ssh_user=req.ssh_user)
        if machine is None:
            console.print(f"[err]cannot read VRAM on {escape(str(getattr(req.client, 'base', '?')))}"
                          "[/err] [dim]— --max-gpu needs nvidia-smi locally or a "
                          "--remote machine[/dim]")
            return None
    if not machine.devices:
        where = req.host_label or machine.source or "this machine"
        console.print(f"[err]cannot read free GPU memory on {escape(str(where))}[/err] "
                      f"[dim]({escape(machine.error or 'no nvidia-smi data')})[/dim]")
        return None

    pinned: Optional[int] = None
    if req.pinned_spec:
        try:
            pinned = resolve_gpu_layers(req.pinned_spec, model_block_count(show_data))
        except (ValueError, RuntimeError) as exc:
            console.print(f"[err]cannot resolve gpu-layers "
                          f"{escape(str(req.pinned_spec))}: {escape(str(exc))}[/err]")
            return None
        if pinned == 0:
            console.print("[err]gpu-layers is set to cpu — --max-gpu needs GPU layers[/err]")
            return None
    elif isinstance(req.num_gpu, int) and req.num_gpu > 0:
        pinned = req.num_gpu

    store = CtxCalibrationStore(Path(req.calibration_path) if req.calibration_path else None)
    key = _calibration_scope(req, mode="gpu", model_digest=model_digest,
                             weight_bytes=weight_bytes, show_data=show_data,
                             machine=machine, pinned_layers=pinned)
    req.calibration_key = key
    req.calibration_path = str(store.path)
    if resident_observation:
        try:
            store.add(key, *resident_observation)
        except OSError:
            pass
    empirical_points = store.points(key)
    fit = estimate_max_gpu_ctx(show_data, weight_bytes or profile.weights_bytes,
                               machine, pinned_layers=pinned, config=req.config,
                               residents=credited, empirical_points=empirical_points,
                               profile=profile)
    if skipped_bytes:
        fit.notes.append(f"other resident models hold {human_bytes(skipped_bytes)} "
                         f"(not credited: max_gpu_credit_residents is off)")
    label = req.host_label
    if label and host_is_self(label):
        label = "this machine"
    print_fit_report(fit, model=req.model, host_label=label, probe=probe)
    if not fit.feasible:
        console.print(f"[err]{escape(fit.error)}[/err]")
        if fit.ctx_no_margin:
            console.print(f"[dim]· without the fit margin this would fit: "
                          f"num_ctx {fit.ctx_no_margin:,} — lower max_gpu_margin_mb "
                          f"(or LLAMA_ARG_FIT_TARGET) or use --probe[/dim]")
        if not probe:
            return None
        console.print("[dim]probing anyway — the estimate's buffers may be "
                      "conservative[/dim]")
        fit.ctx = fit.ctx or fit.ctx_no_margin or min(4096, fit.model_max or 4096)
    if not probe:
        return fit.ctx
    result = probe_max_gpu_ctx(req, fit)
    print_probe_report(result, model=req.model)
    return result.ctx


def parse_fill_percent(text: str) -> Optional[int]:
    """1-100 for a --max-cpu argument ("80" or "80%"), or None."""
    token = (text or "").strip().rstrip("%").strip()
    try:
        value = int(token)
    except ValueError:
        return None
    return value if 1 <= value <= 100 else None


def max_cpu_context(req: FitRequest, *, percent: int, probe: bool = False) -> Optional[int]:
    """Estimate (and optionally verify) the context that fills `percent`% of the
    DRAM available to the model.

    Prints the whole report; returns the chosen num_ctx, or None when it could
    not be determined (the caller then changes nothing).
    """
    if not req.model:
        console.print("[err]no model selected[/err]")
        return None
    checked = parse_fill_percent(str(percent))
    if checked is None:
        console.print("[err]--max-cpu takes a percentage between 1 and 100[/err]")
        return None
    percent = checked
    try:
        show_data = req.client.show(req.model)
    except OllamaError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return None
    weight_bytes = 0
    model_digest = ""
    model_entry: dict[str, Any] = {}
    try:
        wanted = canonical_model_name(req.model)
        for entry in req.client.tags():
            if canonical_model_name(entry.get("name") or "") == wanted:
                model_entry = entry
                weight_bytes = int(entry.get("size") or 0)
                model_digest = str(entry.get("digest") or "")
                break
    except (OllamaError, TypeError, ValueError):
        weight_bytes = 0
    profile = model_profile(show_data, model_entry)
    if not weight_bytes and not profile.weights_bytes:
        console.print(f"[err]cannot size {escape(req.model)} on this machine "
                      f"(/api/tags has no entry)[/err]")
        return None

    # The RAM the model already holds comes back when it reloads at the new
    # context, so it is part of the budget the fill may spend.
    resident_bytes = 0
    resident_observation: Optional[tuple[int, int]] = None
    wanted = canonical_model_name(req.model)
    try:
        for entry in req.client.ps():
            name = str(entry.get("name") or entry.get("model") or "")
            if canonical_model_name(name) != wanted:
                continue
            size = int(entry.get("size") or 0)
            vram = int(entry.get("size_vram") or 0)
            resident_bytes = max(0, size - vram)
            ctx = int(entry.get("context_length") or 0)
            if ctx > 0 and size > vram:
                resident_observation = (ctx, size - vram)
            break
    except (OllamaError, TypeError, ValueError):
        resident_bytes = 0

    ram = req.ram
    if ram is None:
        ram = machine_ram_for(getattr(req.client, "base", "") or "",
                              ssh_target=req.ssh_target, ssh_user=req.ssh_user)
        if ram is None:
            console.print(f"[err]cannot read memory on "
                          f"{escape(str(getattr(req.client, 'base', '?')))}[/err] "
                          f"[dim]— --max-cpu needs /proc/meminfo locally or a "
                          f"--remote machine[/dim]")
            return None
    if not ram.available_bytes:
        where = req.host_label or ram.source or "this machine"
        console.print(f"[err]cannot read free memory on {escape(str(where))}[/err] "
                      f"[dim]({escape(ram.error or 'no /proc/meminfo data')})[/dim]")
        return None

    machine = req.machine
    if machine is None:
        machine = machine_state_for(getattr(req.client, "base", "") or "",
                                    ssh_target=req.ssh_target, ssh_user=req.ssh_user)
        # Unlike --max-gpu this only sharpens the placement: no GPU data at
        # all is fine, the whole model can live on the CPU.
        if machine is None:
            machine = MachineGpu()

    pinned: Optional[int] = None
    if req.pinned_spec:
        try:
            pinned = resolve_gpu_layers(req.pinned_spec, model_block_count(show_data))
        except (ValueError, RuntimeError) as exc:
            console.print(f"[err]cannot resolve gpu-layers "
                          f"{escape(str(req.pinned_spec))}: {escape(str(exc))}[/err]")
            return None
    elif isinstance(req.num_gpu, int) and req.num_gpu >= 0:
        pinned = req.num_gpu

    fit = estimate_max_cpu_ctx(show_data, weight_bytes or profile.weights_bytes,
                               ram, percent=percent,
                               resident_bytes=resident_bytes, machine=machine,
                               pinned_layers=pinned, config=req.config,
                               profile=profile)
    store = CtxCalibrationStore(Path(req.calibration_path) if req.calibration_path else None)
    key = _calibration_scope(req, mode="cpu", model_digest=model_digest,
                             weight_bytes=weight_bytes, show_data=show_data,
                             machine=machine, pinned_layers=pinned,
                             gpu_layers=fit.gpu_layers)
    req.calibration_key = key
    req.calibration_path = str(store.path)
    if resident_observation:
        try:
            store.add(key, *resident_observation)
        except OSError:
            pass
    empirical_points = store.points(key)
    if empirical_points:
        calibrated_fit = estimate_max_cpu_ctx(
            show_data, weight_bytes or profile.weights_bytes, ram, percent=percent,
            resident_bytes=resident_bytes, machine=machine,
            pinned_layers=pinned, config=req.config,
            empirical_points=empirical_points, profile=profile)
        if pinned is None and calibrated_fit.gpu_layers != fit.gpu_layers:
            fit.notes.append("cached memory slope skipped because it predicts a "
                             "different GPU placement")
        else:
            fit = calibrated_fit
    label = req.host_label
    if label and host_is_self(label):
        label = "this machine"
    print_cpu_fit_report(fit, model=req.model, host_label=label, probe=probe)
    if not fit.feasible:
        console.print(f"[err]{escape(fit.error)}[/err]")
        if not probe:
            return None
        console.print("[dim]probing anyway — the estimate's weights may be "
                      "conservative[/dim]")
        fit.ctx = fit.ctx or min(4096, fit.model_max or 4096)
    if not probe:
        return fit.ctx
    result = probe_max_cpu_ctx(req, fit)
    print_cpu_probe_report(result, model=req.model, percent=percent)
    return result.ctx
