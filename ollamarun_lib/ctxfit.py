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

The estimate is built from sources that are actually deterministic:

- The KV-cache size per token is exact for standard attention layers, computed
  from `/api/show` GGUF metadata with llama.cpp's own formula
  (`2 * kv_heads * (key_length + value_length) * element_bytes` per layer),
  including per-layer `head_count_kv` arrays from hybrid models that only
  carry a KV cache on some blocks, `key_length`/`value_length` fallbacks to
  `embedding_length / head_count`, `OLLAMA_NUM_PARALLEL` sequence slices and
  `OLLAMA_KV_CACHE_TYPE` element sizes.
- Weights come from `/api/tags` (`size`, the on-disk blob total, so vision
  projectors are counted).
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
found.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from rich import box
from rich.markup import escape
from rich.table import Table

from .client import OllamaError
from .constants import console
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
from .interactive import model_block_count, resolve_gpu_layers

MIB = 1024 * 1024

DEFAULT_MARGIN_MB = 1024      # llama.cpp's --fit-target default, per device
DEFAULT_OVERHEAD_MB = 256     # compute/output/recurrent buffers the estimate can't read
DEFAULT_CPU_OVERHEAD_MB = 256  # CPU-side buffers the --max-cpu estimate can't read
DEFAULT_PROBES = 8
DEFAULT_CPU_PROBES = 8
CTX_STEP = 256                # llama.cpp aligns fit-reduced context to 256 * streams
ESTIMATE_STEP = 1024          # the estimate rounds down to readable numbers

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


NUM_CTX_HELP = """usage: /num-ctx N | --max-gpu [--probe] | --max-cpu N [--probe]
  N          set the context window (capped to the model's max)
  --max-gpu  use the largest context that still runs the model on the GPU:
             the model's exact KV-cache size per token (from /api/show) and
             on-disk size (/api/tags) against the machine's free VRAM (an
             nvidia-smi probe, locally or over ssh for --remote hosts),
             minus llama.cpp's fit margin (LLAMA_ARG_FIT_TARGET, or
             max_gpu_margin_mb) and an overhead allowance
             (max_gpu_overhead_mb). Models already resident are credited
             back -- their VRAM is released when they reload at the new
             context; max_gpu_credit_residents=false limits that credit to
             the model being sized. A pinned /gpu-layers is honoured.
  --max-cpu N  raise the context until the model's host RAM reaches N% (1-100)
             of the DRAM available to it: /proc/meminfo's MemAvailable -- one
             ssh probe for --remote hosts -- plus the RAM the loaded model
             already holds (/api/ps size - size_vram; a reload returns it),
             less an overhead allowance (max_cpu_overhead_mb). Some layers
             stay on the GPU unless /gpu-layers is cpu; a pinned /gpu-layers
             is honoured and only the host side counts toward the target.
  --probe    verify by loading: starting from the estimate, the model is
             warmed at candidate contexts and the real reading is taken back
             from /api/ps -- the GPU split for --max-gpu, the host RAM for
             --max-cpu -- until the boundary is bracketed
             (max_gpu_probes / max_cpu_probes, default 8)."""


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


def kv_bytes_per_token(show_data: dict[str, Any], *, parallel: int = 1,
                       element_bytes: float = 2.0) -> Optional[int]:
    """KV-cache bytes per context token, or None when the metadata is missing.

    Mirrors llama.cpp: `2 (K+V) * kv_heads * (head_k + head_v) * element_size`
    per layer, per token. `head_count_kv` may be a per-block list (hybrid
    models only cache keys/values on some blocks), and `key_length`/
    `value_length` fall back to `embedding_length / head_count`.
    `parallel` is the number of sequences the cache holds.
    """
    info = _model_info(show_data)
    arch = _arch_prefix(info)
    blocks = model_block_count(show_data)
    if not blocks or not arch:
        return None
    heads = _arch_key(info, arch, "attention.head_count_kv")
    if isinstance(heads, list):
        kv_heads = sum(int(value) for value in heads
                       if isinstance(value, (int, float)) and value > 0)
    elif isinstance(heads, (int, float)) and heads > 0:
        kv_heads = int(heads) * blocks
    else:
        return None
    if not kv_heads:
        return None
    key = _arch_key(info, arch, "attention.key_length")
    value = _arch_key(info, arch, "attention.value_length")
    if not (isinstance(key, (int, float)) and key > 0):
        key = None
    if not (isinstance(value, (int, float)) and value > 0):
        value = key
    if not key:
        embedding = _arch_key(info, arch, "embedding_length")
        head_count = _arch_key(info, arch, "attention.head_count")
        if not (isinstance(embedding, (int, float)) and embedding > 0
                and isinstance(head_count, (int, float)) and head_count > 0):
            return None
        key = value = embedding / head_count
    per_token = int(round(kv_heads * (key + value) * element_bytes))
    return max(1, per_token) * max(1, int(parallel))


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
                         residents: Optional[list[tuple[str, int]]] = None) -> GpuFit:
    """Largest context that keeps the placement on the GPU (no loads, pure math).

    `residents` are the models already holding VRAM as (name, size_vram): the
    free-VRAM reading then excludes what their own instances hold, but that
    memory is released when they reload at the new context, so it is credited
    back into the pool instead of being counted against the placement.
    """
    config = config or {}
    fit = GpuFit(weights_total=max(0, int(weight_bytes or 0)),
                 devices=list(machine.devices), source=machine.source,
                 pinned_layers=pinned_layers,
                 total_layers=model_total_layers(show_data),
                 model_max=trained_context_length(show_data))
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

    fit.kv_per_token = kv_bytes_per_token(show_data, parallel=fit.parallel,
                                          element_bytes=fit.element_bytes) or 0
    if not fit.kv_per_token:
        fit.error = ("cannot size the KV cache from /api/show "
                     "(attention metadata missing for this architecture)")
        return fit

    if pinned_layers is None:
        fit.weights_bytes = fit.weights_total
        fit.kv_gpu_per_token = fit.kv_per_token
        fit.notes.append("all layers are placed on the GPU")
    else:
        total = fit.total_layers or 0
        weight_fraction = min(1.0, max(0.0, pinned_layers / total)) if total else 1.0
        fit.weights_bytes = int(fit.weights_total * weight_fraction)
        kv_fraction = _kv_fraction_on_gpu(show_data, pinned_layers)
        fit.kv_gpu_per_token = max(1, int(fit.kv_per_token * kv_fraction))
        where = f"gpu-layers {pinned_layers}" + (f" of {total}" if total else "")
        kv_note = ("the whole KV cache stays on the GPU" if kv_fraction >= 1
                   else f"KV for {kv_fraction * 100:.0f}% of the cache stays on the GPU")
        fit.notes.append(f"{where} pinned: {kv_note}")

    budget = fit.usable_bytes - fit.weights_bytes - fit.overhead_bytes
    ctx = budget // fit.kv_gpu_per_token if budget > 0 else 0
    if ctx and fit.model_max and ctx > fit.model_max:
        ctx = fit.model_max
        fit.capped_by_model = True
    if ctx >= ESTIMATE_STEP:
        ctx = (ctx // ESTIMATE_STEP) * ESTIMATE_STEP
    fit.ctx = max(0, int(ctx))
    fit.feasible = fit.ctx > 0
    # What the same placement would fit with no margin at all: printed as a
    # hint when the per-device reserve is the only thing making it infeasible.
    usable_unc = (sum(max(0, free * MIB) for _, free in machine.devices)
                  + fit.credited_bytes)
    budget_unc = usable_unc - fit.weights_bytes - fit.overhead_bytes
    unc = budget_unc // fit.kv_gpu_per_token if budget_unc > 0 else 0
    if unc and fit.model_max and unc > fit.model_max:
        unc = fit.model_max
    if unc >= ESTIMATE_STEP:
        unc = (unc // ESTIMATE_STEP) * ESTIMATE_STEP
    fit.ctx_no_margin = int(unc) if unc else None
    if not fit.feasible:
        usable = human_bytes(fit.usable_bytes) if fit.usable_bytes else "0 B"
        if fit.credited_bytes:
            usable += (f" ({human_bytes(fit.free_bytes)} free + "
                       f"{human_bytes(fit.credited_bytes)} resident)")
        fit.error = (f"the GPU placement does not fit: {human_bytes(fit.weights_bytes)} "
                     f"of weights plus {human_bytes(fit.overhead_bytes)} of buffers vs "
                     f"{usable} usable after the "
                     f"{human_bytes(fit.margin_bytes)} margin")
    return fit


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
    notes: list[str] = field(default_factory=list)
    error: str = ""


def estimate_max_cpu_ctx(show_data: dict[str, Any], weight_bytes: int, ram: MachineRam, *,
                         percent: int, resident_bytes: int = 0,
                         machine: Optional[MachineGpu] = None,
                         pinned_layers: Optional[int] = None,
                         config: Optional[dict[str, Any]] = None) -> CpuFit:
    """The context whose host RAM reaches `percent`% of the DRAM available to it.

    The budget is MemAvailable plus the RAM the loaded model already holds
    (it comes back when the model reloads at the new context); other
    programs and models keep theirs. The weights and the per-token KV of the
    CPU-side layers are the deterministic inputs -- what llama.cpp's fit
    keeps on the GPU is simulated from free VRAM unless `/gpu-layers` is
    pinned.
    """
    config = config or {}
    fit = CpuFit(
        percent=max(1, min(100, int(percent))),
        total_bytes=max(0, int(ram.total_bytes or 0)),
        available_bytes=max(0, int(ram.available_bytes or 0)),
        resident_bytes=max(0, int(resident_bytes or 0)),
        weights_total=max(0, int(weight_bytes or 0)),
        pinned_layers=pinned_layers,
        total_layers=model_total_layers(show_data),
        model_max=trained_context_length(show_data),
        source=ram.source,
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
    fit.kv_per_token = kv_bytes_per_token(show_data, parallel=sequences,
                                          element_bytes=element) or 0
    if not fit.kv_per_token:
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

    def placement(at_ctx: int) -> tuple[Optional[int], int, int]:
        """(gpu layers, CPU weights, CPU KV per token) at `at_ctx`.

        llama.cpp fills GPU layers from the top (the output layer first,
        then the last blocks) while weights and the tail's per-layer KV fit
        the usable VRAM; the rest of the model stays on the CPU. With no GPU
        data at all everything is assumed to stay on the CPU.
        """
        total = fit.total_layers or 0
        if fit.pinned_layers is not None:
            layers = max(0, min(total, fit.pinned_layers))
        elif fit.usable_bytes:
            layers = total
            while layers > 0:
                weights = fit.weights_total * layers // total
                gpu_kv = int(fit.kv_per_token * _kv_fraction_on_gpu(show_data, layers))
                if weights + gpu_kv * at_ctx + fit.gpu_overhead_bytes <= fit.usable_bytes:
                    break
                layers -= 1
        else:
            layers = 0
        weights = fit.weights_total * (total - layers) // total
        gpu_kv = int(fit.kv_per_token * _kv_fraction_on_gpu(show_data, layers))
        return layers, weights, max(0, fit.kv_per_token - gpu_kv)

    def footprint(at_ctx: int) -> int:
        _, weights, kv = placement(at_ctx)
        return weights + kv * at_ctx + fit.overhead_bytes

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
        fit.gpu_layers, fit.cpu_weights, fit.cpu_kv_per_token = placement(CTX_STEP)
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
    fit.gpu_layers, fit.cpu_weights, fit.cpu_kv_per_token = placement(fit.ctx)
    fit.footprint_bytes = footprint(fit.ctx)
    fit.budget_ctx = solve(ceiling, fit.budget_bytes) if fit.budget_bytes else None
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
    table.add_row("gpu memory", memory)
    table.add_row("margin", f"{human_bytes(fit.margin_bytes)} per device → "
                            f"{human_bytes(fit.usable_bytes)} usable")
    if fit.pinned_layers is None:
        table.add_row("weights", human_bytes(fit.weights_bytes))
    else:
        table.add_row("weights", f"{human_bytes(fit.weights_bytes)} of "
                                 f"{human_bytes(fit.weights_total)} (pinned layers)")
    kv_line = f"{human_bytes(fit.kv_gpu_per_token)} per token"
    if fit.ctx:
        kv_line += f" · {human_bytes(fit.kv_gpu_per_token * fit.ctx)} at {fit.ctx:,}"
    if fit.kv_gpu_per_token != fit.kv_per_token:
        kv_line += f" (of {human_bytes(fit.kv_per_token)} total)"
    table.add_row("kv cache", kv_line)
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
    table.add_row("cpu weights", f"{human_bytes(fit.cpu_weights)} of "
                                 f"{human_bytes(fit.weights_total)}")
    kv_line = f"{human_bytes(fit.cpu_kv_per_token)} per token"
    if fit.ctx:
        kv_line += f" · {human_bytes(fit.cpu_kv_per_token * fit.ctx)} at {fit.ctx:,}"
    if fit.cpu_kv_per_token != fit.kv_per_token:
        kv_line += f" (of {human_bytes(fit.kv_per_token)} total)"
    table.add_row("kv cache", kv_line)
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
    good: list[tuple[int, int]] = []        # (ctx, measured total) of full-offload probes
    tested = 0

    def measured_line() -> Optional[tuple[float, float]]:
        """(intercept, bytes per token) through the measured full-offload points.

        llama.cpp's footprint is affine in num_ctx: one measured point pins the
        line to the current slope, two or more are fitted exactly, and the next
        candidate is then *solved* for the capacity rather than bisected toward.
        """
        if not good:
            return None
        if len(good) >= 2:
            mean_x = sum(at for at, _ in good) / len(good)
            mean_y = sum(size for _, size in good) / len(good)
            denom = sum((at - mean_x) ** 2 for at, _ in good)
            if denom > 0:
                factor = sum((at - mean_x) * (size - mean_y)
                             for at, size in good) / denom
                if factor > 0:
                    return (mean_y - factor * mean_x, factor)
        at, size = good[0]
        return (size - slope * at, slope)

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
        console.print(f"[dim]probe {tested + 1}: {escape(req.model)} at "
                      f"num_ctx {ctx:,}…[/dim]")
        try:
            req.client.load(req.model, keepalive=req.keepalive,
                            options={**base_options, "num_ctx": ctx})
        except OllamaError as exc:
            result.probes.append(Probe(ctx=ctx, failed=str(exc)[:200]))
            low_bad = ctx if low_bad is None else min(low_bad, ctx)
            tighten_ceiling(footprint(ctx))
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
            tested += 1
            if pct >= 100:
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
            nxt = _align_down(int(ctx * 0.7), step)       # nothing fit yet
        elif k_hi is None:
            nxt = ctx + step                              # no ceiling known
        else:
            if k_hi - k_lo <= slope * step:
                break                                     # converged in bytes
            nxt = solved_ctx(k_hi)
            if nxt is None or nxt == ctx \
                    or (low_bad is not None and nxt >= low_bad - step):
                # The ceiling is extrapolated past the known failure (or the
                # solve stalled): halve the byte interval instead, which never
                # fails to shrink the bracket.
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
    good: list[tuple[int, int]] = []    # (ctx, host bytes) at or below the target
    ceiling = fit.budget_ctx or fit.model_max
    tested = 0

    def measured_line() -> Optional[tuple[float, float]]:
        """(intercept, bytes per token) through the measured points.

        One point pins the line to the current slope, two or more are fitted
        exactly, and the next candidate is solved for the target rather than
        stepped toward it.
        """
        if not good:
            return None
        if len(good) >= 2:
            mean_x = sum(at for at, _ in good) / len(good)
            mean_y = sum(host for _, host in good) / len(good)
            denom = sum((at - mean_x) ** 2 for at, _ in good)
            if denom > 0:
                factor = sum((at - mean_x) * (host - mean_y)
                             for at, host in good) / denom
                if factor > 0:
                    return (mean_y - factor * mean_x, factor)
        at, host = good[0]
        return (host - slope * at, slope)

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
            tested += 1
            if host <= target:
                if best is None or ctx > best:
                    best = ctx
                good.append((ctx, host))
                if fit.model_max and ctx >= fit.model_max:
                    result.capped_by_model = True
                    break
            else:
                low_bad = ctx if low_bad is None else min(low_bad, ctx)
        # ── next candidate: solve the measured line for the target ──────────
        if best is not None and low_bad is not None and low_bad - best <= step:
            break
        if best is None:
            nxt = _align_down(int(ctx * 0.7), step)     # over the target: back off
        else:
            nxt = solved_ctx(target)
            if nxt is None or nxt <= best or (low_bad is not None and nxt >= low_bad):
                nxt = None
            if nxt is None:
                fitted = measured_line()
                projected = (fitted[0] + fitted[1] * (best + step)) if fitted else None
                if low_bad is None and projected is not None and projected > target:
                    break                                    # the boundary is at `best`
                if low_bad is not None:
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
    try:
        wanted = canonical_model_name(req.model)
        for entry in req.client.tags():
            if canonical_model_name(entry.get("name") or "") == wanted:
                weight_bytes = int(entry.get("size") or 0)
                break
    except (OllamaError, TypeError, ValueError):
        weight_bytes = 0
    if not weight_bytes:
        console.print(f"[err]cannot size {escape(req.model)} on this machine "
                      f"(/api/tags has no entry)[/err]")
        return None

    # Models already resident hold VRAM that the free reading excludes but a
    # reload releases: credit it back (always for the model being sized, other
    # residents unless max_gpu_credit_residents is off).
    credited: list[tuple[str, int]] = []
    skipped_bytes = 0
    credit_all = bool(req.config.get("max_gpu_credit_residents", True))
    wanted = canonical_model_name(req.model)
    try:
        for entry in req.client.ps():
            name = str(entry.get("name") or entry.get("model") or "")
            vram = int(entry.get("size_vram") or 0)
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

    fit = estimate_max_gpu_ctx(show_data, weight_bytes, machine,
                               pinned_layers=pinned, config=req.config,
                               residents=credited)
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
    try:
        wanted = canonical_model_name(req.model)
        for entry in req.client.tags():
            if canonical_model_name(entry.get("name") or "") == wanted:
                weight_bytes = int(entry.get("size") or 0)
                break
    except (OllamaError, TypeError, ValueError):
        weight_bytes = 0
    if not weight_bytes:
        console.print(f"[err]cannot size {escape(req.model)} on this machine "
                      f"(/api/tags has no entry)[/err]")
        return None

    # The RAM the model already holds comes back when it reloads at the new
    # context, so it is part of the budget the fill may spend.
    resident_bytes = 0
    wanted = canonical_model_name(req.model)
    try:
        for entry in req.client.ps():
            name = str(entry.get("name") or entry.get("model") or "")
            if canonical_model_name(name) != wanted:
                continue
            size = int(entry.get("size") or 0)
            vram = int(entry.get("size_vram") or 0)
            resident_bytes = max(0, size - vram)
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

    fit = estimate_max_cpu_ctx(show_data, weight_bytes, ram, percent=percent,
                               resident_bytes=resident_bytes, machine=machine,
                               pinned_layers=pinned, config=req.config)
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
