"""The analytic estimates: the largest fully-GPU context, and the DRAM fill."""
from __future__ import annotations

from typing import Any, Optional

from ..helpers import human_bytes
from .calibration import (_apply_empirical_gpu, _kv_fraction_on_gpu,
                          _measured_memory_line)
from .defaults import (CTX_STEP, DEFAULT_CPU_OVERHEAD_MB, DEFAULT_MARGIN_MB,
                       DEFAULT_OVERHEAD_MB, ESTIMATE_STEP, MIB, _align_down,
                       _config_int)
from .machine import MachineGpu, MachineRam
from .models import CpuFit, GpuFit
from .profile import (ModelProfile, kv_bytes_per_token, kv_element_bytes,
                      model_profile, model_total_layers,
                      trained_context_length)


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


