"""Empirical verification: load a candidate context and read /api/ps back."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from rich.markup import escape

from ..client import OllamaError
from ..constants import console
from ..helpers import canonical_model_name, human_bytes
from .calibration import (CtxCalibrationStore, _measured_memory_line,
                          _record_calibration)
from .defaults import (CTX_STEP, DEFAULT_CPU_PROBES, DEFAULT_PROBES, MIB,
                       _align_down, _config_int)
from .machine import MachineGpu
from .models import (CpuFit, FitCancelled, FitRequest, GpuFit, Probe,
                     ProbeResult)


def _check_cancel(req: FitRequest) -> None:
    token = getattr(req, "cancel", None)
    if token is not None and getattr(token, "cancelled", False):
        raise FitCancelled("fit cancelled")


def _progress(req: FitRequest, **data: Any) -> None:
    if req.on_progress is None:
        return
    try:
        req.on_progress(dict(data))
    except Exception:  # a progress sink must never break the fit
        pass




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


def wait_for_ps(client: Any, model: str, ctx: Optional[int] = None,
                *, timeout: float = 8.0) -> Optional[dict[str, Any]]:
    """The /api/ps entry for `model` once it reports the requested context.

    With `ctx` None the first entry for the model answers (used when a load
    carried no explicit `num_ctx` and any window is acceptable).
    """
    wanted = canonical_model_name(model)
    deadline = time.monotonic() + timeout
    while True:
        try:
            for entry in client.ps():
                name = entry.get("name") or entry.get("model") or ""
                if canonical_model_name(name) != wanted:
                    continue
                seen = entry.get("context_length")
                if ctx is None or seen in (None, ctx):
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
        _check_cancel(req)
        _progress(req, stage="probe", mode="gpu", ctx=ctx,
                  attempt=tested + 1, max=probes_max)
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
            entry = wait_for_ps(req.client, req.model, ctx)
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
        _check_cancel(req)
        _progress(req, stage="probe", mode="cpu", ctx=ctx,
                  attempt=tested + 1, max=probes_max)
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
            entry = wait_for_ps(req.client, req.model, ctx)
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


