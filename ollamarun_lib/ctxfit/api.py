"""The orchestration the REPL command and the CLI flag call: fit_context(),
the estimate/probe paths and their structured outcomes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rich.markup import escape

from ..client import OllamaError
from ..constants import console
from ..helpers import canonical_model_name, host_is_self, human_bytes
from ..interactive import model_block_count, resolve_gpu_layers
from .calibration import CtxCalibrationStore, _calibration_scope
from .estimate import estimate_max_cpu_ctx, estimate_max_gpu_ctx
from .machine import MachineGpu, machine_ram_for, machine_state_for
from .models import CpuFit, FitRequest, GpuFit, Probe, ProbeResult
from .probe import (_check_cancel, _progress, probe_max_cpu_ctx,
                    probe_max_gpu_ctx)
from .profile import model_profile
from .report import (print_cpu_fit_report, print_cpu_probe_report,
                     print_fit_report, print_probe_report)

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


@dataclass
class FitOutcome:
    """The structured result of one fit; the compute path never prints.

    `num_ctx` is the answer when the fit succeeded; `error`/`hint` carry the
    CLI's failure text as data when it did not. Cancellation is not an
    outcome -- a stopped fit raises `FitCancelled` at its last checkpoint.
    """

    mode: str = "gpu"                    # "gpu" | "cpu"
    model: str = ""
    probe: bool = False
    percent: Optional[int] = None        # --max-cpu fill percentage
    num_ctx: Optional[int] = None
    source: str = "estimate"             # "estimate" | "probe"
    bound: str = ""                      # "max_gpu" | "max_cpu"
    gpu_layers: Optional[str] = None     # the pinned spec, as requested
    host_label: Optional[str] = None
    fit: Any = None                      # GpuFit | CpuFit (the report source)
    probe_result: Optional[ProbeResult] = None
    calibration_points: int = 0
    error: Optional[str] = None          # plain text; the CLI wraps it in [err]
    hint: Optional[str] = None           # the CLI's dim follow-up line

    @property
    def ok(self) -> bool:
        return self.error is None and self.num_ctx is not None

    def to_dict(self) -> dict[str, Any]:
        """The API shape: numbers and labels only, never paths or markup."""
        payload: dict[str, Any] = {
            "model": self.model, "mode": self.mode, "num_ctx": self.num_ctx,
            "source": self.source, "bound": self.bound,
            "gpu_layers": self.gpu_layers,
        }
        if self.percent is not None:
            payload["percent"] = self.percent
        if self.error is not None:
            payload["error"] = self.error
        if self.fit is not None:
            builder = _cpu_report if self.mode == "cpu" else _gpu_report
            payload["report"] = builder(self.fit, self.probe_result)
        return payload


_GPU_REPORT_FIELDS = (
    "weights_bytes", "weights_total", "weights_source", "kv_per_token",
    "kv_gpu_per_token", "kv_bytes_at_ctx", "kv_bytes_total_at_ctx",
    "recurrent_bytes", "recurrent_gpu_bytes", "free_bytes", "usable_bytes",
    "margin_bytes", "overhead_bytes", "credited_bytes", "pinned_layers",
    "total_layers", "parallel", "element_bytes", "model_max",
    "capped_by_model", "ctx_no_margin", "calibrated", "source",
)
_CPU_REPORT_FIELDS = (
    "percent", "total_bytes", "available_bytes", "resident_bytes",
    "budget_bytes", "target_bytes", "footprint_bytes", "weights_total",
    "cpu_weights", "kv_per_token", "cpu_kv_per_token", "overhead_bytes",
    "gpu_overhead_bytes", "gpu_layers", "total_layers", "pinned_layers",
    "model_max", "budget_ctx", "capped_by_model", "free_bytes",
    "margin_bytes", "usable_bytes", "cpu_kv_bytes", "recurrent_bytes",
    "recurrent_cpu_bytes", "weights_source", "calibrated", "source",
)


def _probe_dicts(probes: list[Probe]) -> list[dict[str, Any]]:
    rows = []
    for probe in probes:
        row: dict[str, Any] = {"ctx": probe.ctx}
        if probe.failed:
            row["failed"] = probe.failed
        else:
            row["total_bytes"] = probe.total
            row["vram_bytes"] = probe.vram
            row["host_bytes"] = probe.host
            row["gpu_pct"] = probe.pct
        rows.append(row)
    return rows


def _gpu_report(fit: GpuFit, probe_result: Optional[ProbeResult]) -> dict[str, Any]:
    report = {name: getattr(fit, name) for name in _GPU_REPORT_FIELDS}
    report["devices"] = [{"total_bytes": total, "free_bytes": free}
                         for total, free in fit.devices]
    report["residents"] = [{"model": name, "size_vram": size}
                           for name, size in fit.residents]
    report["notes"] = list(fit.notes)
    if probe_result is not None:
        report["probes"] = _probe_dicts(probe_result.probes)
        report["probe_reason"] = probe_result.reason or None
    return report


def _cpu_report(fit: CpuFit, probe_result: Optional[ProbeResult]) -> dict[str, Any]:
    report = {name: getattr(fit, name) for name in _CPU_REPORT_FIELDS}
    report["devices"] = [{"total_bytes": total, "free_bytes": free}
                         for total, free in fit.devices]
    report["notes"] = list(fit.notes)
    if probe_result is not None:
        report["probes"] = _probe_dicts(probe_result.probes)
        report["probe_reason"] = probe_result.reason or None
    return report


def fit_context(req: FitRequest, *, mode: str,
                percent: Optional[int] = None,
                probe: bool = False) -> FitOutcome:
    """Compute one context fit; never prints and never raises for failures.

    The CLI's `max_gpu_context()`/`max_cpu_context()` print from the outcome;
    the serve API returns it as data. A cancelled `req.cancel` stops the fit
    at its next checkpoint with `FitCancelled`.
    """
    if mode == "cpu":
        return _fit_cpu(req, percent=percent, probe=probe)
    return _fit_gpu(req, probe=probe)


def _fit_gpu(req: FitRequest, *, probe: bool) -> FitOutcome:
    outcome = FitOutcome(mode="gpu", model=req.model, probe=probe,
                         bound="max_gpu", gpu_layers=req.pinned_spec)
    _check_cancel(req)
    _progress(req, stage="estimate", mode="gpu")
    if not req.model:
        outcome.error = "no model selected"
        return outcome
    try:
        show_data = req.client.show(req.model)
    except OllamaError as exc:
        outcome.error = str(exc)
        return outcome
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
        outcome.error = (f"cannot size {req.model} on this machine "
                         f"(/api/tags has no entry)")
        return outcome

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

    _check_cancel(req)
    _progress(req, stage="machine", mode="gpu")
    machine = req.machine
    if machine is None:
        machine = machine_state_for(getattr(req.client, "base", "") or "",
                                    ssh_target=req.ssh_target, ssh_user=req.ssh_user)
        if machine is None:
            outcome.error = (f"cannot read VRAM on "
                             f"{getattr(req.client, 'base', '?')}")
            outcome.hint = ("--max-gpu needs nvidia-smi locally or a "
                            "--remote machine")
            return outcome
    if not machine.devices:
        where = req.host_label or machine.source or "this machine"
        outcome.error = f"cannot read free GPU memory on {where}"
        outcome.hint = machine.error or "no nvidia-smi data"
        return outcome

    pinned: Optional[int] = None
    if req.pinned_spec:
        try:
            pinned = resolve_gpu_layers(req.pinned_spec, model_block_count(show_data))
        except (ValueError, RuntimeError) as exc:
            outcome.error = (f"cannot resolve gpu-layers "
                             f"{req.pinned_spec}: {exc}")
            return outcome
        if pinned == 0:
            outcome.error = "gpu-layers is set to cpu — --max-gpu needs GPU layers"
            return outcome
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
    outcome.host_label = label
    outcome.fit = fit
    outcome.calibration_points = len(empirical_points)
    if not fit.feasible and not probe:
        return outcome
    if not probe:
        outcome.num_ctx = fit.ctx
        return outcome
    if not fit.feasible:
        fit.ctx = fit.ctx or fit.ctx_no_margin or min(4096, fit.model_max or 4096)
    _check_cancel(req)
    result = probe_max_gpu_ctx(req, fit)
    outcome.probe_result = result
    outcome.num_ctx = result.ctx
    outcome.source = "probe"
    return outcome


def max_gpu_context(req: FitRequest, *, probe: bool = False) -> Optional[int]:
    """Estimate (and optionally verify) the largest 100%-GPU context.

    Prints the whole report; returns the chosen num_ctx, or None when it could
    not be determined (the caller then changes nothing).
    """
    outcome = fit_context(req, mode="gpu", probe=probe)
    if outcome.error is not None:
        console.print(f"[err]{escape(outcome.error)}[/err]")
        if outcome.hint:
            console.print(f"[dim]— {escape(outcome.hint)}[/dim]")
        return None
    fit = outcome.fit
    print_fit_report(fit, model=req.model, host_label=outcome.host_label,
                     probe=probe)
    if not fit.feasible:
        console.print(f"[err]{escape(fit.error or '')}[/err]")
        if fit.ctx_no_margin:
            console.print(f"[dim]· without the fit margin this would fit: "
                          f"num_ctx {fit.ctx_no_margin:,} — lower max_gpu_margin_mb "
                          f"(or LLAMA_ARG_FIT_TARGET) or use --probe[/dim]")
        if not probe:
            return None
        console.print("[dim]probing anyway — the estimate's buffers may be "
                      "conservative[/dim]")
    if outcome.probe_result is not None:
        print_probe_report(outcome.probe_result, model=req.model)
    return outcome.num_ctx


def parse_fill_percent(text: str) -> Optional[int]:
    """1-100 for a --max-cpu argument ("80" or "80%"), or None."""
    token = (text or "").strip().rstrip("%").strip()
    try:
        value = int(token)
    except ValueError:
        return None
    return value if 1 <= value <= 100 else None


def _fit_cpu(req: FitRequest, *, percent: Optional[int],
             probe: bool) -> FitOutcome:
    outcome = FitOutcome(mode="cpu", model=req.model, probe=probe,
                         bound="max_cpu", gpu_layers=req.pinned_spec)
    _check_cancel(req)
    _progress(req, stage="estimate", mode="cpu")
    if not req.model:
        outcome.error = "no model selected"
        return outcome
    checked = parse_fill_percent(str(percent))
    if checked is None:
        outcome.error = "--max-cpu takes a percentage between 1 and 100"
        return outcome
    percent = checked
    outcome.percent = checked
    try:
        show_data = req.client.show(req.model)
    except OllamaError as exc:
        outcome.error = str(exc)
        return outcome
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
        outcome.error = (f"cannot size {req.model} on this machine "
                         f"(/api/tags has no entry)")
        return outcome

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

    _check_cancel(req)
    _progress(req, stage="machine", mode="cpu")
    ram = req.ram
    if ram is None:
        ram = machine_ram_for(getattr(req.client, "base", "") or "",
                              ssh_target=req.ssh_target, ssh_user=req.ssh_user)
        if ram is None:
            outcome.error = (f"cannot read memory on "
                             f"{getattr(req.client, 'base', '?')}")
            outcome.hint = ("--max-cpu needs /proc/meminfo locally or a "
                            "--remote machine")
            return outcome
    if not ram.available_bytes:
        where = req.host_label or ram.source or "this machine"
        outcome.error = f"cannot read free memory on {where}"
        outcome.hint = ram.error or "no /proc/meminfo data"
        return outcome

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
            outcome.error = (f"cannot resolve gpu-layers "
                             f"{req.pinned_spec}: {exc}")
            return outcome
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
    outcome.host_label = label
    outcome.fit = fit
    outcome.calibration_points = len(empirical_points)
    if not fit.feasible and not probe:
        return outcome
    if not probe:
        outcome.num_ctx = fit.ctx
        return outcome
    if not fit.feasible:
        fit.ctx = fit.ctx or min(4096, fit.model_max or 4096)
    _check_cancel(req)
    result = probe_max_cpu_ctx(req, fit)
    outcome.probe_result = result
    outcome.num_ctx = result.ctx
    outcome.source = "probe"
    return outcome


def max_cpu_context(req: FitRequest, *, percent: int, probe: bool = False) -> Optional[int]:
    """Estimate (and optionally verify) the context that fills `percent`% of the
    DRAM available to the model.

    Prints the whole report; returns the chosen num_ctx, or None when it could
    not be determined (the caller then changes nothing).
    """
    outcome = fit_context(req, mode="cpu", percent=percent, probe=probe)
    if outcome.error is not None:
        console.print(f"[err]{escape(outcome.error)}[/err]")
        if outcome.hint:
            console.print(f"[dim]— {escape(outcome.hint)}[/dim]")
        return None
    fit = outcome.fit
    print_cpu_fit_report(fit, model=req.model, host_label=outcome.host_label,
                         probe=probe)
    if not fit.feasible:
        console.print(f"[err]{escape(fit.error or '')}[/err]")
        if not probe:
            return None
        console.print("[dim]probing anyway — the estimate's weights may be "
                      "conservative[/dim]")
    if outcome.probe_result is not None:
        print_cpu_probe_report(outcome.probe_result, model=req.model,
                               percent=outcome.percent or percent)
    return outcome.num_ctx
