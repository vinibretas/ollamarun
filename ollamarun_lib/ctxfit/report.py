"""The rich reports behind `/num-ctx --max-gpu` and `--max-cpu`."""
from __future__ import annotations

from typing import Optional

from rich import box
from rich.markup import escape
from rich.table import Table

from ..constants import console
from ..helpers import human_bytes
from .defaults import MIB
from .models import CpuFit, GpuFit, ProbeResult

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


