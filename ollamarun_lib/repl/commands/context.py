"""Context commands: `/context` and `/summarize`."""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.markup import escape
from rich.table import Table

from ...constants import console
from ...helpers import source_color, usage_color

if TYPE_CHECKING:
    from ..core import Repl


#: One colour per message role, matching the transcript, so the counts read
#: at a glance instead of as a wall of dim numbers.
_ROLE_STYLES = {"user": "you", "assistant": "ai", "system": "info", "tool": "dim"}


def cmd_context(repl: Repl, argument: str) -> None:
    report = repl.context_report()
    model = str(report.get("model") or "?")
    window = report.get("window")
    source = str(report.get("source") or "unknown")
    model_max = report.get("model_max")
    num_ctx = report.get("num_ctx")
    prompt = report.get("prompt")
    completion = report.get("completion")
    used = report.get("used")
    counts = report.get("messages") or {}
    total = int(counts.get("total") or 0)

    table = Table(title="context", box=box.ROUNDED, header_style="bold magenta",
                  title_justify="left")
    table.add_column("field", style="dim")
    table.add_column("value", justify="right")
    table.add_row("model", f"[info]{escape(model)}[/info]")
    if window:
        style = source_color(source)
        table.add_row("context length",
                      f"[bold]{window}[/bold] tok "
                      f"[{style}]({escape(source)})[/{style}]")
    else:
        table.add_row("context length", f"[warn]{escape(source)}[/warn]")
    if model_max and (window is None or model_max != window):
        table.add_row("model max", f"[dim]{model_max} tok[/dim]")
    if num_ctx is not None:
        table.add_row("num_ctx", f"[info]{num_ctx}[/info]")
    table.add_row("messages", f"[bold]{total}[/bold]")
    for role in sorted(counts):
        if role == "total" or not counts.get(role):
            continue
        style = _ROLE_STYLES.get(role)
        value = str(counts[role])
        if style:
            value = f"[{style}]{value}[/{style}]"
        table.add_row(f"  {role}", value)
    table.add_row("last prompt tokens",
                  str(prompt if prompt is not None else "?"))
    table.add_row("last completion tokens",
                  str(completion if completion is not None else "?"))
    if used:
        pct = used * 100 / window if window else None
        color = usage_color(pct)
        table.add_row("context used", f"[{color}]{used}[/{color}]")
        if pct is not None:
            table.add_row("usage", f"[{color}]{pct:.1f}%[/{color}]")
            table.add_row("free", f"{max(0, window - used)} tok")
    else:
        table.add_row("context used", "[dim]?[/dim]")
    console.print(table)


def cmd_summarize(repl: Repl, argument: str) -> None:
    repl.summarize(ask=not argument)
