"""`/stats`, `/queue` and `/bye`."""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.markup import escape
from rich.table import Table

from ...constants import console
from ...helpers import human_duration, shortcut

if TYPE_CHECKING:
    from ..core import Repl


def cmd_stats(repl: Repl, argument: str) -> None:
    stats = repl.session.stats
    table = Table(title="stats", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("session", repl.session.id)
    table.add_row("model", repl.session.model)
    table.add_row("turns", str(stats.get("turns", 0)))
    table.add_row("messages", str(len(repl.session.messages)))
    table.add_row("prompt tokens", str(stats.get("prompt_tokens", 0)))
    table.add_row("completion tokens", str(stats.get("completion_tokens", 0)))
    table.add_row("generation time", human_duration(stats.get("duration_s")))
    if stats.get("last_eval_tps"):
        table.add_row("last speed", f"{stats['last_eval_tps']:.1f} tok/s")
    if stats.get("last_prompt_tokens"):
        table.add_row("last prompt", f"{stats['last_prompt_tokens']} tok")
    if stats.get("last_completion_tokens"):
        table.add_row("last completion", f"{stats['last_completion_tokens']} tok")
    used = repl.context_used()
    if used:
        table.add_row("last context", f"{used} tok")
    console.print(table)


def cmd_queue(repl: Repl, argument: str) -> None:
    """Show or edit what was typed while a turn was running."""
    parts = argument.split()
    topic = parts[0].lower() if parts else "list"
    if topic in {"list", ""}:
        if not repl.queue:
            console.print("[dim]queue is empty[/dim]")
            return
        table = Table(title="queued", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("#", justify="right")
        table.add_column("kind")
        table.add_column("text")
        for index, item in enumerate(repl.queue.items, 1):
            table.add_row(str(index),
                          "you" if item.kind == "message" else "cmd",
                          escape(shortcut(item.text, 72)))
        console.print(table)
        console.print("[dim]messages are delivered at the next sub-task boundary · "
                      "commands run when the turn ends[/dim]")
        return
    if topic in {"clear", "flush", "drop"}:
        count = repl.queue.clear()
        console.print(f"[ok]dropped {count} queued item(s)[/ok]")
        repl.ui.notify("queue_changed")
        return
    if topic in {"rm", "remove", "del"}:
        if len(parts) < 2 or not parts[1].isdigit():
            console.print("[err]usage: /queue rm N[/err]")
            return
        item = repl.queue.remove(int(parts[1]))
        if item is None:
            console.print("[err]no such queued item[/err]")
            return
        console.print(f"[ok]dropped[/ok] {escape(shortcut(item.text, 72))}")
        repl.ui.notify("queue_changed")
        return
    console.print("[err]usage: /queue [list|clear|rm N][/err]")


def cmd_bye(repl: Repl, argument: str) -> bool:
    return False


