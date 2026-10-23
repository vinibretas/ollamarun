"""Context commands: `/context` and `/summarize`."""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.table import Table

from ...constants import console

if TYPE_CHECKING:
    from ..core import Repl


def cmd_context(repl: Repl, argument: str) -> None:
    messages = repl.session.messages
    roles: dict[str, int] = {}
    for message in messages:
        roles[message.get("role", "?")] = roles.get(message.get("role", "?"), 0) + 1
    prompt = repl.session.stats.get("last_prompt_tokens")
    completion = repl.session.stats.get("last_completion_tokens")
    used = repl.context_used()
    total = repl.context_length()
    table = Table(title="context", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("messages", str(len(messages)))
    for role, count in sorted(roles.items()):
        table.add_row(f"  {role}", str(count))
    table.add_row("last prompt tokens", str(prompt if prompt is not None else "?"))
    table.add_row("last completion tokens",
                  str(completion if completion is not None else "?"))
    table.add_row("context used", str(used if used is not None else "?"))
    table.add_row("context length", str(total if total else "unknown"))
    if used and total:
        pct = used * 100 / total
        color = "err" if pct >= 90 else "warn" if pct >= 70 else "ok"
        table.add_row("usage", f"[{color}]{pct:.0f}%[/{color}]")
    console.print(table)


def cmd_summarize(repl: Repl, argument: str) -> None:
    repl.summarize(ask=not argument)


