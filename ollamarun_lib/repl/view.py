"""The `ollama ps` table shared by `/ps` and the preload glance."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from rich import box
from rich.markup import escape
from rich.table import Table

from ..helpers import human_bytes, short_stamp

if TYPE_CHECKING:
    from .core import Repl


def ps_table(entries: list[dict[str, Any]]) -> Table:
    table = Table(title="ollama ps", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("model")
    table.add_column("size")
    table.add_column("VRAM")
    table.add_column("quant")
    table.add_column("until")
    for entry in entries:
        details = entry.get("details") or {}
        table.add_row(
            escape(str(entry.get("name") or entry.get("model") or "?")),
            human_bytes(entry.get("size")),
            human_bytes(entry.get("size_vram")),
            escape(str(details.get("quantization_level") or "?")),
            short_stamp(entry.get("expires_at")) or "?")
    return table


def _last_assistant(repl: Repl) -> Optional[dict[str, Any]]:
    for message in reversed(repl.session.messages):
        if message.get("role") == "assistant" and message.get("content"):
            return message
    return None


