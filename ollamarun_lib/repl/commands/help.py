"""The `/help` command: the command table and its topics."""
from __future__ import annotations

from typing import TYPE_CHECKING
from rich import box
from rich.markup import escape
from rich.table import Table
from ...constants import console
from .tables import COMMAND_INFO

if TYPE_CHECKING:
    from ..core import Repl
from .tools import _cmd_tools_help_table


def cmd_help(repl: Repl, argument: str) -> None:
    if argument == "tools":
        _cmd_tools_help_table()
        return
    if argument == "shortcuts":
        table = Table(title="shortcuts", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("key")
        table.add_column("action")
        entries = [
            ("Ctrl+D", "close the session (and its background host)"),
            ("Ctrl+C", ("cancel input · twice within 10s while busy detaches "
                        "and leaves the session running")
             if repl.host_mode else "cancel input / generation"),
            ("Ctrl+T", "expand or collapse the thinking panel; sticks for "
                       "the session"),
            ("Ctrl+Q", "stop the current generation (inline)"),
            ("Ctrl+G", "$OLLAMA_EDITOR for the current prompt"),
            ("Ctrl+J", "insert newline"),
            ("Ctrl+R", "search prompt history"),
            ("Up/Down", "history navigation"),
            ("Tab", "complete commands, ids, models, @files"),
            ("@path", "inline a file into the prompt"),
        ]
        if repl.tui_mode:
            entries = [entry for entry in entries
                       if entry[0] != "Ctrl+Q"] + [
                ("Enter", "send — queues the message while a turn is running"),
                ("Ctrl+B", "send what is typed right now (a running tool finishes "
                           "first)"),
                ("Esc Esc", "stop the turn (first Esc arms it)"),
                ("Ctrl+Q", "quit (twice while busy or with a queued message)"),
                ("Ctrl+L", "clear the transcript pane"),
                ("PgUp/PgDn", "scroll the transcript"),
                ("/queue", "show, remove or drop queued messages and commands"),
            ]
        for key, action in entries:
            table.add_row(key, action)
        console.print(table)
        return
    table = Table(title="commands", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("command")
    table.add_column("usage", style="dim")
    table.add_column("description")
    for command, (usage, description) in COMMAND_INFO.items():
        table.add_row(escape(command), escape(usage), description)
    console.print(table)


