"""Interactive helpers: $EDITOR handoff, fuzzy model/session pickers, and
reading a model's context window from `/api/show`.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from rich import box
from rich.markup import escape
from rich.prompt import Prompt
from rich.table import Table

from .client import OllamaClient, OllamaError
from .constants import console
from .helpers import human_bytes, shortcut, short_stamp
from .session import Session, SessionStore


def open_in_editor(initial: str = "") -> Optional[str]:
    """Edit text in $OLLAMA_EDITOR / $EDITOR; None when the editor fails."""
    editor = os.environ.get("OLLAMA_EDITOR") or os.environ.get("EDITOR") or "vi"
    handle = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8")
    path = Path(handle.name)
    try:
        handle.write(initial)
        handle.close()
        subprocess.call(shlex.split(editor) + [str(path)])
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        console.print(f"[err]editor failed: {exc}[/err]")
        return None
    finally:
        path.unlink(missing_ok=True)


def pick_model(client: OllamaClient, current: Optional[str] = None) -> Optional[str]:
    try:
        models = client.tags()
    except OllamaError as exc:
        console.print(f"[err]{exc}[/err]")
        return None
    if not models:
        console.print("[warn]no models installed — try `ollama pull llama3.2`[/warn]")
        return None
    models.sort(key=lambda item: item.get("modified_at") or "", reverse=True)
    table = Table(title="models", box=box.ROUNDED, header_style="bold magenta",
                  row_styles=["", "dim"])
    table.add_column("#", justify="right")
    table.add_column("name")
    table.add_column("size", justify="right")
    table.add_column("modified")
    table.add_column("family", style="dim")
    for index, model in enumerate(models, 1):
        details = model.get("details") or {}
        modified = short_stamp(model.get("modified_at"))
        marker = " [ok]←[/ok]" if model.get("name") == current else ""
        table.add_row(str(index), escape(model.get("name", "?")) + marker,
                      human_bytes(model.get("size")), modified,
                      escape(details.get("family", "")))
    console.print(table)
    while True:
        answer = Prompt.ask("model number, name or 'q' to quit", default="q").strip()
        if answer in {"q", "", "quit"}:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(models):
            return models[int(answer) - 1]["name"]
        for model in models:
            if model.get("name") == answer:
                return answer
        matches = [model["name"] for model in models if answer.lower() in model["name"].lower()]
        if len(matches) == 1:
            return matches[0]
        if matches:
            console.print("[dim]matches: " + ", ".join(matches) + "[/dim]")
        else:
            console.print("[warn]no match[/warn]")


def pick_session(store: SessionStore, filter_text: str = "") -> Optional[Session]:
    entries = store.entries()
    if filter_text:
        needle = filter_text.lower()
        entries = [entry for entry in entries
                   if needle in json.dumps(entry, default=str).lower()]
    if not entries:
        console.print("[warn]no saved sessions[/warn]")
        return None
    shown = entries[:25]
    table = Table(title=f"sessions ({len(entries)})", box=box.ROUNDED,
                  header_style="bold magenta", row_styles=["", "dim"])
    for column in ("#", "id", "title", "model", "updated", "msgs"):
        table.add_column(column, justify="right" if column in {"#", "msgs"} else "left")
    for index, entry in enumerate(shown, 1):
        table.add_row(str(index), entry["id"],
                      escape(shortcut(entry.get("title", ""), 40)),
                      escape(entry.get("model", "")),
                      short_stamp(entry.get("updated_at"))[:16],
                      str(entry.get("messages", 0)))
    console.print(table)
    while True:
        answer = Prompt.ask("session number, ID, filter or 'q' to quit", default="q").strip()
        if answer in {"q", "", "quit"}:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(shown):
            try:
                return store.load(shown[int(answer) - 1]["id"])
            except (FileNotFoundError, ValueError) as exc:
                console.print(f"[err]{exc}[/err]")
                return None
        session = store.resolve(answer)
        if session is not None:
            return session
        console.print("[dim]filtering…[/dim]")
        return pick_session(store, filter_text=answer)


_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)


def model_context_length(show_data: dict[str, Any],
                         parameters: Optional[dict[str, Any]] = None) -> Optional[int]:
    parameters = parameters or {}
    if isinstance(parameters.get("num_ctx"), int):
        return parameters["num_ctx"]
    for key, value in (show_data.get("model_info") or {}).items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    raw = show_data.get("parameters")
    if isinstance(raw, str):
        match = re.search(r"num_ctx\s+(\d+)", raw)
        if match:
            return int(match.group(1))
    return None
