"""Interactive helpers: $EDITOR handoff, fuzzy model/session pickers, and
reading a model's context window from `/api/show`.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from rich import box
from rich.markup import escape
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .client import OllamaClient, OllamaError
from .constants import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
    console,
)
from .helpers import (
    human_bytes,
    is_local_host,
    local_free_disk,
    shortcut,
    short_stamp,
    ssh_free_disk,
)
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


def disk_free_line(client: OllamaClient, config: dict[str, Any], *,
                   ssh_target: Optional[str] = None,
                   host_label: Optional[str] = None) -> str:
    """One-line free-space summary for the machine behind the client.

    SSH only when the caller knows a target for it (the `--remote` machines);
    local hosts use `shutil.disk_usage`, anything else degrades to "unknown".
    """
    if ssh_target:
        usage = ssh_free_disk(ssh_target, user=config.get("remote_ssh_user"))
        where = host_label or ssh_target
    elif is_local_host(client.base):
        usage, where = local_free_disk(), "this machine"
    else:
        usage, where = None, host_label or client.base
    if not usage:
        return f"free disk space on {where}: unknown"
    free, total, mount = usage
    return (f"free disk space on {where}: {human_bytes(free)} of "
            f"{human_bytes(total)} ({mount})")


def pull_with_progress(client: OllamaClient, model: str, *, quiet: bool = False) -> None:
    """Pull a model, streaming layer progress through a rich progress view."""
    if quiet:
        client.pull_model(model, None)
        return
    columns = [TextColumn("[progress.description]{task.description}"),
               BarColumn(), DownloadColumn(), TransferSpeedColumn(),
               TimeRemainingColumn()]
    tasks: dict[str, Any] = {}
    with Progress(*columns, console=console, transient=True) as progress:
        headline = progress.add_task(f"pulling {model}", total=None)

        def handle(event: dict[str, Any]) -> None:
            status = str(event.get("status") or "")
            digest = str(event.get("digest") or "")
            total = event.get("total")
            if digest and total:
                task_id = tasks.get(digest)
                if task_id is None:
                    task_id = progress.add_task(
                        f"  {digest.split(':')[-1][:12]}", total=total)
                    tasks[digest] = task_id
                progress.update(task_id, completed=event.get("completed") or 0)
            elif status:
                progress.update(headline, description=f"pulling {model} · {status}")

        client.pull_model(model, handle)
        progress.update(headline, description=f"pulled {model}")


def ensure_model(client: OllamaClient, model: str, config: dict[str, Any], *,
                 ssh_target: Optional[str] = None, host_label: Optional[str] = None,
                 assume_yes: bool = False, pick_on_decline: bool = False,
                 quiet: bool = False, interactive: Optional[bool] = None) -> Optional[str]:
    """Return the model to run, pulling it first when it is not installed.

    `None` means the model is unavailable and the caller should stop. With
    `assume_yes` (`--pull`) a missing model is pulled without asking; on a
    terminal it is otherwise offered with a y/N confirmation after printing
    the target machine's free disk space. `pick_on_decline` opens the model
    picker for that machine when the user declines; `quiet` suppresses the
    informational lines (`--raw`/`--json`/non-tty). `interactive` is only
    overridden by tests.
    """
    if not model:
        return None
    if client.model_exists(model):
        return model
    if host_label:
        where = host_label
    elif is_local_host(client.base):
        where = "the local server"
    else:
        where = client.base
    console.print(f"[warn]model {escape(model)} is not installed on "
                  f"{escape(str(where))}[/warn]")
    if interactive is None:
        interactive = console.is_terminal and sys.stdin.isatty()
    if not quiet or interactive:
        summary = disk_free_line(client, config, ssh_target=ssh_target,
                                 host_label=host_label)
        console.print(f"[dim]{escape(summary)}[/dim]")
    pull = assume_yes
    if not pull and interactive:
        pull = Confirm.ask(f"pull [key]{escape(model)}[/key]?", default=False)
    if not pull:
        if interactive and pick_on_decline:
            chosen = pick_model(client, model)
            if chosen:
                return chosen
        console.print("[dim]nothing to run — pull it with --pull[/dim]")
        return None
    try:
        pull_with_progress(client, model, quiet=quiet or not console.is_terminal)
    except OllamaError as exc:
        console.print(f"[err]pull failed: {escape(str(exc))}[/err]")
        return None
    if not quiet:
        console.print(f"[ok]pulled[/ok] {escape(model)}")
    return model


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
