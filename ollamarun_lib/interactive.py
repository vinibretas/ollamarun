"""Interactive helpers: $EDITOR handoff, fuzzy model/session pickers (the
model picker sorts by name/size/parameters/context/modified), and reading a
model's context window from `/api/show`.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

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

if TYPE_CHECKING:
    from .ui import Ui


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


# ── model list: sorting and rendering ──────────────────────────────────────

#: Sort keys accepted by the model picker and `--sort`; `modified` (newest
#: first) is the default order.
MODEL_SORT_KEYS: tuple[str, ...] = ("modified", "name", "size", "parameters",
                                    "context")
#: Convenience spellings accepted in addition to the canonical keys.
MODEL_SORT_ALIASES = {
    "recent": "modified", "newest": "modified", "mtime": "modified",
    "param": "parameters", "params": "parameters", "ctx": "context",
    "context-length": "context", "contextlength": "context",
}

_PARAMETER_SIZE_RE = re.compile(
    r"^(?:(?P<count>\d+(?:\.\d+)?)\s*x\s*)?(?P<each>\d+(?:\.\d+)?)\s*(?P<unit>[BMK])$",
    re.IGNORECASE)
_PARAMETER_UNITS = {"b": 1.0, "m": 0.001, "k": 0.000001}


def parse_model_sort(text: Optional[str]) -> str:
    """Canonical sort key for `text`; ValueError names the valid keys."""
    key = (text or "").strip().lower()
    key = MODEL_SORT_ALIASES.get(key, key)
    if key not in MODEL_SORT_KEYS:
        raise ValueError(f"unknown sort key {text!r} — use "
                         + ", ".join(MODEL_SORT_KEYS))
    return key


def model_parameter_count(model: dict[str, Any]) -> Optional[float]:
    """Parameter count in billions from `/api/tags` `details.parameter_size`.

    Handles plain sizes ("7.6B", "70B", "134M") and mixture-of-experts totals
    ("8x7B" -> 56). Returns None when the tag has no usable value.
    """
    raw = str((model.get("details") or {}).get("parameter_size") or "").strip()
    match = _PARAMETER_SIZE_RE.match(raw)
    if not match:
        return None
    count = float(match.group("each")) * _PARAMETER_UNITS[match.group("unit").lower()]
    if match.group("count"):
        count *= float(match.group("count"))
    return count


def sort_models(models: list[dict[str, Any]], key: str,
                *, contexts: Optional[dict[str, Optional[int]]] = None
                ) -> list[dict[str, Any]]:
    """A new list ordered by `key` (one of MODEL_SORT_KEYS).

    `modified` keeps the original newest-first order; `name` is A→Z; the
    numeric keys list the largest first. Models without a value sort last,
    alphabetically among themselves.
    """
    if key == "name":
        return sorted(models, key=lambda item: str(item.get("name") or "").lower())
    if key == "modified":
        return sorted(models, key=lambda item: item.get("modified_at") or "",
                      reverse=True)
    if key == "size":
        value_of: Callable[[dict[str, Any]], Any] = lambda item: item.get("size")
    elif key == "parameters":
        value_of = model_parameter_count
    else:  # context
        known = contexts or {}
        value_of = lambda item: known.get(str(item.get("name") or ""))
    decorated: list[tuple[bool, float, str, int, dict[str, Any]]] = []
    for index, item in enumerate(models):
        value = value_of(item)
        decorated.append((value is None, -(float(value) if value else 0.0),
                          str(item.get("name") or "").lower(), index, item))
    decorated.sort(key=lambda entry: entry[:4])
    return [entry[4] for entry in decorated]


#: Context windows read from `/api/show`, keyed by (host, model) for the
#: process lifetime -- sorting by context must not re-query on every render.
_CONTEXT_CACHE: dict[tuple[str, str], Optional[int]] = {}


def cached_context_length(client: OllamaClient, model: str, *,
                          refresh: bool = False) -> Optional[int]:
    """The model's context window from `/api/show`, cached per (host, model)."""
    key = (client.base, model)
    if refresh or key not in _CONTEXT_CACHE:
        try:
            show = client.show(model)
        except OllamaError:
            show = {}
        _CONTEXT_CACHE[key] = model_context_length(show)
    return _CONTEXT_CACHE[key]


def fetch_context_lengths(client: OllamaClient, models: list[dict[str, Any]],
                          *, workers: int = 8) -> dict[str, Optional[int]]:
    """Context window per model name, fetched in parallel and cached.

    One `/api/show` per uncached model, behind a transient status line (the
    picker only calls this when a context sort is requested); a failure is
    cached as None, shown as "?" and sorted last.
    """
    names = [str(model.get("name")) for model in models if model.get("name")]
    missing = [name for name in names if (client.base, name) not in _CONTEXT_CACHE]
    if missing:
        def fetch() -> None:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(1, workers)) as pool:
                list(pool.map(lambda name: cached_context_length(client, name),
                              missing))

        if console.is_terminal:
            with console.status("[dim]reading context lengths…[/dim]", spinner="dots"):
                fetch()
        else:
            fetch()
    return {name: _CONTEXT_CACHE.get((client.base, name)) for name in names}


def _format_context(value: Optional[int]) -> str:
    if not value:
        return "?"
    return f"{value / 1000:.0f}k" if value >= 1000 else str(value)


def model_table(models: list[dict[str, Any]], current: Optional[str] = None,
                *, sort: str = "modified",
                contexts: Optional[dict[str, Optional[int]]] = None) -> Table:
    """The picker/list table: one row per model, in the given order.

    `params` always shows the tag's parameter size; a `ctx` column appears
    once contexts are known (i.e. after a context sort fetched them).
    """
    contexts = contexts or {}
    title = "models" if sort == "modified" else f"models · by {sort}"
    table = Table(title=title, box=box.ROUNDED, header_style="bold magenta",
                  row_styles=["", "dim"])
    table.add_column("#", justify="right")
    table.add_column("name")
    table.add_column("size", justify="right")
    table.add_column("params", justify="right")
    if contexts:
        table.add_column("ctx", justify="right")
    table.add_column("modified")
    table.add_column("family", style="dim")
    for index, model in enumerate(models, 1):
        details = model.get("details") or {}
        marker = " [ok]←[/ok]" if model.get("name") == current else ""
        row = [str(index), escape(model.get("name", "?")) + marker,
               human_bytes(model.get("size")),
               escape(str(details.get("parameter_size") or "?"))]
        if contexts:
            row.append(_format_context(contexts.get(str(model.get("name") or ""))))
        row += [short_stamp(model.get("modified_at")),
                escape(str(details.get("family") or ""))]
        table.add_row(*row)
    return table


def _parse_sort_command(text: str) -> tuple[bool, Optional[str]]:
    """(is_sort, key) for a picker input; key None means "show the usage".

    Accepts `--sort KEY`, `-s KEY`, `--sort=KEY` and `sort KEY` (the bare
    word `sort` stays a model name, so a model literally called "sort" is
    still selectable).
    """
    parts = text.replace("=", " ", 1).split()
    if not parts:
        return False, None
    head = parts[0].lower()
    if head not in {"--sort", "-s"} and not (head == "sort" and len(parts) > 1):
        return False, None
    if len(parts) < 2:
        return True, None
    try:
        return True, parse_model_sort(parts[1])
    except ValueError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return True, None


def pick_model(client: OllamaClient, current: Optional[str] = None, *,
               ui: Optional["Ui"] = None, sort: str = "modified",
               on_sort: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Show the installed models and return the chosen name (None on quit).

    The prompt takes a number, an exact name, a unique substring, or a sort
    command (`--sort KEY`, `-s KEY`, `sort KEY`) that re-orders the table in
    place and keeps the loop running; `on_sort` reports a sort chosen here so
    the caller can remember it.
    """
    try:
        models = client.tags()
    except OllamaError as exc:
        console.print(f"[err]{exc}[/err]")
        return None
    if not models:
        console.print("[warn]no models installed — try `ollama pull llama3.2`[/warn]")
        return None
    try:
        sort = parse_model_sort(sort)
    except ValueError:
        sort = "modified"
    contexts: dict[str, Optional[int]] = {}
    if sort == "context":
        contexts = fetch_context_lengths(client, models)
    models = sort_models(models, sort, contexts=contexts)
    console.print(model_table(models, current, sort=sort, contexts=contexts))
    prompt_text = "model number, name, --sort KEY or 'q' to quit"
    while True:
        if ui is not None:
            answer = (ui.ask(prompt_text, default="q") or "").strip()
        else:
            answer = Prompt.ask(prompt_text, default="q").strip()
        if answer in {"q", "", "quit"}:
            return None
        is_sort, key = _parse_sort_command(answer)
        if is_sort:
            if key is None:
                console.print("[dim]sort keys: " + ", ".join(MODEL_SORT_KEYS) + "[/dim]")
            elif key == sort:
                console.print(f"[dim]already sorted by {key}[/dim]")
            else:
                sort = key
                if on_sort is not None:
                    on_sort(sort)
                if sort == "context" and not contexts:
                    contexts = fetch_context_lengths(client, models)
                models = sort_models(models, sort, contexts=contexts)
                console.print(model_table(models, current, sort=sort,
                                          contexts=contexts))
            continue
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


def pick_session(store: SessionStore, filter_text: str = "", *,
                 ui: Optional["Ui"] = None) -> Optional[Session]:
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
        if ui is not None:
            answer = (ui.ask("session number, ID, filter or 'q' to quit",
                             default="q") or "").strip()
        else:
            answer = Prompt.ask("session number, ID, filter or 'q' to quit",
                                default="q").strip()
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
        return pick_session(store, filter_text=answer, ui=ui)


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
                 quiet: bool = False, interactive: Optional[bool] = None,
                 ui: Optional["Ui"] = None, sort: str = "modified") -> Optional[str]:
    """Return the model to run, pulling it first when it is not installed.

    `None` means the model is unavailable and the caller should stop. With
    `assume_yes` (`--pull`) a missing model is pulled without asking; on a
    terminal it is otherwise offered with a y/N confirmation after printing
    the target machine's free disk space. `pick_on_decline` opens the model
    picker for that machine when the user declines (ordered by `sort`);
    `quiet` suppresses the informational lines (`--raw`/`--json`/non-tty).
    `interactive` is only overridden by tests.
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
        if ui is not None:
            pull = ui.confirm(f"pull [key]{escape(model)}[/key]?", default=False)
        else:
            pull = Confirm.ask(f"pull [key]{escape(model)}[/key]?", default=False)
    if not pull:
        if interactive and pick_on_decline:
            chosen = (pick_model(client, model, ui=ui, sort=sort) if ui is not None
                      else pick_model(client, model, sort=sort))
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


def model_block_count(show_data: dict[str, Any]) -> Optional[int]:
    """Transformer block count from `/api/show`'s `<arch>.block_count`."""
    for key, value in (show_data.get("model_info") or {}).items():
        if (key == "block_count" or key.endswith(".block_count")) \
                and isinstance(value, int) and value > 0:
            return value
    return None


_LAYER_SPEC_RE = re.compile(r"^(?P<sign>[+-])?(?P<number>\d+)(?P<percent>%)?$")

GPU_LAYERS_USAGE = ("N (absolute), P% (of the model's layers), +N/-N (adjust the "
                    "current value), all, cpu or auto")


def parse_gpu_layers_spec(text: Optional[str]) -> tuple[str, Optional[int]]:
    """('abs'|'pct'|'delta'|'all'|'cpu'|'auto', number) for a gpu-layers spec."""
    spec = (text or "").strip().lower()
    if spec in {"auto", "default", "dynamic"}:
        return "auto", None
    if spec == "all":
        return "all", None
    if spec in {"cpu", "none"}:
        return "cpu", None
    match = _LAYER_SPEC_RE.match(spec)
    if not match:
        raise ValueError(f"invalid gpu layers {text!r} — use {GPU_LAYERS_USAGE}")
    number = int(match.group("number"))
    if match.group("percent"):
        if match.group("sign"):
            raise ValueError("a percentage cannot have a sign (use N% alone)")
        if number > 100:
            raise ValueError("a percentage must be between 0 and 100")
        return "pct", number
    if match.group("sign") == "-":
        return "delta", -number
    if match.group("sign") == "+":
        return "delta", number
    return "abs", number


def resolve_gpu_layers(spec: Optional[str], block_count: Optional[int],
                       current: Optional[int] = None) -> Optional[int]:
    """Resolve a gpu-layers spec to `num_gpu` (None = leave it automatic).

    Percent resolves against `block_count + 1` (the value Ollama treats as a
    full offload, including the output layer). A delta adjusts `current`, or
    a full offload when no value is set yet, so `-4` means "all layers except
    the last 4". Raises ValueError for a bad spec and RuntimeError when the
    spec cannot be resolved without knowing the model's layer count.
    """
    if spec is None:
        return None
    kind, number = parse_gpu_layers_spec(spec)
    if kind == "auto":
        return None
    if kind == "cpu":
        return 0
    if kind == "all":
        return block_count + 1 if block_count is not None else 999
    if kind == "abs":
        return number
    if kind == "pct":
        if block_count is None:
            raise RuntimeError(f"cannot resolve {spec} without the model's layer count")
        full = block_count + 1
        return (number * full + 50) // 100
    if kind == "delta":
        if current is None:
            if block_count is None:
                raise RuntimeError(f"cannot resolve {spec} without the model's layer count")
            base = block_count + 1
        else:
            base = current
        value = max(0, base + number)
        if block_count is not None:
            value = min(value, block_count + 1)
        return value
    return None


def gpu_layers_label(spec: Optional[str], block_count: Optional[int]) -> str:
    """One-line status for a gpu-layers spec, e.g. '50% → 17 of 37 layers'."""
    if not spec:
        return "auto (ollama decides)"
    try:
        value = resolve_gpu_layers(spec, block_count)
    except (ValueError, RuntimeError):
        return f"{spec} (unresolved — layer count unavailable)"
    if value is None:
        return "auto (ollama decides)"
    if value == 0:
        return f"{spec} → cpu only"
    if block_count is None:
        return f"{spec} → all layers" if value >= 999 else f"{spec} → {value} layers"
    return f"{spec} → {min(value, block_count + 1)} of {block_count + 1} layers"
