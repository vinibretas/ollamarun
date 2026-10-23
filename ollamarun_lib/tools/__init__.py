"""Tool registry, safety policy and executor for the agent loop.

`ToolSpec` describes one tool (name, description, JSON-schema parameters and
a handler); `default_registry()` builds the built-in set (shell, python,
files, web -- `web_search` when `ddgs` is installed, plus `fetch_url`, which
returns readable page text rather than raw markup -- and, when `tts-remote`
is available, speech) from the definition files in `defs/tools/`, one JSON
per tool, with `~/etc/config_files/ollamarun-defs/tools/` overriding the
shipped files by name.
`approval_decision()` classifies a call as auto-approved or needing user
confirmation (read-only tools and provably-harmless shell/python calls are
safe; anything else asks), and `execute_tool()` runs the handler with a
timeout and caps the text that goes back to the model.

The confirmation prompt can store generalized, session-scoped `ToolRule`s
-- `shell: mkdir -p :*`, `path: ~/proj/**`, `python: imports os,sys` --
suggested by `suggest_rules()` from the call being asked about and tested
by `rule_matches()`. `dangerous_command()` is the floor no rule ever
covers: substitutions, redirections, privilege/wrapper/interpreter
commands, disk-level tools, catastrophic `rm` and pipes into an
interpreter always prompt again.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

from ..constants import DEFS_DIR, DEFS_OVERRIDE_DIR, console
from ..client import OllamaError
from ..helpers import (canonical_model_name, diff_preview, host_is_self,
                       human_bytes, is_local_host, local_free_disk,
                       local_gpu_probe, local_ram_probe, run_cancellable,
                       short_stamp, ssh_free_disk, ssh_gpu_probe,
                       ssh_ram_probe)

from .spec import (MAX_PDF_VISION_PAGES, TRUNCATE_TAIL, Decision, ToolCancelled,
                   ToolContext, ToolError, ToolHandlerResult, ToolOutcome,
                   ToolSpec, _clamp_timeout, truncate_output)
from .policy import (DEFAULT_SHELL_ALLOW, GIT_CONFIG_SAFE, GIT_READONLY,
                     PYTHON_DANGEROUS_ATTRS, PYTHON_DANGEROUS_CALLS,
                     PYTHON_DANGEROUS_NAMES, SAFE_PYTHON_MODULES,
                     approval_decision, python_import_roots, python_safety,
                     shell_safety)
from .args import call_detail, call_fingerprint, normalize_arguments
from .rules import (PATH_RULE_TOOLS, RULE_FLOOR_COMMANDS, RULE_KINDS,
                    SHELL_INTERPRETERS, PolicyCheck, ToolRule,
                    dangerous_command, matching_rule, policy_check, rule_label,
                    rule_matches, suggest_rules)
from .workspace import _resolve_in_workspace, _workspace_base
from .html import MAX_PAGE_LINKS, extract_html_text
from .pdf import (MAX_PDF_BYTES, MAX_PDF_CHARS, MAX_PDF_PAGES, PDF_VISION_DPI,
                  PDF_VISION_PREFERENCE, PDF_VISION_PROMPT,
                  PDF_VISION_TIMEOUT, extract_pdf_text, parse_page_spec,
                  pdf_page_count, pick_vision_model, render_pdf_page,
                  select_pdf_pages, vision_models_installed, vision_transcribe)
from .net import (BOT_WALL_MARKERS, BROWSER_HEADERS, DEFAULT_SEARCH_RESULTS,
                  FETCH_TIMEOUT, JS_SHELL_MARKERS, MAX_ERROR_DETAIL,
                  MAX_SEARCH_RESULTS, SEARCH_ENGINE_HOSTS, SEARCH_TIMEOUT,
                  SOFT_BLOCK_MARKERS, format_fetch_response,
                  search_engine_query)
from .web_search import (DDGS, _handler_web_search, format_search_results,
                         run_web_search, search_available)
from .fetch_url import _handler_fetch_url
from .read_pdf import _handler_read_pdf

MAX_LIST_ENTRIES = 200
MAX_READ_CHARS = 200_000


#: Tool names models tend to invent for the web tools are mapped onto the
#: real ones through each definition's `aliases` list (see the defs loader
#: below); `tool_aliases()` builds that map, `resolve_tool_alias()` uses it.
#: Tool-call ids and namespaces also leak into the function name itself
#: (Ollama's `call_` ids, OpenAI's `functions.`/`tool.` styles), so an
#: unknown name is retried once with that prefix stripped -- `call_fetch_url`
#: still runs `fetch_url` instead of failing as an unknown tool.
_TOOL_NAME_PREFIXES = ("call_", "call.", "tool_", "tool.",
                       "functions.", "function.")


def _strip_tool_name_prefix(key: str) -> str:
    """The name left after one conventional tool-call prefix (else '')."""
    for prefix in _TOOL_NAME_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return ""


def resolve_tool_alias(name: str) -> str:
    """The real tool for a name a model may have invented (identity otherwise).

    The alias map is tried first; on a miss the name is stripped of a
    conventional id/namespace prefix and retried against the aliases and
    the canonical definition names, so `call_fetch_url` -> `fetch_url` and
    `tool_run_shell` -> `run_shell`.
    """
    key = str(name or "").strip().lower().replace("-", "_")
    aliases = tool_aliases()
    match = aliases.get(key)
    if match:
        return match
    stripped = _strip_tool_name_prefix(key)
    if stripped:
        match = aliases.get(stripped)
        if match:
            return match
        if stripped in {definition.name for definition in tool_definitions()}:
            return stripped
    return name



# ── handlers ────────────────────────────────────────────────────────────────


def _handler_shell(args: dict[str, Any], ctx: ToolContext) -> str:
    command = str(args.get("command") or "").strip()
    if not command:
        raise ToolError("missing 'command'")
    cwd = _resolve_in_workspace(
        str(args.get("cwd") or getattr(ctx, "workspace", None) or ctx.cwd),
        ctx)
    if not cwd.is_dir():
        raise ToolError(f"working directory does not exist: {cwd}")
    timeout = _clamp_timeout(args.get("timeout"), ctx.timeout)
    environment = {**os.environ, "BASH_ENV": os.devnull}
    try:
        proc = run_cancellable(["bash", "-c", command], cwd=str(cwd),
                               env=environment, timeout=timeout,
                               cancel=getattr(ctx, "cancel", None))
    except subprocess.TimeoutExpired:
        raise ToolError(f"command timed out after {timeout:.0f}s") from None
    except OSError as exc:
        raise ToolError(f"could not run command: {exc}") from exc
    return _combine_exit(proc.returncode, proc.stdout, proc.stderr)


def _handler_python(args: dict[str, Any], ctx: ToolContext) -> str:
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ToolError("missing 'code'")
    timeout = _clamp_timeout(args.get("timeout"), ctx.timeout)
    workdir = _workspace_base(ctx)
    try:
        proc = run_cancellable([sys.executable, "-I", "-"], input_text=code,
                               cwd=str(workdir), timeout=timeout,
                               cancel=getattr(ctx, "cancel", None))
    except subprocess.TimeoutExpired:
        raise ToolError(f"python timed out after {timeout:.0f}s") from None
    except OSError as exc:
        raise ToolError(f"could not run python: {exc}") from exc
    return _combine_exit(proc.returncode, proc.stdout, proc.stderr)


def _combine_exit(returncode: int, stdout: str, stderr: str) -> str:
    parts = []
    if stdout.strip():
        parts.append(stdout.rstrip("\n"))
    if stderr.strip():
        parts.append("[stderr]\n" + stderr.rstrip("\n"))
    body = "\n".join(parts) or "(no output)"
    return f"exit {returncode}\n{body}"


def _handler_read_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path = _resolve_in_workspace(str(args.get("path") or ""), ctx)
    if not path.is_file():
        raise ToolError(f"not a file: {path}")
    try:
        with path.open("rb") as handle:
            head = handle.read(5)
    except OSError as exc:
        raise ToolError(f"could not read {path}: {exc}") from exc
    if head == b"%PDF-":
        raise ToolError(
            f"{path} is a PDF, which read_file cannot show as text; call "
            f"read_pdf with this path (it extracts the text layer and reads "
            f"scans with a vision model)")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(MAX_READ_CHARS)
    except OSError as exc:
        raise ToolError(f"could not read {path}: {exc}") from exc
    lines = text.splitlines()
    try:
        offset = max(0, int(args.get("offset") or 0))
    except (TypeError, ValueError):
        offset = 0
    limit = args.get("limit")
    try:
        limit = max(1, int(limit)) if limit is not None else None
    except (TypeError, ValueError):
        limit = None
    window = lines[offset:offset + limit] if limit else lines[offset:]
    numbered = "\n".join(f"{offset + index + 1}: {line}"
                         for index, line in enumerate(window))
    return f"{path} ({len(lines)} lines shown)\n{numbered}"



def _handler_write_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path = _resolve_in_workspace(str(args.get("path") or ""), ctx)
    if "content" not in args:
        raise ToolError("missing 'content'")
    content = str(args.get("content") or "")
    append = bool(args.get("append"))
    try:
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        raise ToolError(f"could not write {path}: {exc}") from exc
    verb = "appended" if append else "wrote"
    return ToolHandlerResult(f"{verb} {len(content)} chars to {path}", [str(path)])


def _handler_list_dir(args: dict[str, Any], ctx: ToolContext) -> str:
    path = _resolve_in_workspace(
        str(args.get("path") or getattr(ctx, "workspace", None) or ctx.cwd),
        ctx)
    if not path.is_dir():
        raise ToolError(f"not a directory: {path}")
    try:
        entries = sorted(path.iterdir(), key=lambda entry: (not entry.is_dir(),
                                                            entry.name.lower()))
    except OSError as exc:
        raise ToolError(f"could not list {path}: {exc}") from exc
    lines = [f"{path} ({len(entries)} entries)"]
    for entry in entries[:MAX_LIST_ENTRIES]:
        try:
            info = entry.stat()
            detail = "dir" if entry.is_dir() else f"{info.st_size} B"
            stamp = short_stamp(time.strftime("%Y-%m-%dT%H:%M:%S",
                                             time.localtime(info.st_mtime)))
        except OSError:
            detail, stamp = "?", "?"
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{entry.name}{suffix}  ({detail}, {stamp})")
    if len(entries) > MAX_LIST_ENTRIES:
        lines.append(f"… {len(entries) - MAX_LIST_ENTRIES} more entries")
    return "\n".join(lines)


def _handler_edit_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path = _resolve_in_workspace(str(args.get("path") or ""), ctx)
    old = args.get("old_string")
    new = args.get("new_string")
    if not str(path) or str(path) == ".":
        raise ToolError("missing 'path'")
    if not isinstance(old, str) or not old:
        raise ToolError("missing 'old_string' (must be a non-empty exact fragment)")
    if not isinstance(new, str):
        raise ToolError("missing 'new_string'")
    if not path.is_file():
        raise ToolError(f"not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"not a utf-8 text file: {path}") from exc
    except OSError as exc:
        raise ToolError(f"could not read {path}: {exc}") from exc
    count = text.count(old)
    if count == 0:
        raise ToolError(f"old_string not found in {path} — read the file first and "
                        f"copy the exact text to replace")
    replace_all = bool(args.get("replace_all"))
    if count > 1 and not replace_all:
        raise ToolError(f"old_string appears {count} times in {path}; include more "
                        f"surrounding context or pass replace_all=true")
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    tmp_name: Optional[str] = None
    try:
        mode = path.stat().st_mode
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                        dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
        os.chmod(tmp_name, mode & 0o7777)
        os.replace(tmp_name, path)
        tmp_name = None
    except OSError as exc:
        raise ToolError(f"could not write {path}: {exc}") from exc
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    replaced = count if replace_all else 1
    if replaced == 1:
        first = text[:text.find(old)].count("\n") + 1
        where = f" at line {first}"
    else:
        where = ""
    diff = diff_preview(old, new, limit=30)
    result = f"edited {path}: replaced {replaced} occurrence(s){where}"
    text = f"{result}\n{diff}" if diff else result
    return ToolHandlerResult(text, [str(path)])


def _handler_make_dir(args: dict[str, Any], ctx: ToolContext) -> str:
    raw = str(args.get("path") or "").strip()
    if not raw:
        raise ToolError("missing 'path'")
    path = _resolve_in_workspace(raw, ctx)
    if path.is_dir():
        return f"directory already exists: {path}"
    if path.exists():
        raise ToolError(f"a file exists at {path}")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolError(f"could not create {path}: {exc}") from exc
    return f"created directory {path}"


def _transfer_paths(args: dict[str, Any],
                    ctx: ToolContext) -> tuple[Path, Path, bool]:
    source = str(args.get("source") or "").strip()
    destination = str(args.get("destination") or "").strip()
    if not source:
        raise ToolError("missing 'source'")
    if not destination:
        raise ToolError("missing 'destination'")
    return (_resolve_in_workspace(source, ctx),
            _resolve_in_workspace(destination, ctx),
            bool(args.get("overwrite")))


def _handler_move_file(args: dict[str, Any], ctx: ToolContext) -> str:
    source, destination, overwrite = _transfer_paths(args, ctx)
    if not source.exists() and not source.is_symlink():
        raise ToolError(f"source does not exist: {source}")
    if destination.exists() or destination.is_symlink():
        if destination.is_dir():
            raise ToolError(f"destination is an existing directory: {destination}")
        if not overwrite:
            raise ToolError(f"destination exists: {destination} (pass overwrite=true "
                            f"to replace it)")
    try:
        shutil.move(str(source), str(destination))
    except (OSError, shutil.Error) as exc:
        raise ToolError(f"could not move {source} to {destination}: {exc}") from exc
    message = f"moved {source} → {destination}"
    if destination.is_file():  # directories are never recursive bundles
        return ToolHandlerResult(message, [str(destination)])
    return message


def _handler_copy_file(args: dict[str, Any], ctx: ToolContext) -> str:
    source, destination, overwrite = _transfer_paths(args, ctx)
    if not source.exists() and not source.is_symlink():
        raise ToolError(f"source does not exist: {source}")
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise ToolError(f"destination exists: {destination} (pass overwrite=true "
                            f"to replace it)")
        if destination.is_dir():
            raise ToolError(f"destination is an existing directory: {destination}")
    try:
        if source.is_dir():
            shutil.copytree(str(source), str(destination))
        else:
            shutil.copy2(str(source), str(destination))
    except (OSError, shutil.Error) as exc:
        raise ToolError(f"could not copy {source} to {destination}: {exc}") from exc
    message = f"copied {source} → {destination}"
    if not source.is_dir() and destination.is_file():
        return ToolHandlerResult(message, [str(destination)])
    return message


def _handler_publish_artifact(args: dict[str, Any], ctx: ToolContext) -> "str | ToolHandlerResult":
    """Snapshot one workspace file as a downloadable artifact.

    The explicit path for files created by `run_python`/`run_shell` (the
    file tools publish their own results automatically). Jailed to the
    workspace root (the API) or the working directory (the CLI), and capped
    by the store's per-file limit.
    """
    raw = str(args.get("path") or "").strip()
    if not raw:
        raise ToolError("missing 'path'")
    publisher = getattr(ctx, "artifacts", None)
    if publisher is None:
        raise ToolError("publishing artifacts needs the serve artifact store")
    base = Path(getattr(ctx, "workspace", None) or ctx.cwd).resolve()
    candidate = Path(raw).expanduser()
    target = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if target != base and base not in target.parents:
        raise ToolError(f"refused: {raw} is outside the working directory")
    if not target.is_file():
        raise ToolError(f"no such file: {raw}")
    limit = publisher.store.max_bytes
    if 0 < limit < target.stat().st_size:
        raise ToolError(f"refused: {raw} exceeds the {limit} byte artifact limit")
    return ToolHandlerResult(f"published {raw} as a downloadable artifact",
                             [str(target)])


def _handler_delete_file(args: dict[str, Any], ctx: ToolContext) -> str:
    raw = str(args.get("path") or "").strip()
    if not raw:
        raise ToolError("missing 'path'")
    path = _resolve_in_workspace(raw, ctx)
    if not path.exists() and not path.is_symlink():
        raise ToolError(f"no such file or directory: {path}")
    trash = shutil.which("trash-put")
    if not trash:
        raise ToolError("trash-put not found; refusing to delete without a trash")
    try:
        proc = run_cancellable([trash, str(path)], timeout=ctx.timeout,
                               cancel=getattr(ctx, "cancel", None))
    except subprocess.TimeoutExpired:
        raise ToolError("trash-put timed out") from None
    except OSError as exc:
        raise ToolError(f"could not run trash-put: {exc}") from exc
    if proc.returncode != 0:
        raise ToolError(f"trash-put failed: "
                        f"{(proc.stderr or proc.stdout).strip() or 'unknown error'}")
    return f"moved to trash: {path}"


def _handler_search_sessions(args: dict[str, Any], ctx: ToolContext) -> str:
    store = ctx.store
    if store is None:
        raise ToolError("session search is unavailable in this run")
    query = str(args.get("query") or "").strip()
    if not query:
        raise ToolError("missing 'query'")
    try:
        limit = int(args.get("limit") or 5)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(limit, 20))
    try:
        results = store.search(query, limit)
    except OSError as exc:
        raise ToolError(f"could not search sessions: {exc}") from exc
    if not results:
        return f"no saved session matches {query!r}"
    lines = [f"{len(results)} saved session(s) matching {query!r}:"]
    for session, snippet in results:
        title = " ".join(str(session.display_title()).split())
        lines.append(f"- {session.id} · {title} · model {session.model or '?'} "
                     f"· updated {short_stamp(session.updated_at)}")
        lines.append(f"  …{' '.join(str(snippet).split())}")
    lines.append("(Use /show SESSION in the REPL to read a full conversation.)")
    return "\n".join(lines)


def _handler_list_models(args: dict[str, Any], ctx: ToolContext) -> str:
    client = ctx.client
    if client is None:
        raise ToolError("no Ollama server connection in this run")
    loaded_only = bool(args.get("loaded_only"))
    try:
        loaded = client.ps()
        installed = [] if loaded_only else client.tags()
    except OllamaError as exc:
        raise ToolError(str(exc)) from exc
    by_name = {canonical_model_name(entry.get("name", "")): entry for entry in loaded}
    if loaded_only:
        if not loaded:
            return "no models currently loaded"
        lines = [f"{len(loaded)} model(s) loaded on {client.base}:"]
        for entry in loaded:
            lines.append("- " + _resident_summary(entry))
        return "\n".join(lines)
    if not installed:
        return f"no models installed on {client.base}"
    lines = [f"{len(installed)} model(s) on {client.base} ({len(loaded)} loaded):"]
    for entry in installed:
        name = str(entry.get("name") or "?")
        details = entry.get("details") or {}
        bits = [name]
        if entry.get("size"):
            bits.append(human_bytes(entry["size"]))
        if details.get("parameter_size"):
            bits.append(str(details["parameter_size"]))
        if details.get("quantization_level"):
            bits.append(str(details["quantization_level"]))
        if entry.get("modified_at"):
            bits.append("modified " + short_stamp(entry["modified_at"]))
        resident = by_name.get(canonical_model_name(name))
        if resident is not None:
            bits.append("loaded (" + ", ".join(_resident_details(resident)) + ")")
        lines.append("- " + " · ".join(bits))
    return "\n".join(lines)


def _resident_summary(entry: dict[str, Any]) -> str:
    """`name · X VRAM · unloads at TIME` for one /api/ps entry."""
    name = str(entry.get("name") or entry.get("model") or "?")
    return " · ".join([name, *_resident_details(entry)])


def _resident_details(entry: dict[str, Any]) -> list[str]:
    """The VRAM/keep-alive bits of one /api/ps entry, without the name."""
    bits = []
    size = entry.get("size")
    vram = entry.get("size_vram")
    if vram:
        bit = f"{human_bytes(vram)} VRAM"
        if size and size > vram:
            bit += f" of {human_bytes(size)}"
        bits.append(bit)
    elif size:
        bits.append(human_bytes(size))
    if entry.get("expires_at"):
        bits.append("unloads " + short_stamp(entry["expires_at"]))
    return bits


def _handler_system_status(args: dict[str, Any], ctx: ToolContext) -> str:
    target = str(args.get("target") or "ollama").strip().lower()
    if target not in {"ollama", "local"}:
        raise ToolError("'target' must be 'ollama' or 'local'")
    base = str(getattr(ctx.client, "base", "") or "")
    ssh_target = None
    if target == "ollama" and base and not (host_is_self(base)
                                            or is_local_host(base)):
        if not ctx.ssh_target:
            raise ToolError(
                f"the Ollama host {base} is remote; run with --remote NAME so "
                f"its resources can be probed over ssh, or pass target=\"local\" "
                f"for this machine")
        ssh_target = ctx.ssh_target
    if target == "local":
        label = "this machine"
    elif ssh_target:
        label = f"{ctx.ssh_target} (ollama at {base})"
    elif base:
        label = f"this machine (ollama at {base})"
    else:
        label = "this machine"
    lines = [label]

    if ssh_target:
        probe_cancel = getattr(ctx, "cancel", None)
        ram = ssh_ram_probe(ssh_target, user=ctx.ssh_user, cancel=probe_cancel)
        gpu = ssh_gpu_probe(ssh_target, user=ctx.ssh_user, cancel=probe_cancel)
        disk = ssh_free_disk(ssh_target, user=ctx.ssh_user, cancel=probe_cancel)
    else:
        ram = local_ram_probe()
        gpu = local_gpu_probe()
        disk = local_free_disk()
    if ram:
        total, available = ram
        lines.append(f"RAM: {human_bytes(available)} free of {human_bytes(total)}")
    else:
        lines.append("RAM: unknown")
    devices = gpu[0] if gpu else []
    if devices:
        for index, (total_mb, free_mb) in enumerate(devices):
            lines.append(f"GPU {index}: {human_bytes(free_mb * 1024 * 1024)} free "
                         f"of {human_bytes(total_mb * 1024 * 1024)}")
    else:
        lines.append("GPU: no nvidia-smi data")
    if disk:
        free, total, where = disk
        lines.append(f"disk (models): {human_bytes(free)} free of {human_bytes(total)} "
                     f"at {where}")
    else:
        lines.append("disk (models): unknown")
    return "\n".join(lines)


# ── environment awareness (context_status / session_info) ───────────────────


def _handler_context_status(args: dict[str, Any], ctx: ToolContext) -> str:
    report = ctx.context_report
    if report is None:
        raise ToolError("context status is unavailable in this run")
    data = report()
    model = str(data.get("model") or "?")
    base = str(data.get("base") or "")
    remote = data.get("remote")
    where = (f"{remote} (ollama at {base})" if remote else
             f"this machine ({base})" if base else "this machine")
    lines = [f"model: {model} on {where}"]
    window = data.get("window")
    model_max = data.get("model_max")
    if window is None:
        lines.append("context window: unknown (server unreachable or the model "
                     "is not loaded yet)")
    else:
        source = str(data.get("source") or "unknown")
        detail = f"{window} tokens ({source}"
        if model_max and model_max != window:
            detail += f"; model default {model_max}"
        detail += ")"
        lines.append(f"context window: {detail}")
    used = data.get("used")
    prompt = data.get("prompt")
    completion = data.get("completion")
    if used is None:
        lines.append("context used: nothing yet (no request has been sent)")
    else:
        lines.append(f"context used: {used} tokens = last prompt {prompt or 0} "
                     f"+ last reply {completion or 0}")
        if window:
            free = max(0, window - used)
            lines.append(f"free: {free} tokens ({used * 100 / window:.1f}% used)")
    counts = data.get("messages") or {}
    total = int(counts.get("total") or 0)
    if total:
        parts = [f"{role} {count}" for role, count in sorted(counts.items())
                 if role != "total" and count]
        lines.append(f"messages: {total}" + (f" ({' · '.join(parts)})" if parts else ""))
    if used and window and used / window >= 0.8:
        lines.append("note: the context window is nearly full — keep this answer "
                     "short or offer to summarize older messages")
    return "\n".join(lines)


def _handler_session_info(args: dict[str, Any], ctx: ToolContext) -> str:
    report = ctx.session_report
    if report is None:
        raise ToolError("session info is unavailable in this run")
    data = report()
    now = datetime.now().astimezone()
    lines = [f"now: {now.strftime('%Y-%m-%d %H:%M %z')} ({now.strftime('%A')})"]
    session = data.get("session") or {}
    if session:
        lines.append(
            f"session: {session.get('id')} \"{session.get('title')}\" · "
            f"{session.get('messages', 0)} messages · {session.get('turns', 0)} turns"
            f" · started {short_stamp(session.get('created_at'))}")
    model_bits = [str(data.get("model") or "?")]
    details = data.get("details") or {}
    if details.get("parameter_size"):
        model_bits.append(str(details["parameter_size"]))
    if details.get("quantization_level"):
        model_bits.append(str(details["quantization_level"]))
    capabilities = details.get("capabilities") or []
    if capabilities:
        model_bits.append("capabilities: " + ", ".join(str(cap) for cap in capabilities))
    lines.append("model: " + " · ".join(model_bits))
    base = str(data.get("base") or "")
    remote = data.get("remote")
    where = (f"{remote} ({base})" if remote else
             f"this machine ({base})" if base else "?")
    lines.append(f"server: {where}")
    cwd = data.get("cwd") or "?"
    lines.append(f"tools run on: {socket.gethostname()} ({sys.platform}) · cwd {cwd}")
    settings = []
    parameters = data.get("parameters") or {}
    for key in sorted(parameters)[:8]:
        settings.append(f"{key} {parameters[key]}")
    keepalive = data.get("keepalive")
    settings.append("keepalive " + (str(keepalive) if keepalive else "default"))
    if data.get("gpu_layers"):
        settings.append("gpu layers " + str(data["gpu_layers"]))
    lines.append("settings: " + " · ".join(settings))
    lines.append(f"modes: thinking {data.get('thinking')} · render {data.get('render')} "
                 f"· system prompt {data.get('system_prompt')}")
    tools = data.get("tools") or {}
    tool_bits = ["on" if tools.get("enabled") else "off",
                 f"policy {tools.get('policy')}",
                 f"round {tools.get('round')} of {tools.get('max_rounds')}"]
    disabled = tools.get("disabled") or []
    tool_bits.append("disabled: " + (", ".join(disabled) if disabled else "none"))
    lines.append("tools: " + " · ".join(tool_bits))
    return "\n".join(lines)


# ── chem inventory (chemInventoryAPI) ───────────────────────────────────────


def run_chem_inventory(binary: str, argv: list[str], timeout: float,
                       cancel: Optional[Any] = None) -> Any:
    """Run `chemInventoryAPI … --json` and parse the JSON it prints."""
    try:
        proc = run_cancellable([binary, *argv, "--json"], timeout=timeout,
                               cancel=cancel)
    except subprocess.TimeoutExpired:
        raise ToolError(f"chemInventoryAPI timed out after {timeout:.0f}s") from None
    except OSError as exc:
        raise ToolError(f"could not run chemInventoryAPI: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        if len(detail) > 400:
            detail = detail[:399] + "…"
        raise ToolError(f"chemInventoryAPI failed: {detail or 'unknown error'}")
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        raise ToolError(f"chemInventoryAPI returned invalid JSON: {exc}") from exc


def _chem_run(ctx: ToolContext, argv: list[str]) -> Any:
    binary = find_chem_inventory(ctx)
    if not binary:
        raise ToolError("chemInventoryAPI not found (PATH or config chem_inventory)")
    return run_chem_inventory(binary, argv, min(ctx.timeout, 90),
                              cancel=getattr(ctx, "cancel", None))


def _flatten(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(item) for item in value)
    return " ".join(str(value).split())


def _chem_row(row: dict[str, Any]) -> str:
    """One compact `#id · name · CAS · amount · location` line from a record."""
    bits = []
    if row.get("id") is not None:
        bits.append(f"#{row['id']}")
    name = row.get("name") or row.get("containername")
    if name:
        bits.append(_flatten(name))
    if row.get("cas"):
        bits.append(f"CAS {_flatten(row['cas'])}")
    amount = " ".join(str(part) for part in (row.get("size"), row.get("unit")) if part)
    if amount.strip():
        bits.append(amount.strip())
    location = row.get("locationtext") or row.get("locationString")
    if location:
        bits.append(f"at {_flatten(location)}")
    if row.get("barcode"):
        bits.append(f"barcode {_flatten(row['barcode'])}")
    if row.get("supplier"):
        bits.append(f"from {_flatten(row['supplier'])}")
    return " · ".join(bits) or "(empty record)"


def _handler_chem_search(args: dict[str, Any], ctx: ToolContext) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ToolError("missing 'query'")
    argv = ["search", query]
    kind = str(args.get("type") or "").strip()
    if kind:
        argv += ["--type", kind]
    payload = _chem_run(ctx, argv)
    if not isinstance(payload, dict):
        payload = {}
    containers = payload.get("containers") or []
    orders = payload.get("orders") or []
    if not containers and not orders:
        return f"no containers match {query!r}"
    try:
        limit = int(args.get("limit") or 10)
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(limit, 50))
    lines = [f"{len(containers)} container(s) matching {query!r}:"]
    for row in containers[:limit]:
        lines.append("- " + _chem_row(row))
    if len(containers) > limit:
        lines.append(f"… {len(containers) - limit} more container(s) (raise limit)")
    if orders:
        lines.append(f"{len(orders)} matching order request(s)")
    return "\n".join(lines)


def _handler_chem_container(args: dict[str, Any], ctx: ToolContext) -> str:
    raw = args.get("id")
    if raw is None or not str(raw).strip():
        raise ToolError("missing 'id'")
    data = _chem_run(ctx, ["container", "info", str(raw)])
    if not isinstance(data, dict) or not data:
        return f"container {raw}: nothing found"
    lines = [f"container #{data.get('id', raw)}"]
    pairs = [
        ("name", data.get("containername") or data.get("name")),
        ("CAS", data.get("cas")),
        ("location", data.get("locationString") or data.get("locationtext")),
        ("amount", " ".join(str(part) for part in (data.get("size"), data.get("unit"))
                            if part).strip()),
        ("supplier", data.get("supplier")),
        ("acquired", data.get("dateacquired")),
        ("barcode", data.get("barcode")),
        ("formula", data.get("molecularformula")),
        ("MW", data.get("molecularweight")),
        ("exact mass", data.get("exactmass")),
        ("IUPAC", data.get("iupacname")),
        ("SMILES", data.get("isomericsmiles")),
        ("comments", data.get("comments")),
    ]
    for label, value in pairs:
        if value not in (None, "", "N/A"):
            lines.append(f"{label}: {_flatten(value)}")
    for prefix, header in (("cf-", "custom fields"), ("sf-", "substance fields")):
        fields = [(key[len(prefix):], value) for key, value in data.items()
                  if key.startswith(prefix) and not key.endswith("-value")
                  and value not in (None, "")]
        if fields:
            lines.append(header + ": " + "; ".join(
                f"{key}={_flatten(value)}" for key, value in fields[:20]))
    codes = []
    for entry in data.get("ghs") or []:
        codes += [code for code in str(entry.get("hazardstatements") or "").split("|")
                  if code]
    if codes:
        lines.append("GHS: " + ", ".join(codes))
    if data.get("needsretrieval"):
        lines.append("(substance info not retrieved yet)")
    return "\n".join(lines)


def _handler_chem_locations(args: dict[str, Any], ctx: ToolContext) -> str:
    location = str(args.get("location") or "").strip()
    if location:
        payload = _chem_run(ctx, ["location", "containers", location])
        containers = payload.get("containers") or [] if isinstance(payload, dict) else []
        if not containers:
            return f"no containers in {location!r}"
        lines = [f"{len(containers)} container(s) in {location!r}:"]
        for row in containers[:40]:
            lines.append("- " + _chem_row(row))
        if len(containers) > 40:
            lines.append(f"… {len(containers) - 40} more container(s)")
        return "\n".join(lines)
    payload = _chem_run(ctx, ["location", "list"])
    entries = payload if isinstance(payload, list) else []
    if not entries:
        return "no locations"
    children: dict[Any, list] = {}
    for loc in entries:
        children.setdefault(loc.get("parent") or 0, []).append(loc)
    for bucket in children.values():
        bucket.sort(key=lambda item: _flatten(item.get("name") or "").lower())
    ordered: list[tuple[int, dict[str, Any]]] = []
    seen: set[Any] = set()

    def walk(parent: Any, depth: int) -> None:
        for loc in children.get(parent, []):
            if loc.get("id") in seen:
                continue
            seen.add(loc.get("id"))
            ordered.append((depth, loc))
            walk(loc.get("id"), depth + 1)

    walk(0, 0)
    for loc in entries:  # orphans (parent id not present) still get listed
        if loc.get("id") not in seen:
            ordered.append((0, loc))
    lines = [f"{len(entries)} location(s) (id · containers):"]
    for depth, loc in ordered:
        name = _flatten(loc.get("name") or "?")
        count = loc.get("numbercontainers", 0)
        lines.append(f"- {'  ' * depth}{name} (#{loc.get('id')}, {count})")
    return "\n".join(lines)



# ── tts-remote (optional) ───────────────────────────────────────────────────


def find_tts_remote(ctx: ToolContext) -> Optional[str]:
    candidate = ctx.tts_remote or "tts-remote"
    found = shutil.which(candidate) if os.path.sep not in candidate else (
        candidate if Path(candidate).exists() else None)
    if found:
        return found
    # tts-remote lives in its own submodule: ~/bin/tts-remote/tts-remote
    fallback = Path.home() / "bin" / "tts-remote"
    if fallback.is_file():
        return str(fallback)
    entry = fallback / "tts-remote"
    return str(entry) if entry.is_file() else None


def find_chem_inventory(ctx: ToolContext) -> Optional[str]:
    """The chemInventoryAPI CLI: config override, PATH, then the submodule."""
    candidate = ctx.chem_inventory or "chemInventoryAPI"
    found = shutil.which(candidate) if os.path.sep not in candidate else (
        candidate if Path(candidate).exists() else None)
    if found:
        return found
    # chemInventoryAPI lives in a submodule: ~/bin/chemInventoryAPI/chemInventoryAPI
    fallback = Path.home() / "bin" / "chemInventoryAPI"
    if fallback.is_file():
        return str(fallback)
    entry = fallback / "chemInventoryAPI"
    return str(entry) if entry.is_file() else None


def chem_inventory_available() -> bool:
    """True when the chemInventoryAPI CLI can be found (the `chem` gate)."""
    return find_chem_inventory(ToolContext()) is not None


def speech_hosts(report: str) -> str:
    """`synthesis host → playback host` note from a `say --json` report.

    Disk hits carry the machine that originally rendered the audio, marked
    `(cached)`. A missing, older or unreadable report yields no note --
    never an error -- so the tool keeps working with a tts-remote that
    does not know `--json` yet.
    """
    try:
        payload = json.loads(report or "")
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    host = str(payload.get("host") or "").strip()
    play = str(payload.get("play_host") or "").strip()
    if not host or not play:
        return ""
    if str(payload.get("source") or "") in {"cache", "saved"}:
        host = f"{host} (cached)"
    return f"{host} → {play}"


def run_tts_say(binary: str, text: str, *, voice: Optional[str] = None,
                lang: Optional[str] = None, markdown: bool = False,
                timeout: float = 120.0, cancel: Optional[Any] = None
                ) -> subprocess.CompletedProcess:
    """Queue speech via `tts-remote say`; an older CLI is retried without --json.

    The `--json` report behind the speak hosts note is new, so a tts-remote
    that does not know the flag yet still gets to speak -- it just comes back
    without the report. Raises subprocess.TimeoutExpired when the CLI never
    returns; the caller turns that into its own message. `cancel` makes the
    synthesis killable by a stopped turn.
    """
    command = [binary, "say", "-q", "--json"]
    if markdown:
        command.append("--markdown")
    if voice:
        command += ["--voice", str(voice)]
    if lang:
        command += ["--lang", str(lang)]
    proc = run_cancellable(command, input_text=text, timeout=timeout,
                           cancel=cancel)
    if proc.returncode != 0 and "--json" in (proc.stderr or ""):
        proc = run_cancellable([part for part in command if part != "--json"],
                               input_text=text, timeout=timeout, cancel=cancel)
    return proc


def _handler_speak(args: dict[str, Any], ctx: ToolContext) -> str:
    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ToolError("missing 'text'")
    binary = find_tts_remote(ctx)
    if not binary:
        raise ToolError("tts-remote not found on PATH")
    voice = args.get("voice")
    lang = args.get("lang")
    try:
        proc = run_tts_say(binary, text, voice=voice, lang=lang, timeout=ctx.timeout,
                           cancel=getattr(ctx, "cancel", None))
    except subprocess.TimeoutExpired:
        raise ToolError("tts-remote timed out") from None
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise ToolError(f"tts-remote failed: {detail or 'unknown error'}")
    ctx.title_note = speech_hosts(proc.stdout)
    note = []
    if voice:
        note.append(f"voice={voice}")
    if lang:
        note.append(f"lang={lang}")
    suffix = f" ({', '.join(note)})" if note else ""
    return f"queued {len(text)} chars for speech{suffix}"


def _handler_list_voices(args: dict[str, Any], ctx: ToolContext) -> str:
    binary = find_tts_remote(ctx)
    if not binary:
        raise ToolError("tts-remote not found on PATH")
    try:
        proc = run_cancellable([binary, "voices", "--json"], timeout=ctx.timeout,
                               cancel=getattr(ctx, "cancel", None))
    except subprocess.TimeoutExpired:
        raise ToolError("tts-remote voices timed out") from None
    if proc.returncode != 0:
        raise ToolError(f"tts-remote voices failed: "
                        f"{(proc.stderr or proc.stdout).strip() or 'unknown error'}")
    try:
        payload = json.loads(proc.stdout)
    except ValueError as exc:
        raise ToolError(f"bad voices JSON: {exc}") from exc
    lines = []
    default = payload.get("default_voice")
    if default:
        lines.append(f"default voice: {default}")
    for entry in payload.get("voices") or []:
        bits = [str(entry.get("slug", "?"))]
        for key in ("style", "lang", "name"):
            if entry.get(key):
                bits.append(str(entry[key]))
        suffix = " [off]" if entry.get("enabled") is False else ""
        lines.append(" | ".join(bits) + suffix)
    if payload.get("styles"):
        lines.append("built-in styles: " + ", ".join(payload["styles"]))
    return "\n".join(lines) or "no voices reported"


# ── definitions (defs/tools/*.json + config overrides) ──────────────────────


@dataclass
class ToolDefinition:
    """One tool's declaration, loaded from its JSON file."""
    name: str
    description: str
    parameters: dict[str, Any]
    handler: str
    read_only: bool = False
    always_safe: bool = False
    #: capability the tool needs to appear: None, "search" or "tts"
    requires: Optional[str] = None
    #: names models tend to invent for this tool
    aliases: list[str] = field(default_factory=list)


#: `handler` name in a tool JSON -> the function that executes it.
#: Handlers return plain text, or a `ToolHandlerResult` when they produced
#: files (`execute_tool()` registers them through `ctx.artifacts`).
HANDLERS: dict[str, Callable[[dict[str, Any], ToolContext],
                             Union[str, ToolHandlerResult]]] = {
    "shell": _handler_shell,
    "python": _handler_python,
    "read_file": _handler_read_file,
    "read_pdf": _handler_read_pdf,
    "write_file": _handler_write_file,
    "edit_file": _handler_edit_file,
    "make_dir": _handler_make_dir,
    "move_file": _handler_move_file,
    "copy_file": _handler_copy_file,
    "delete_file": _handler_delete_file,
    "list_dir": _handler_list_dir,
    "publish_artifact": _handler_publish_artifact,
    "search_sessions": _handler_search_sessions,
    "list_models": _handler_list_models,
    "system_status": _handler_system_status,
    "context_status": _handler_context_status,
    "session_info": _handler_session_info,
    "chem_search": _handler_chem_search,
    "chem_container": _handler_chem_container,
    "chem_locations": _handler_chem_locations,
    "web_search": _handler_web_search,
    "fetch_url": _handler_fetch_url,
    "speak": _handler_speak,
    "list_voices": _handler_list_voices,
}

REQUIREMENTS = {"search", "tts", "chem", "artifacts"}

#: `requires` value -> what `/tools NAME help` says a missing tool needs.
REQUIREMENT_HINTS: dict[str, str] = {
    "search": "the ddgs package (pip install ddgs)",
    "tts": "the tts-remote command",
    "chem": "the chemInventoryAPI command (config: chem_inventory)",
    "artifacts": "the serve artifact store (`ollamarun serve`)",
}

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_DEFS_CACHE: Optional[list[ToolDefinition]] = None
_ALIASES_CACHE: Optional[dict[str, str]] = None


def tool_definition_dirs() -> list[Path]:
    """The definition layers, lowest priority first (config overrides ship on top)."""
    return [DEFS_DIR / "tools", DEFS_OVERRIDE_DIR / "tools"]


def _parse_definition(path: Path) -> ToolDefinition:
    """One JSON file -> ToolDefinition; raises ValueError with the reason."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"bad JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    name = str(data.get("name") or "").strip()
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"bad tool name {name!r}")
    description = str(data.get("description") or "").strip()
    if not description:
        raise ValueError("missing 'description'")
    parameters = data.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise ValueError("'parameters' must be a JSON-schema object")
    handler = str(data.get("handler") or "").strip()
    if handler not in HANDLERS:
        raise ValueError(f"unknown handler {handler!r}")
    requires = data.get("requires")
    if requires is not None and requires not in REQUIREMENTS:
        raise ValueError(f"unknown requirement {requires!r} (search/tts/chem/artifacts)")
    aliases = data.get("aliases") or []
    if not isinstance(aliases, list) or not all(
            isinstance(alias, str) for alias in aliases):
        raise ValueError("'aliases' must be a list of strings")
    return ToolDefinition(
        name=name, description=description, parameters=parameters,
        handler=handler,
        read_only=bool(data.get("read_only", False)),
        always_safe=bool(data.get("always_safe", False)),
        requires=requires,
        aliases=[alias.strip().lower() for alias in aliases if alias.strip()])


def load_tool_definitions(
        directories: Optional[list[Path]] = None) -> list[ToolDefinition]:
    """Every tool definition; a same-named file in a later layer wins.

    Bad files and duplicate tool names are skipped with a warning: a broken
    edit must never take the whole tool harness down.
    """
    if directories is not None:
        folders = [Path(directory) for directory in directories]
    else:
        folders = tool_definition_dirs()
    files: dict[str, Path] = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            files[path.stem] = path  # the later folder's file replaces it
    definitions: list[ToolDefinition] = []
    seen: dict[str, Path] = {}
    for stem in sorted(files):
        path = files[stem]
        try:
            definition = _parse_definition(path)
        except ValueError as exc:
            console.print(f"[warn]ignoring tool definition {path}: {exc}[/warn]")
            continue
        if definition.name in seen:
            console.print(f"[warn]ignoring tool definition {path}: duplicate name "
                          f"{definition.name!r} (already defined by "
                          f"{seen[definition.name]})[/warn]")
            continue
        seen[definition.name] = path
        definitions.append(definition)
    return definitions


def tool_definitions() -> list[ToolDefinition]:
    """The cached default-path definitions (a restart picks up file edits)."""
    global _DEFS_CACHE
    if _DEFS_CACHE is None:
        _DEFS_CACHE = load_tool_definitions()
    return _DEFS_CACHE


def tool_aliases() -> dict[str, str]:
    """alias -> tool name, built from each definition's `aliases` list."""
    global _ALIASES_CACHE
    if _ALIASES_CACHE is None:
        aliases: dict[str, str] = {}
        for definition in tool_definitions():
            for alias in definition.aliases:
                aliases.setdefault(alias, definition.name)
        _ALIASES_CACHE = aliases
    return _ALIASES_CACHE


def default_registry(*, tts: Optional[bool] = None,
                     search: Optional[bool] = None,
                     chem: Optional[bool] = None,
                     artifacts: Optional[bool] = None) -> dict[str, ToolSpec]:
    """The tool set, built from the definition files (defs/tools/*.json).

    `tts` forces the speech tools on/off; None auto-detects a `tts-remote`
    binary (the tools are simply absent when there is none). `search` does
    the same for `web_search` against the `ddgs` package, and `chem` for
    the chemInventoryAPI tools against the CLI. `artifacts` says whether
    the host runs an artifact store (the serve API does; the CLI does not),
    gating `publish_artifact`.
    """
    if search is None:
        search = search_available()
    if tts is None:
        fallback = Path.home() / "bin" / "tts-remote"
        tts = bool(shutil.which("tts-remote")) \
            or fallback.is_file() or (fallback / "tts-remote").is_file()
    if chem is None:
        chem = chem_inventory_available()
    available = {"search": bool(search), "tts": bool(tts), "chem": bool(chem),
                 "artifacts": bool(artifacts)}
    registry: dict[str, ToolSpec] = {}
    for definition in tool_definitions():
        if definition.requires and not available.get(definition.requires):
            continue
        registry[definition.name] = ToolSpec(
            definition.name, definition.description, definition.parameters,
            HANDLERS[definition.handler],
            read_only=definition.read_only, always_safe=definition.always_safe,
            requires=definition.requires, aliases=list(definition.aliases))
    return registry


def tool_schemas(registry: dict[str, ToolSpec],
                 disabled: Optional[set[str]] = None) -> list[dict[str, Any]]:
    disabled = disabled or set()
    schemas = []
    for name, spec in registry.items():
        if name in disabled:
            continue
        schemas.append({"type": "function", "function": {
            "name": spec.name, "description": spec.description,
            "parameters": spec.parameters}})
    return schemas


def _publish_artifacts(paths: list[str],
                       ctx: ToolContext) -> tuple[list[dict[str, Any]], list[str]]:
    """Register a handler's produced files through the turn's publisher.

    Regular files only (directories are skipped silently); a registration
    failure becomes a note in the tool text, never a failed tool -- the
    write itself succeeded and the refs that registered still attach.
    """
    publisher = getattr(ctx, "artifacts", None)
    refs: list[dict[str, Any]] = []
    notes: list[str] = []
    if publisher is None:
        return refs, notes
    for path in paths:
        try:
            if not path or not os.path.isfile(path):
                continue
            ref = publisher.publish_path(path)
        except Exception as exc:  # ApiError/OSError: a note, not a failure
            notes.append(f"[artifact not published: {path} ({exc})]")
            continue
        refs.append(ref.to_dict())
    return refs, notes


def execute_tool(spec: ToolSpec, args: dict[str, Any],
                 ctx: ToolContext) -> ToolOutcome:
    """Run one handler, converting everything into a model-readable result."""
    started = time.monotonic()
    outcome = ToolOutcome()
    ctx.title_note = ""
    cancel = getattr(ctx, "cancel", None)
    if cancel is not None and cancel.cancelled:
        outcome.text = f"{spec.name} not run: the turn was stopped"
        outcome.error = "cancelled"
        outcome.cancelled = True
        return outcome
    try:
        result = spec.handler(args, ctx)
        if isinstance(result, ToolHandlerResult):
            outcome.text = result.text
            outcome.artifacts, notes = _publish_artifacts(result.artifact_paths, ctx)
            if notes:
                outcome.text = f"{outcome.text}\n" + "\n".join(notes)
        else:
            outcome.text = str(result)
        outcome.ok = True
    except ToolCancelled as exc:
        outcome.text = str(exc)
        outcome.error = "cancelled"
        outcome.cancelled = True
    except ToolError as exc:
        outcome.text = str(exc)
        outcome.error = str(exc)
    except subprocess.TimeoutExpired as exc:
        outcome.text = f"{spec.name} timed out ({exc})"
        outcome.error = outcome.text
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # a broken tool must not kill the session
        outcome.text = f"{spec.name} failed: {exc}"
        outcome.error = outcome.text
    outcome.duration = time.monotonic() - started
    outcome.text = truncate_output(outcome.text, ctx.max_output)
    outcome.title_note = ctx.title_note
    return outcome


#: Tools whose handlers block on network I/O. The runner moves them to a
#: daemon worker so a stopped turn abandons the wait instead of hanging on it;
#: the result is simply discarded.
ABANDON_ON_CANCEL = frozenset({"fetch_url", "web_search"})


def _cancelled_outcome(name: str) -> ToolOutcome:
    return ToolOutcome(text=f"{name} cancelled: the turn was stopped; result discarded",
                       error="cancelled", cancelled=True)


class ToolRunner:
    """Runs one tool call, honouring the turn's cancel token.

    Subprocess tools kill their process group (through `run_cancellable()`);
    network tools run in a daemon worker and are *abandoned* on cancel -- the
    work finishes on its own and its result is dropped, nothing waits for it.
    """

    def __init__(self, poll: float = 0.05) -> None:
        self.poll = poll

    def run(self, spec: ToolSpec, args: dict[str, Any],
            ctx: ToolContext) -> ToolOutcome:
        cancel = getattr(ctx, "cancel", None)
        if cancel is not None and cancel.cancelled:
            return _cancelled_outcome(spec.name)
        if cancel is not None and spec.name in ABANDON_ON_CANCEL:
            return self._run_abandonable(spec, args, ctx)
        return execute_tool(spec, args, ctx)

    def _run_abandonable(self, spec: ToolSpec, args: dict[str, Any],
                         ctx: ToolContext) -> ToolOutcome:
        box: "queue.Queue[ToolOutcome]" = queue.Queue(maxsize=1)

        def work() -> None:
            try:
                box.put(execute_tool(spec, args, ctx))
            except BaseException:  # pragma: no cover - execute_tool never raises
                pass

        worker = threading.Thread(target=work, name=f"tool-{spec.name}", daemon=True)
        worker.start()
        while True:
            try:
                return box.get(timeout=self.poll)
            except queue.Empty:
                if ctx.cancel is not None and ctx.cancel.cancelled:
                    return _cancelled_outcome(spec.name)


#: Public surface re-exported by `ollamarun_lib/__init__.py`.  Built from the
#: namespace so a name that moves into a submodule keeps exporting exactly as
#: before; the submodules themselves are not part of the surface.
__all__ = [name for name, value in globals().items()
           if not name.startswith("_") and not isinstance(value, types.ModuleType)]
