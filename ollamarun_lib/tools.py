"""Tool registry, safety policy and executor for the agent loop.

`ToolSpec` describes one tool (name, description, JSON-schema parameters and
a handler); `default_registry()` builds the built-in set (shell, python,
files, fetch, and -- when `tts-remote` is available -- speech).
`approval_decision()` classifies a call as auto-approved or needing user
confirmation (read-only tools and provably-harmless shell/python calls are
safe; anything else asks), and `execute_tool()` runs the handler with a
timeout and caps the text that goes back to the model.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from .helpers import short_stamp

FETCH_TIMEOUT = 30.0
MAX_LIST_ENTRIES = 200
MAX_READ_CHARS = 200_000

#: Commands `run_shell` may execute unattended (basename of argv[0]).
DEFAULT_SHELL_ALLOW = {
    "ls", "pwd", "cat", "head", "tail", "grep", "rg", "find", "wc", "file",
    "stat", "tree", "du", "df", "sort", "cut", "tr", "diff", "echo", "date",
    "which", "whoami", "uname", "git",
}

#: `git` subcommands that only read the repository.
GIT_READONLY = {
    "status", "log", "diff", "show", "branch", "remote", "rev-parse",
    "ls-files", "ls-tree", "blame", "describe", "tag", "shortlog", "whatchanged",
    "rev-list", "cat-file", "grep", "config",  # config only via --get/--list checks
}

#: `git config` needs its reading forms to stay read-only.
GIT_CONFIG_SAFE = {"--get", "--get-all", "--get-regexp", "--list", "-l", "--get-urlmatch"}

#: Modules `run_python` may import unattended.
SAFE_PYTHON_MODULES = {
    "math", "cmath", "statistics", "json", "re", "datetime", "itertools",
    "functools", "collections", "random", "string", "decimal", "fractions",
    "textwrap", "unicodedata", "bisect", "heapq", "operator", "copy",
    "pprint", "uuid", "hashlib", "base64", "binascii", "csv", "time",
    "calendar", "zoneinfo", "dataclasses", "enum", "typing", "abc",
    "numbers", "array", "struct", "codecs",
}

#: Names/calls/attributes that make a python snippet unsafe even without imports.
PYTHON_DANGEROUS_NAMES = {
    "os", "sys", "subprocess", "shutil", "pathlib", "socket", "requests",
    "urllib", "ctypes", "importlib", "builtins", "glob", "tempfile", "signal",
    "threading", "multiprocessing", "asyncio", "pickle", "marshal", "http",
    "ftplib", "smtplib", "telnetlib", "pty", "fcntl", "mmap", "resource",
}
PYTHON_DANGEROUS_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "exit", "quit", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "memoryview", "help",
}
PYTHON_DANGEROUS_ATTRS = {
    "system", "popen", "run", "call", "check_call", "check_output", "Popen",
    "remove", "unlink", "rmtree", "mkdir", "makedirs", "rmdir", "rename",
    "touch", "chmod", "chown", "symlink", "link", "kill", "fork",
    "execv", "execve", "spawn", "spawnl", "spawnv", "connect", "urlopen",
    "write_text", "write_bytes", "send", "sendall", "pipe", "stat",
}

TRUNCATE_TAIL = 160


class ToolError(Exception):
    """A tool could not run; the message goes back to the model as-is."""


@dataclass
class ToolContext:
    """Runtime settings shared by every handler of one session."""
    cwd: str = "."
    timeout: float = 120.0
    max_output: int = 4000
    tts_remote: Optional[str] = None


@dataclass
class ToolOutcome:
    text: str = ""
    ok: bool = False
    duration: float = 0.0
    error: Optional[str] = None


@dataclass
class Decision:
    allowed: bool
    reason: str = ""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], str]
    read_only: bool = False
    always_safe: bool = False


# ── output plumbing ─────────────────────────────────────────────────────────


def truncate_output(text: str, limit: int) -> str:
    if not limit or len(text) <= limit:
        return text
    head = max(0, limit - TRUNCATE_TAIL - 60)
    omitted = len(text) - head - TRUNCATE_TAIL
    return (text[:head]
            + f"\n…[truncated: {omitted} of {len(text)} chars omitted]…\n"
            + text[-TRUNCATE_TAIL:])


def _clamp_timeout(value: Any, default: float) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return default
    if seconds <= 0:
        return default
    return min(seconds, default)


# ── safety analysis ─────────────────────────────────────────────────────────


def shell_safety(command: str, shell_allow: Optional[set[str]] = None) -> str:
    """Return '' when every segment of `command` is provably read-only.

    Anything else returns the human-readable reason it needs confirmation.
    Best-effort by design: a reason only ever means "ask", never "run".
    """
    allow = shell_allow if shell_allow is not None else DEFAULT_SHELL_ALLOW
    if not command or not command.strip():
        return "empty command"
    if "$" in command:
        return "command contains a variable or substitution"
    if "`" in command:
        return "command contains a backtick substitution"
    if ">" in command or "<" in command:
        return "command contains a redirection"
    if "(" in command or ")" in command or "{" in command or "}" in command:
        return "command contains grouping or brace expansion"
    for segment in re.split(r"(?:\|\||&&|[;|&\n])", command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            parts = shlex.split(segment)
        except ValueError as exc:
            return f"unparsable command ({exc})"
        if not parts:
            continue
        first = os.path.basename(parts[0])
        if "=" in parts[0]:
            return f"environment assignment {parts[0]!r}"
        if first not in allow:
            return f"{first!r} is not on the safe-command list"
        problem = _dangerous_shell_args(first, parts[1:])
        if problem:
            return problem
    return ""


def _dangerous_shell_args(command: str, args: list[str]) -> str:
    if command == "rg" and any(arg.startswith("--pre") for arg in args):
        return "rg --pre runs an arbitrary command"
    if command == "find":
        for flag in ("-exec", "-execdir", "-ok", "-okdir", "-delete"):
            if flag in args:
                return f"find {flag} can modify or run files"
        if any(arg.startswith(("-fprint", "-fprintf", "-fls")) for arg in args):
            return "find can write files"
    if command == "git":
        if "-c" in args or "-C" in args:
            return "git -c/-C changes how git runs"
        if any(arg.startswith("--exec-path") for arg in args):
            return "git --exec-path runs an arbitrary program"
        if any(arg == "--output" or arg.startswith("--output=") for arg in args):
            return "git --output writes a file"
        subcommand = next((arg for arg in args if not arg.startswith("-")), "")
        if subcommand not in GIT_READONLY:
            return f"git {subcommand or '(no subcommand)'} is not read-only"
        if subcommand == "config":
            if not any(arg in GIT_CONFIG_SAFE for arg in args):
                return "git config without --get/--list writes config"
            if any(arg.startswith("--file=") or arg == "--file" for arg in args):
                return "git config --file may write another repo's config"
        if subcommand in {"branch", "tag", "remote"}:
            writes = {"-d", "-D", "--delete", "-m", "-M", "-c", "-C", "--set-upstream"}
            if subcommand != "remote" and any(arg in writes for arg in args):
                return f"git {subcommand} can modify refs"
            if subcommand == "remote":
                subs = ("add", "remove", "rm", "rename", "set-url", "prune", "update")
                remaining = [a for a in args if a != subcommand and not a.startswith("-")]
                if remaining and remaining[0] in subs:
                    return "git remote can change remotes"
    if command == "sort" and any(arg.startswith("-o") or arg.startswith("--output")
                                for arg in args):
        return "sort -o writes a file"
    if command == "tree" and any(arg.startswith("-o") for arg in args):
        return "tree -o writes a file"
    if command == "date" and any(arg in {"-s", "--set"} or arg.startswith("--set=")
                                for arg in args):
        return "date -s sets the system clock"
    return ""


def python_safety(code: str) -> str:
    """Return '' when the snippet cannot touch the filesystem or the OS."""
    if not code or not code.strip():
        return "empty code"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"python does not parse ({exc.msg})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in SAFE_PYTHON_MODULES:
                    return f"imports {alias.name!r}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in SAFE_PYTHON_MODULES:
                return f"imports from {node.module or '.'!r}"
            if any(alias.name == "*" for alias in node.names):
                return "wildcard import"
        elif isinstance(node, ast.Call):
            called = _call_name(node.func)
            if called in PYTHON_DANGEROUS_CALLS:
                return f"calls {called}()"
        elif isinstance(node, ast.Name):
            if node.id in PYTHON_DANGEROUS_NAMES or node.id in PYTHON_DANGEROUS_CALLS:
                return f"uses {node.id}"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return f"uses dunder attribute .{node.attr}"
            if node.attr in PYTHON_DANGEROUS_ATTRS:
                return f"uses .{node.attr}"
    return ""


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def approval_decision(spec: ToolSpec, args: dict[str, Any], *,
                      policy: str = "allowlist",
                      shell_allow: Optional[set[str]] = None) -> Decision:
    """Classify a call as auto-approved or needing confirmation."""
    if spec.always_safe or spec.read_only:
        return Decision(True)
    if policy == "auto":
        return Decision(True, "auto policy")
    if policy == "ask":
        return Decision(False, "policy is ask")
    if spec.name == "run_shell":
        problem = shell_safety(str(args.get("command") or ""), shell_allow)
        return Decision(not problem, problem)
    if spec.name == "run_python":
        problem = python_safety(str(args.get("code") or ""))
        return Decision(not problem, problem)
    return Decision(False, f"{spec.name} modifies state")


def call_fingerprint(name: str, args: dict[str, Any]) -> str:
    basis = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    return f"{name}:{basis}"


def call_detail(name: str, args: dict[str, Any]) -> str:
    """One-line human summary of a tool call for the result panel."""
    if name in {"run_shell", "run_python"}:
        text = str(args.get("command") if name == "run_shell" else args.get("code") or "")
    elif name == "write_file":
        text = f"{args.get('path')} ({len(str(args.get('content') or ''))} chars)"
    elif name in {"read_file", "list_dir"}:
        text = str(args.get("path") or ".")
    elif name == "fetch_url":
        text = str(args.get("url") or "")
    elif name == "speak":
        text = str(args.get("text") or "")
    else:
        text = json.dumps(args, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= 160 else text[:159] + "…"


def normalize_arguments(raw: Any) -> Optional[dict[str, Any]]:
    """Tool-call arguments may arrive as a dict or a JSON string."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except ValueError:
            return None
        return value if isinstance(value, dict) else None
    return None


# ── handlers ────────────────────────────────────────────────────────────────


def _handler_shell(args: dict[str, Any], ctx: ToolContext) -> str:
    command = str(args.get("command") or "").strip()
    if not command:
        raise ToolError("missing 'command'")
    cwd = Path(str(args.get("cwd") or ctx.cwd)).expanduser()
    if not cwd.is_dir():
        raise ToolError(f"working directory does not exist: {cwd}")
    timeout = _clamp_timeout(args.get("timeout"), ctx.timeout)
    environment = {**os.environ, "BASH_ENV": os.devnull}
    try:
        proc = subprocess.run(["bash", "-c", command], cwd=str(cwd),
                              capture_output=True, text=True, timeout=timeout,
                              errors="replace", env=environment)
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
    try:
        proc = subprocess.run([sys.executable, "-I", "-"], input=code,
                              capture_output=True, text=True, timeout=timeout,
                              errors="replace", cwd=ctx.cwd)
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
    path = Path(str(args.get("path") or "")).expanduser()
    if not path.is_file():
        raise ToolError(f"not a file: {path}")
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
    path = Path(str(args.get("path") or "")).expanduser()
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
    return f"{verb} {len(content)} chars to {path}"


def _handler_list_dir(args: dict[str, Any], ctx: ToolContext) -> str:
    path = Path(str(args.get("path") or ctx.cwd)).expanduser()
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


def _handler_fetch_url(args: dict[str, Any], ctx: ToolContext) -> str:
    url = str(args.get("url") or "").strip()
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolError("url must be an absolute http:// or https:// URL")
    timeout = _clamp_timeout(args.get("timeout"), min(ctx.timeout, FETCH_TIMEOUT))
    try:
        response = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "ollamarun-tool/1.0"})
    except requests.RequestException as exc:
        raise ToolError(f"request failed: {exc}") from exc
    try:
        body = response.text
    finally:
        response.close()
    return f"HTTP {response.status_code}\n{body}"


# ── tts-remote (optional) ───────────────────────────────────────────────────


def find_tts_remote(ctx: ToolContext) -> Optional[str]:
    candidate = ctx.tts_remote or "tts-remote"
    found = shutil.which(candidate) if os.path.sep not in candidate else (
        candidate if Path(candidate).exists() else None)
    if found:
        return found
    fallback = Path.home() / "bin" / "tts-remote"
    return str(fallback) if fallback.exists() else None


def _handler_speak(args: dict[str, Any], ctx: ToolContext) -> str:
    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ToolError("missing 'text'")
    binary = find_tts_remote(ctx)
    if not binary:
        raise ToolError("tts-remote not found on PATH")
    command = [binary, "say", "-q"]
    voice = args.get("voice")
    lang = args.get("lang")
    if voice:
        command += ["--voice", str(voice)]
    if lang:
        command += ["--lang", str(lang)]
    try:
        proc = subprocess.run(command, input=text, capture_output=True,
                              text=True, timeout=ctx.timeout)
    except subprocess.TimeoutExpired:
        raise ToolError("tts-remote timed out") from None
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise ToolError(f"tts-remote failed: {detail or 'unknown error'}")
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
        proc = subprocess.run([binary, "voices", "--json"], capture_output=True,
                              text=True, timeout=ctx.timeout)
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


# ── registry ────────────────────────────────────────────────────────────────


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


def default_registry(*, tts: Optional[bool] = None) -> dict[str, ToolSpec]:
    """The built-in tool set, keyed by name.

    `tts` forces the speech tools on/off; None auto-detects a `tts-remote`
    binary (the tools are simply absent when there is none).
    """
    specs = [
        ToolSpec(
            "run_shell",
            "Run a shell command (bash -lc) and return its output and exit code. "
            "Use it for listings, search, git and similar read-only commands; "
            "anything that modifies the system asks the user first.",
            _object_schema({
                "command": {"type": "string", "description": "the command to run"},
                "cwd": {"type": "string", "description": "working directory (optional)"},
                "timeout": {"type": "number", "description": "seconds (optional)"},
            }, ["command"]),
            _handler_shell),
        ToolSpec(
            "run_python",
            "Execute a short Python 3 program and return its stdout/stderr. Good "
            "for calculations and text processing; snippets that touch files or "
            "the OS ask the user first.",
            _object_schema({
                "code": {"type": "string", "description": "the python source to run"},
                "timeout": {"type": "number", "description": "seconds (optional)"},
            }, ["code"]),
            _handler_python),
        ToolSpec(
            "read_file",
            "Read a text file (optionally a line range) from the local machine.",
            _object_schema({
                "path": {"type": "string", "description": "file path"},
                "offset": {"type": "integer", "description": "first line (0-based)"},
                "limit": {"type": "integer", "description": "how many lines"},
            }, ["path"]),
            _handler_read_file, read_only=True),
        ToolSpec(
            "write_file",
            "Write (or append) text to a file, creating it if needed. Asks the "
            "user for confirmation unless tools are running unlocked.",
            _object_schema({
                "path": {"type": "string", "description": "file path"},
                "content": {"type": "string", "description": "text to write"},
                "append": {"type": "boolean", "description": "append instead of replacing"},
            }, ["path", "content"]),
            _handler_write_file),
        ToolSpec(
            "list_dir",
            "List a directory with sizes and modification times.",
            _object_schema({
                "path": {"type": "string", "description": "directory (default: cwd)"},
            }, []),
            _handler_list_dir, read_only=True),
        ToolSpec(
            "fetch_url",
            "HTTP GET a URL and return the response body as text.",
            _object_schema({
                "url": {"type": "string", "description": "absolute http(s) URL"},
                "timeout": {"type": "number", "description": "seconds (optional)"},
            }, ["url"]),
            _handler_fetch_url, read_only=True),
    ]
    registry = {spec.name: spec for spec in specs}
    if tts is None:
        available = bool(shutil.which("tts-remote")) \
            or (Path.home() / "bin" / "tts-remote").exists()
    else:
        available = bool(tts)
    if available:
        registry["speak"] = ToolSpec(
            "speak",
            "Speak text out loud through tts-remote (queued in the background; "
            "returns immediately). Pass voice/lang to override the default voice.",
            _object_schema({
                "text": {"type": "string", "description": "what to say"},
                "voice": {"type": "string", "description": "voice slug (optional)"},
                "lang": {"type": "string", "description": "ISO language code (optional)"},
            }, ["text"]),
            _handler_speak, always_safe=True)
        registry["list_voices"] = ToolSpec(
            "list_voices",
            "List the voices available to the speak tool (slug, language, style).",
            _object_schema({}, []),
            _handler_list_voices, read_only=True)
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


def execute_tool(spec: ToolSpec, args: dict[str, Any],
                 ctx: ToolContext) -> ToolOutcome:
    """Run one handler, converting everything into a model-readable result."""
    started = time.monotonic()
    outcome = ToolOutcome()
    try:
        outcome.text = spec.handler(args, ctx)
        outcome.ok = True
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
    return outcome
