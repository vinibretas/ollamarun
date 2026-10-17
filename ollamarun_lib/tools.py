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

import ast
import fnmatch
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from dataclasses import dataclass, field
from html.parser import HTMLParser
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from .constants import DEFS_DIR, DEFS_OVERRIDE_DIR, console
from .client import OllamaError
from .helpers import (canonical_model_name, diff_preview, host_is_self,
                      human_bytes, is_local_host, local_free_disk,
                      local_gpu_probe, local_ram_probe, now_iso, short_stamp,
                      ssh_free_disk, ssh_gpu_probe, ssh_ram_probe)

FETCH_TIMEOUT = 30.0
MAX_LIST_ENTRIES = 200
MAX_READ_CHARS = 200_000

#: Browser-ish headers: the plain requests user-agent is a bot flag for many
#: sites and gets served challenges or error pages instead of content.
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/json,"
               "text/plain;q=0.9,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
}

SEARCH_TIMEOUT = 30.0
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 10
MAX_PAGE_LINKS = 25
MAX_PDF_CHARS = 40_000
MAX_PDF_PAGES = 50
MAX_ERROR_DETAIL = 300

#: Markers that identify a page which is only a JavaScript shell (or a bot
#: wall served with a 200) rather than content.
JS_SHELL_MARKERS = (
    "enablejs", "enable javascript", "javascript is disabled",
    "please turn javascript on", "requires javascript",
)
BOT_WALL_MARKERS = (
    "just a moment", "checking your browser", "cf-browser-verification",
    "challenge-platform", "cf-chl", "access denied", "attention required",
    "are you a robot", "unusual traffic", "captcha",
)
#: The subset that also shows up on challenge pages served with a 200.
SOFT_BLOCK_MARKERS = (
    "just a moment", "checking your browser", "cf-browser-verification",
    "challenge-platform", "cf-chl",
)

#: Hosts whose /search-style pages never return results to a plain HTTP
#: client; `fetch_url` reroutes them to `web_search` instead.
SEARCH_ENGINE_HOSTS = (
    "google.com", "bing.com", "duckduckgo.com", "search.yahoo.com",
    "search.brave.com", "ecosia.org", "mojeek.com", "startpage.com",
    "search.marginalia.nu", "yandex.com",
)

try:  # the search backend for the web_search tool
    from ddgs import DDGS  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    DDGS = None

#: Commands `run_shell` may execute unattended (basename of argv[0]).
DEFAULT_SHELL_ALLOW = {
    "ls", "pwd", "cat", "head", "tail", "grep", "rg", "find", "wc", "file",
    "stat", "tree", "du", "df", "sort", "cut", "tr", "diff", "echo", "date",
    "which", "whoami", "uname", "git",
}

#: How a command line is split into segments for per-segment safety checks.
_SHELL_SPLIT = re.compile(r"(?:\|\||&&|[;|&\n])")

#: Commands a stored approval rule may never cover (`dangerous_command`):
#: privilege escalation, remote shells, wrappers that smuggle another
#: command, interpreters (equivalent to arbitrary code execution), and
#: disk/power/network level tools.
RULE_FLOOR_COMMANDS = {
    "sudo", "doas", "su", "ssh", "scp", "sftp", "rsync", "env", "xargs",
    "nice", "nohup", "timeout", "watch", "stdbuf", "setsid", "eval", "exec",
    "sh", "bash", "zsh", "dash", "ksh", "fish", "python", "python3", "node",
    "perl", "ruby", "php", "lua",
    "dd", "fdisk", "sfdisk", "parted", "wipefs", "shred", "mkfs", "mount",
    "umount", "swapoff", "swapon", "shutdown", "reboot", "halt", "poweroff",
    "telinit", "nc", "ncat", "netcat", "iptables", "nft",
}

#: Interpreters that make `curl … | sh` style pipes never rule-coverable.
SHELL_INTERPRETERS = {"sh", "bash", "zsh", "dash", "ksh", "fish",
                      "python", "python3", "node", "perl", "ruby"}

#: Tools whose `path` argument can back a `path` approval rule.
PATH_RULE_TOOLS = {"read_file", "write_file", "list_dir", "edit_file",
                   "make_dir", "delete_file"}

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
    #: Live handles injected by the REPL: the Ollama client (`list_models`,
    #: `system_status`) and the session store (`search_sessions`). Typed Any,
    #: so this module keeps its one-way dependency graph; None outside a REPL.
    client: Any = None
    store: Any = None
    #: `--remote` details for probing the machine running Ollama: the ssh
    #: target/user, or None on the local machine.
    ssh_target: Optional[str] = None
    ssh_user: Optional[str] = None
    #: Path/command of the chemInventoryAPI CLI (config `chem_inventory`).
    chem_inventory: Optional[str] = None
    #: URL -> formatted result, so a repeated fetch is answered from the cache
    #: (the model is told instead of silently re-downloading the same page).
    fetched: dict[str, str] = field(default_factory=dict)
    #: lowercased query -> formatted web_search results, for the same reason.
    searches: dict[str, str] = field(default_factory=dict)
    #: Free-form note a handler wants in the result panel title (the speak
    #: tool sets `optiplex → g3` there); execute_tool() moves it onto the
    #: ToolOutcome, so callers never read this directly.
    title_note: str = ""


@dataclass
class ToolOutcome:
    text: str = ""
    ok: bool = False
    duration: float = 0.0
    error: Optional[str] = None
    #: Extra title text for the result panel (empty for most tools).
    title_note: str = ""


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
    for segment in _SHELL_SPLIT.split(command):
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
    findings = _python_findings(tree)
    return findings[0][0] if findings else ""


def _python_findings(tree: ast.AST, *,
                     strict: bool = False) -> list[tuple[str, str, str]]:
    """Every unsafe construct in `tree` as (message, kind, root), walk order.

    `python_safety` reports the first entry, so the messages and their order
    are exactly what the old first-problem-wins walk produced. `strict`
    additionally flags calls and bare names that shadow PYTHON_DANGEROUS_ATTRS
    (`system(...)` after `from os import system`), which stored `python`
    approval rules must never waive.
    """
    findings: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in SAFE_PYTHON_MODULES:
                    findings.append((f"imports {alias.name!r}", "import", root))
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in SAFE_PYTHON_MODULES:
                findings.append((f"imports from {node.module or '.'!r}",
                                 "importfrom", root or "."))
            if any(alias.name == "*" for alias in node.names):
                findings.append(("wildcard import", "wildcard", ""))
        elif isinstance(node, ast.Call):
            called = _call_name(node.func)
            if called in PYTHON_DANGEROUS_CALLS or (
                    strict and called in PYTHON_DANGEROUS_ATTRS):
                findings.append((f"calls {called}()", "call", called))
        elif isinstance(node, ast.Name):
            if node.id in PYTHON_DANGEROUS_NAMES or node.id in PYTHON_DANGEROUS_CALLS:
                findings.append((f"uses {node.id}", "name", node.id))
            elif strict and node.id in PYTHON_DANGEROUS_ATTRS:
                findings.append((f"uses {node.id}", "name", node.id))
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                findings.append((f"uses dunder attribute .{node.attr}",
                                 "attr", node.attr))
            elif node.attr in PYTHON_DANGEROUS_ATTRS:
                findings.append((f"uses .{node.attr}", "attr", node.attr))
    return findings


def python_import_roots(code: str) -> Optional[list[str]]:
    """Sorted module roots `code` imports, or None when ineligible.

    Ineligible means a syntax error, or a relative/wildcard import -- none
    of which a `python` approval rule may ever cover.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level or any(alias.name == "*" for alias in node.names):
                return None
            roots.add((node.module or "").split(".")[0])
    return sorted(roots)


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
    elif name == "edit_file":
        old = str(args.get("old_string") or "")
        new = str(args.get("new_string") or "")
        text = (f"{args.get('path')} ({len(old.splitlines())} → "
                f"{len(new.splitlines())} lines)")
    elif name in {"make_dir", "delete_file"}:
        text = str(args.get("path") or "")
    elif name in {"move_file", "copy_file"}:
        text = f"{args.get('source')} → {args.get('destination')}"
    elif name == "search_sessions":
        text = str(args.get("query") or "")
    elif name == "list_models":
        text = "loaded only" if args.get("loaded_only") else "installed"
    elif name == "system_status":
        text = str(args.get("target") or "ollama")
    elif name == "chem_search":
        text = str(args.get("query") or "")
    elif name == "chem_container":
        text = f"container {args.get('id')}"
    elif name == "chem_locations":
        text = str(args.get("location") or "all locations")
    elif name == "fetch_url":
        text = str(args.get("url") or "")
    elif name == "web_search":
        text = str(args.get("query") or "")
    elif name == "speak":
        text = str(args.get("text") or "")
    else:
        text = json.dumps(args, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= 160 else text[:159] + "…"


#: Tool names models tend to invent for the web tools are mapped onto the
#: real ones through each definition's `aliases` list (see the defs loader
#: below); `tool_aliases()` builds that map, `resolve_tool_alias()` uses it.
def resolve_tool_alias(name: str) -> str:
    """The real tool for a name a model may have invented (identity otherwise)."""
    key = str(name or "").strip().lower().replace("-", "_")
    return tool_aliases().get(key, name)


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


# ── approval rules ──────────────────────────────────────────────────────────

RULE_KINDS = {"call", "shell", "path", "python"}


@dataclass
class ToolRule:
    """A stored, generalized approval grant for one session.

    `call` holds an exact call fingerprint, `shell` a literal command prefix
    with an implicit `:*` tail (`mkdir -p :*`), `path` a glob relative to
    `base` (a realpath captured when the rule was made) and `python` a
    comma-separated set of module roots the snippet may import/use.
    """

    kind: str
    pattern: str
    base: str = ""
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "pattern": self.pattern,
                "base": self.base, "created": self.created}

    @classmethod
    def from_dict(cls, data: Any) -> Optional["ToolRule"]:
        if not isinstance(data, dict):
            return None
        kind = str(data.get("kind") or "")
        pattern = str(data.get("pattern") or "")
        if kind not in RULE_KINDS or not pattern:
            return None
        return cls(kind=kind, pattern=pattern, base=str(data.get("base") or ""),
                   created=str(data.get("created") or ""))


def _shorten_home(path: str) -> str:
    home = str(Path.home())
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def rule_label(rule: ToolRule) -> str:
    """One-line description of a rule, for prompts and `/tools rules`."""
    if rule.kind == "call":
        return "exactly this call"
    if rule.kind == "shell":
        return f"shell: {rule.pattern}"
    if rule.kind == "path":
        return f"path: {_shorten_home(rule.base)}/{rule.pattern}"
    if rule.kind == "python":
        return "python: imports " + ", ".join(p for p in rule.pattern.split(",") if p)
    return rule.kind


def dangerous_command(command: str) -> str:
    """Reason `command` may never be auto-approved by a rule ('' = coverable).

    Rules soften the "not on the safe list" class of asks only. Substitutions,
    redirections, privilege/wrapper/interpreter commands, disk-level tools,
    catastrophic `rm` and pipes into an interpreter always prompt again, even
    when they appear under a stored rule's prefix.
    """
    if not command or not command.strip():
        return "empty command"
    for char, why in (("$", "variable or substitution"),
                      ("`", "backtick substitution"),
                      (">", "output redirection"), ("<", "input redirection"),
                      ("(", "grouping"), (")", "grouping"),
                      ("{", "brace expansion"), ("}", "brace expansion")):
        if char in command:
            return why
    for match in re.finditer(r"(?<!\|)\|(?!\|)\s*([^\s|;&]+)", command):
        token = os.path.basename(match.group(1))
        if token in SHELL_INTERPRETERS:
            return f"pipes into {token!r}"
    for segment in _SHELL_SPLIT.split(command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            parts = shlex.split(segment)
        except ValueError:
            return "unparsable command"
        if not parts:
            continue
        first = os.path.basename(parts[0])
        if "=" in parts[0]:
            return f"environment assignment {parts[0]!r}"
        if first in RULE_FLOOR_COMMANDS:
            return f"{first!r} is never covered by a rule"
        if first == "rm" and _rm_targets_everything(parts[1:]):
            return "rm of a whole root"
    return ""


def _rm_targets_everything(args: list[str]) -> bool:
    recursive = any(arg == "--recursive" or (arg.startswith("-") and "r" in arg[1:])
                    for arg in args if arg.startswith("-"))
    if not recursive:
        return False
    for target in (arg for arg in args if not arg.startswith("-")):
        stripped = target.rstrip("/") or target
        if stripped in {"", "/", "/*", "~", "~/*", "*", ".", "..", "./*", "../*"}:
            return True
    return False


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _rule_args_block(command: str, args: list[str], prefix: list[str]) -> str:
    """Reason a stored rule may not cover these args ('' = ok).

    Like `_dangerous_shell_args`, except an ordinary mutating git subcommand
    is allowed when the rule prefix pins that subcommand token: `git add :*`
    covers `git add`, while a bare `git :*` still does not. `git -c/-C`,
    `git config` writes, ref deletion and the other structural hazards keep
    their veto under any prefix.
    """
    problem = _dangerous_shell_args(command, args)
    if not problem:
        return ""
    if command == "git" and problem.startswith("git ") \
            and problem.endswith("is not read-only"):
        subcommand = next((arg for arg in args if not arg.startswith("-")), "")
        if subcommand and len(prefix) >= 2 and prefix[1] == subcommand:
            return ""
    return problem


def _shell_rule_matches(rule: ToolRule, command: str,
                        shell_allow: Optional[set[str]]) -> bool:
    if dangerous_command(command):
        return False
    try:
        tokens = shlex.split(rule.pattern)
    except ValueError:
        return False
    if not tokens:
        return False
    wildcard = tokens[-1] == ":*"
    prefix = tokens[:-1] if wildcard else tokens
    if not prefix:
        return False
    head = os.path.basename(prefix[0])
    for segment in _SHELL_SPLIT.split(command):
        segment = segment.strip()
        if not segment:
            continue
        if shell_safety(segment, shell_allow) == "":
            continue  # already safe-listed: no rule needed for this segment
        try:
            parts = shlex.split(segment)
        except ValueError:
            return False
        if not parts or "=" in parts[0]:
            return False
        if os.path.basename(parts[0]) != head or parts[1:len(prefix)] != prefix[1:]:
            return False
        if not wildcard and len(parts) != len(prefix):
            return False
        if _rule_args_block(head, parts[1:], prefix):
            return False
    return True


def _path_rule_matches(rule: ToolRule, name: str, args: dict[str, Any],
                       cwd: str) -> bool:
    if name not in PATH_RULE_TOOLS or not rule.base or not rule.pattern:
        return False
    raw = str(args.get("path") or "").strip()
    if not raw:
        if name != "list_dir":
            return False
        target = Path(cwd)
    else:
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = Path(cwd) / target
    try:
        relative = Path(os.path.realpath(target)).relative_to(
            Path(os.path.realpath(rule.base)))
    except ValueError:
        return False
    rel = relative.as_posix()
    if fnmatch.fnmatchcase(rel, rule.pattern):
        return True
    if rule.pattern.endswith("/**") and rel == rule.pattern[:-3]:
        return True  # `dir/**` also covers listing the directory itself
    return False


def _python_rule_matches(rule: ToolRule, code: str) -> bool:
    allowed = {part for part in rule.pattern.split(",") if part}
    if not allowed:
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    imported = False
    for _message, kind, root in _python_findings(tree, strict=True):
        if kind in {"import", "importfrom"}:
            if root not in allowed:
                return False
            imported = True
        elif kind == "name":
            if root not in allowed:
                return False
        else:
            return False
    return imported


def rule_matches(rule: ToolRule, name: str, args: dict[str, Any], *,
                 cwd: str = ".", shell_allow: Optional[set[str]] = None) -> bool:
    """True when a stored rule covers this exact call."""
    if rule.kind == "shell":
        return name == "run_shell" and _shell_rule_matches(
            rule, str(args.get("command") or ""), shell_allow)
    if rule.kind == "path":
        return _path_rule_matches(rule, name, args, cwd)
    if rule.kind == "python":
        return name == "run_python" and _python_rule_matches(
            rule, str(args.get("code") or ""))
    if rule.kind == "call":
        return call_fingerprint(name, args) == rule.pattern
    return False


def matching_rule(rules: list[ToolRule], name: str, args: dict[str, Any], *,
                  cwd: str = ".",
                  shell_allow: Optional[set[str]] = None) -> Optional[ToolRule]:
    """First stored rule covering the call, or None."""
    for rule in rules:
        if rule_matches(rule, name, args, cwd=cwd, shell_allow=shell_allow):
            return rule
    return None


def _suggest_shell_rule(command: str, cwd: str) -> list[ToolRule]:
    if not command or dangerous_command(command):
        return []
    if any(separator in command for separator in ("|", "&&", ";", "&", "\n")):
        return []  # compound commands only ever get the exact-call grant
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    if not tokens or "=" in tokens[0]:
        return []
    head = os.path.basename(tokens[0])
    index: Optional[int] = None
    for position in range(1, len(tokens)):
        token = tokens[position]
        if token.startswith("-"):
            continue
        if "/" in token or token.startswith("~"):
            index = position
            break
        try:
            exists = (Path(cwd) / token).exists()
        except OSError:
            exists = False
        if exists:
            index = position
            break
    if index is not None:
        prefix = tokens[:index]
    elif len(tokens) >= 3:
        prefix = tokens[:-1]
    elif len(tokens) == 2:
        prefix = tokens[:1]
    else:
        return []
    if not prefix or prefix == tokens:
        return []
    if _rule_args_block(head, tokens[1:], prefix):
        return []
    return [ToolRule(kind="shell", pattern=" ".join(prefix) + " :*",
                     created=now_iso())]


def _suggest_path_rules(name: str, args: dict[str, Any], cwd: str) -> list[ToolRule]:
    raw = str(args.get("path") or "").strip()
    if not raw:
        if name != "list_dir":
            return []
        raw = cwd
    target = Path(raw).expanduser()
    if not target.is_absolute():
        target = Path(cwd) / target
    resolved = Path(os.path.realpath(target))
    if not resolved.name:
        return []
    base = str(resolved.parent)
    within = _within(resolved, Path(os.path.realpath(cwd)))
    if resolved.is_dir():
        pattern = f"{resolved.name}/**" if within else resolved.name
        return [ToolRule(kind="path", pattern=pattern, base=base,
                         created=now_iso())]
    rules = [ToolRule(kind="path", pattern=resolved.name, base=base,
                      created=now_iso())]
    if within:  # tree rules are only offered inside the working directory
        rules.append(ToolRule(kind="path", pattern="**", base=base,
                              created=now_iso()))
    return rules


def _suggest_python_rule(code: str) -> list[ToolRule]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    findings = _python_findings(tree, strict=True)
    if not findings:
        return []
    roots = {root for _message, kind, root in findings
             if kind in {"import", "importfrom"}}
    if not roots:
        return []
    for _message, kind, root in findings:
        if kind in {"import", "importfrom", "name"} and root in roots:
            continue
        return []
    return [ToolRule(kind="python", pattern=",".join(sorted(roots)),
                     created=now_iso())]


def suggest_rules(name: str, args: dict[str, Any], *, cwd: str = ".",
                  limit: int = 2) -> list[ToolRule]:
    """Generalized rules that would cover this call, most specific first."""
    if name == "run_shell":
        rules = _suggest_shell_rule(str(args.get("command") or ""), cwd)
    elif name in PATH_RULE_TOOLS:
        rules = _suggest_path_rules(name, args, cwd)
    elif name == "run_python":
        rules = _suggest_python_rule(str(args.get("code") or ""))
    else:
        rules = []
    return rules[:limit]


# ── web: page reading and search ────────────────────────────────────────────


class _HTMLTextExtractor(HTMLParser):
    """Collect readable text and links from one HTML document.

    Boilerplate (scripts, styles, navigation, forms) is dropped, block tags
    become line breaks and headings/list items keep a markdown-ish marker.
    Content inside `<article>`/`<main>` is tracked separately so a page's
    chrome can be left out of the result.
    """

    SKIP_TAGS = {
        "script", "style", "noscript", "template", "svg", "iframe", "canvas",
        "form", "button", "select", "option", "nav", "header", "footer",
        "aside", "dialog", "map", "object", "embed",
    }
    BLOCK_TAGS = {
        "address", "article", "blockquote", "dd", "div", "dl", "dt",
        "fieldset", "figcaption", "figure", "hr", "li", "main", "ol", "p",
        "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead",
        "tr", "ul",
    }
    HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
    MAIN_TAGS = {"article", "main"}

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_chunks: list[str] = []
        self.chunks: list[tuple[str, bool]] = []
        self.links: list[tuple[str, str, bool]] = []
        self._in_title = False
        self._in_head = False
        self._skip_depth = 0
        self._main_depth = 0
        self._open_link: Optional[tuple[list[str], str, bool]] = None

    def _emit(self, text: str) -> None:
        if not text:
            return
        self.chunks.append((text, self._main_depth > 0))
        if self._open_link is not None:
            self._open_link[0].append(text)

    def _break(self, count: int = 1) -> None:
        self.chunks.append(("\n" * count, self._main_depth > 0))

    @staticmethod
    def _usable_href(href: str) -> bool:
        href = href.strip()
        return bool(href) and not href.startswith(
            ("#", "javascript:", "mailto:", "tel:", "data:"))

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self.MAIN_TAGS:
            self._main_depth += 1
        if tag == "title":
            self._in_title = True
        elif tag == "head":
            self._in_head = True
        elif tag in self.HEADING_TAGS:
            self._break(2)
            self._emit("#" * self.HEADING_TAGS[tag] + " ")
        elif tag == "li":
            self._break()
            self._emit("- ")
        elif tag in self.BLOCK_TAGS or tag == "br":
            self._break()
        elif tag == "a":
            href = dict(attrs).get("href") or ""
            if self._usable_href(href):
                self._open_link = ([], href, self._main_depth > 0)

    def handle_startendtag(self, tag: str,
                           attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS or self._skip_depth:
            return
        if tag in self.BLOCK_TAGS or tag == "br":
            self._break()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag == "head":
            self._in_head = False
        elif tag == "a" and self._open_link is not None:
            text, href, in_main = self._open_link
            label = " ".join("".join(text).split())
            self.links.append((label, href, in_main))
            self._open_link = None
        elif tag in self.HEADING_TAGS or tag in self.BLOCK_TAGS or tag == "li":
            self._break()
        if tag in self.MAIN_TAGS:
            self._main_depth = max(0, self._main_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data:
            return
        if self._in_title:
            self.title_chunks.append(data)
        if self._in_head:
            return
        self._emit(data)


def _normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_html_text(html: str,
                      base_url: str = "") -> tuple[str, str, list[tuple[str, str]]]:
    """Return (title, readable text, links) for one HTML document.

    The `<article>`/`<main>` region wins when it holds most of the text;
    otherwise the whole body is used. Links come back as absolute, decoded,
    de-duplicated (label, url) pairs, in page order.
    """
    parser = _HTMLTextExtractor(base_url)
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup must never fail a fetch
        pass
    title = _normalize_text("".join(parser.title_chunks))
    full = _normalize_text("".join(chunk for chunk, _ in parser.chunks))
    main = _normalize_text("".join(chunk for chunk, in_main in parser.chunks
                                   if in_main))
    if main and len(main) >= max(200, int(0.2 * len(full))):
        text = main
        entries = [entry for entry in parser.links if entry[2]]
    else:
        text = full
        entries = parser.links
    seen: set[str] = set()
    links: list[tuple[str, str]] = []
    for label, href, _ in entries:
        url = urllib.parse.urljoin(base_url, href)
        if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
            continue
        if url in seen:
            continue
        seen.add(url)
        links.append((label, url))
        if len(links) >= MAX_PAGE_LINKS:
            break
    return title, text, links


def extract_pdf_text(data: bytes, max_chars: int = MAX_PDF_CHARS) -> Optional[str]:
    """Text of a PDF from its bytes, or None when nothing can extract it.

    An empty string means the PDF was read but carries no extractable text
    (typically a scan).
    """
    if not data:
        return None
    text = _pdf_text_reader(data, max_chars)
    if text is None:
        text = _pdf_text_tool(data, max_chars)
    return text


def _pdf_text_reader(data: bytes, max_chars: int) -> Optional[str]:
    """pypdf-based extraction (in-process); None when pypdf is missing/broken."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        reader = PdfReader(io.BytesIO(data))
        chunks: list[str] = []
        total = 0
        for page in list(reader.pages)[:MAX_PDF_PAGES]:
            page_text = page.extract_text() or ""
            chunks.append(page_text)
            total += len(page_text)
            if total >= max_chars:
                break
    except Exception:
        return None
    return _normalize_text("\n\n".join(chunks))[:max_chars]


def _pdf_text_tool(data: bytes, max_chars: int) -> Optional[str]:
    """pdftotext fallback; None when the binary is missing or fails."""
    binary = shutil.which("pdftotext")
    if not binary:
        return None
    try:
        proc = subprocess.run([binary, "-layout", "-", "-"], input=data,
                              capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout.decode("utf-8", errors="replace")
    return _normalize_text(text)[:max_chars]


def _http_phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return ""


def _status_url_line(url: str, final_url: str, status: int) -> str:
    phrase = _http_phrase(status)
    line = f"HTTP {status}" + (f" {phrase}" if phrase else "")
    line += f" — {final_url or url}"
    if final_url and final_url != url:
        line += f" (asked for {url})"
    return line


def _looks_like_js_shell(body: str, text: str) -> bool:
    lower = (body or "").lower()
    if any(marker in lower
           for marker in JS_SHELL_MARKERS + SOFT_BLOCK_MARKERS):
        return True
    # A large document that yields almost no text is a scripted page (or a
    # wall) no matter what it says.
    return len(text) < 120 and len(body) > 4000


def _failure_note(status: int, content_type: str, body: str) -> str:
    """What went wrong and what to do next -- never the site's error page."""
    lower = (body or "").lower()
    blocked = any(marker in lower for marker in BOT_WALL_MARKERS)
    if status in (404, 410):
        return ("the page does not exist at this URL. Do not guess more URLs on "
                "this site; call web_search to find the right page — and if the "
                "first search misses it, search again with different words.")
    if status in (401, 407):
        return ("the page needs a login. If it should be public, call web_search "
                "for another source; otherwise ask the user for access.")
    if status == 429:
        return ("the site is rate-limiting requests. Wait a moment, or call "
                "web_search for another source.")
    if blocked:
        return ("blocked by a bot-protection challenge (Cloudflare/Akamai-style), "
                "not the real page. Do not retry this URL or guess others here; "
                "call web_search for another source.")
    if status == 403:
        return ("the site refused the request, so nothing readable came back. "
                "Do not retry this URL; call web_search for another source.")
    if status >= 500:
        return ("the server had a temporary error. The URL may work later; "
                "meanwhile call web_search for a cached or alternative source.")
    if status >= 400:
        return "the server rejected the request. Check the URL or call web_search."
    return "the request did not return usable content."


def _error_detail(content_type: str, body: str) -> str:
    """Short excerpt of a failed response's own message (JSON/plain text)."""
    body = (body or "").strip()
    if not body or "html" in content_type:
        return ""
    text = " ".join(body.split())
    return text[:MAX_ERROR_DETAIL] + ("…" if len(text) > MAX_ERROR_DETAIL else "")


def _is_json(content_type: str) -> bool:
    return content_type == "application/json" or content_type.endswith("+json")


def _is_textual(content_type: str) -> bool:
    if content_type.startswith("text/"):
        return True
    return content_type.endswith("+xml") or content_type in {
        "application/json", "application/xml", "application/javascript",
        "application/x-javascript", "application/xhtml+xml",
        "application/rss+xml", "application/atom+xml",
    }


def _decode_response(response: Any) -> str:
    """Decoded body text, preferring UTF-8 when the site omits its charset.

    `requests` follows the RFC and assumes latin-1 for `text/html` without a
    charset, which mangles UTF-8 pages (docs sites, news, ...).
    """
    data = response.content or b""
    encoding = (getattr(response, "encoding", None) or "").lower()
    if encoding in {"", "iso-8859-1", "latin-1", "ascii"}:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            pass
    try:
        return response.text
    except (AttributeError, UnicodeDecodeError):
        return data.decode("utf-8", errors="replace")


def format_fetch_response(*, url: str, final_url: str, status: int,
                          content_type: str, body: str, size: int = 0) -> str:
    """The model-readable text for one HTTP response.

    HTML becomes extracted text plus the page's links, JSON is pretty-printed,
    error pages are replaced by an explanation of what happened and what to do
    next, and JavaScript-only shells say so instead of looking empty.
    """
    head = _status_url_line(url, final_url or url, status)
    if status >= 400:
        lines = [head, f"fetch failed: {_failure_note(status, content_type, body)}"]
        detail = _error_detail(content_type, body)
        if detail:
            lines.append(f"server said: {detail}")
        return "\n".join(lines)
    if not body.strip() and not size:
        return head + "\n(empty response body)"
    if "html" in content_type or (not content_type and "<html" in body[:2000].lower()):
        title, text, links = extract_html_text(body, base_url=final_url or url)
        if len(text) < 120 and _looks_like_js_shell(body, text):
            lines = [head]
            if title:
                lines.append(f"Title: {title}")
            lines.append(
                "no readable text: this page is rendered by JavaScript (only "
                "the app shell came back). Fetching it again will not help — "
                "call web_search and pick a result that serves static text, an "
                "API endpoint or a docs page; if nothing fits, search again "
                "with different words.")
            return "\n".join(lines)
        parts = [head]
        if title:
            parts.append(f"Title: {title}")
        parts.append("")
        parts.append(text or "(no readable text)")
        if links:
            parts.append("")
            parts.append("Links on this page:")
            parts.extend(f"- {label or link} — {link}" for label, link in links)
        return "\n".join(parts)
    if _is_json(content_type):
        try:
            pretty = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
        except ValueError:
            pretty = body
        return f"{head}\n{pretty}"
    if _is_textual(content_type):
        return f"{head}\n{body}"
    shown = size or len(body)
    return (f"{head}\n(binary content: {content_type or 'unknown type'}, "
            f"{shown} bytes — not shown as text)")


def search_engine_query(url: str) -> str:
    """The query inside a search-engine URL, or '' when it is not one."""
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return ""
    known = any(host == candidate or host.endswith("." + candidate)
                for candidate in SEARCH_ENGINE_HOSTS)
    googleish = host.startswith("google.") or ".google." in host
    if not (known or googleish):
        return ""
    params = urllib.parse.parse_qs(parsed.query)
    for key in ("q", "query", "p", "text", "search_query"):
        values = params.get(key)
        if values and values[0].strip():
            return values[0].strip()
    return ""


def search_available() -> bool:
    """True when the web_search backend (ddgs) can be imported."""
    return DDGS is not None


def format_search_results(query: str, results: list[tuple[str, str, str]]) -> str:
    if not results:
        return (f"web_search: no results for {query!r}. That is not proof the "
                f"information does not exist — search again with different or "
                f"narrower words (synonyms, exact phrases, site:, a year) instead "
                f"of deciding it is not available; only report it as not found "
                f"after a few rephrased searches come up empty (do not guess URLs).")
    lines = [f"web_search: {query!r} — {len(results)} result(s)"]
    for index, (title, url, snippet) in enumerate(results, 1):
        lines.append(f"{index}. {title or '(untitled)'}")
        lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {' '.join(snippet.split())[:300]}")
    lines.append("")
    lines.append("Next: call fetch_url on the most promising result(s) to read them. "
                 "If none of them answers the question, search again with different "
                 "or narrower words instead of settling.")
    return "\n".join(lines)


def run_web_search(query: str, max_results: int = DEFAULT_SEARCH_RESULTS,
                   timeout: float = SEARCH_TIMEOUT) -> Optional[str]:
    """Search through ddgs; None when the backend is not installed."""
    client_class = DDGS
    if client_class is None:
        return None
    try:
        with client_class(timeout=timeout) as client:
            raw = client.text(query, max_results=max_results)
    except TypeError:  # a backend whose constructor signature differs
        try:
            with client_class() as client:
                raw = client.text(query, max_results=max_results)
        except Exception as exc:
            raise ToolError(f"web search failed: {_short_error(exc)}") from exc
    except Exception as exc:
        raise ToolError(f"web search failed: {_short_error(exc)}") from exc
    results: list[tuple[str, str, str]] = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "")
        link = str(entry.get("href") or entry.get("url")
                   or entry.get("link") or "")
        snippet = str(entry.get("body") or entry.get("snippet")
                      or entry.get("description") or "")
        if title or link:
            results.append((title, link, snippet))
    return format_search_results(query, results)


def _short_error(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text[:200] or exc.__class__.__name__


def _request_error(exc: requests.RequestException) -> str:
    """A short, actionable reason for a failed request."""
    if isinstance(exc, requests.Timeout):
        return "the request timed out (the site may be slow or blocking this machine)"
    text = str(exc)
    if ("Name or service not known" in text or "Failed to resolve" in text
            or "No address associated with hostname" in text):
        return ("the host name could not be resolved — the domain is probably "
                "misspelled or wrong; call web_search to find the real URL")
    match = re.search(r"Caused by ([^\n]+)", text)
    if match:
        text = match.group(1)
    return _short_error(text)


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


def _handler_edit_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path = Path(str(args.get("path") or "")).expanduser()
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
    try:
        mode = path.stat().st_mode
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                        dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
        os.chmod(tmp_name, mode & 0o7777)
        os.replace(tmp_name, path)
    except OSError as exc:
        raise ToolError(f"could not write {path}: {exc}") from exc
    replaced = count if replace_all else 1
    if replaced == 1:
        first = text[:text.find(old)].count("\n") + 1
        where = f" at line {first}"
    else:
        where = ""
    diff = diff_preview(old, new, limit=30)
    result = f"edited {path}: replaced {replaced} occurrence(s){where}"
    return f"{result}\n{diff}" if diff else result


def _handler_make_dir(args: dict[str, Any], ctx: ToolContext) -> str:
    raw = str(args.get("path") or "").strip()
    if not raw:
        raise ToolError("missing 'path'")
    path = Path(raw).expanduser()
    if path.is_dir():
        return f"directory already exists: {path}"
    if path.exists():
        raise ToolError(f"a file exists at {path}")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolError(f"could not create {path}: {exc}") from exc
    return f"created directory {path}"


def _transfer_paths(args: dict[str, Any]) -> tuple[Path, Path, bool]:
    source = str(args.get("source") or "").strip()
    destination = str(args.get("destination") or "").strip()
    if not source:
        raise ToolError("missing 'source'")
    if not destination:
        raise ToolError("missing 'destination'")
    return Path(source).expanduser(), Path(destination).expanduser(), \
        bool(args.get("overwrite"))


def _handler_move_file(args: dict[str, Any], ctx: ToolContext) -> str:
    source, destination, overwrite = _transfer_paths(args)
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
    return f"moved {source} → {destination}"


def _handler_copy_file(args: dict[str, Any], ctx: ToolContext) -> str:
    source, destination, overwrite = _transfer_paths(args)
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
    return f"copied {source} → {destination}"


def _handler_delete_file(args: dict[str, Any], ctx: ToolContext) -> str:
    raw = str(args.get("path") or "").strip()
    if not raw:
        raise ToolError("missing 'path'")
    path = Path(raw).expanduser()
    if not path.exists() and not path.is_symlink():
        raise ToolError(f"no such file or directory: {path}")
    trash = shutil.which("trash-put")
    if not trash:
        raise ToolError("trash-put not found; refusing to delete without a trash")
    try:
        proc = subprocess.run([trash, str(path)], capture_output=True, text=True,
                              timeout=ctx.timeout)
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
        ram = ssh_ram_probe(ssh_target, user=ctx.ssh_user)
        gpu = ssh_gpu_probe(ssh_target, user=ctx.ssh_user)
        disk = ssh_free_disk(ssh_target, user=ctx.ssh_user)
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


# ── chem inventory (chemInventoryAPI) ───────────────────────────────────────


def run_chem_inventory(binary: str, argv: list[str], timeout: float) -> Any:
    """Run `chemInventoryAPI … --json` and parse the JSON it prints."""
    try:
        proc = subprocess.run([binary, *argv, "--json"], capture_output=True,
                              text=True, timeout=timeout, errors="replace")
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
    return run_chem_inventory(binary, argv, min(ctx.timeout, 90))


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


def _handler_fetch_url(args: dict[str, Any], ctx: ToolContext) -> str:
    url = str(args.get("url") or "").strip()
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolError("url must be an absolute http:// or https:// URL")

    query = search_engine_query(url)
    if query:
        timeout = _clamp_timeout(args.get("timeout"),
                                 min(ctx.timeout, SEARCH_TIMEOUT))
        results = run_web_search(query, DEFAULT_SEARCH_RESULTS, timeout)
        if results is None:
            raise ToolError(
                "this is a search-engine results URL, which only returns a "
                "JavaScript shell when fetched; the web_search tool searches "
                "from here instead, but its ddgs backend is not installed")
        return (f"(search-engine URL: ran web_search for {query!r} instead of "
                f"fetching the page)\n{results}")

    previous = ctx.fetched.get(url)
    if previous:
        preview = previous[:400] + ("…" if len(previous) > 400 else "")
        return (f"(already fetched {url} earlier in this session — not fetching "
                f"it again)\nstart of the earlier result:\n{preview}")

    timeout = _clamp_timeout(args.get("timeout"), min(ctx.timeout, FETCH_TIMEOUT))
    try:
        response = requests.get(url, timeout=timeout,
                                headers=dict(BROWSER_HEADERS),
                                allow_redirects=True)
    except requests.RequestException as exc:
        raise ToolError(f"could not fetch {url}: {_request_error(exc)}") from exc
    try:
        status = response.status_code
        content_type = (response.headers.get("Content-Type") or "")
        content_type = content_type.split(";")[0].strip().lower()
        size = len(response.content or b"")
        if status < 400 and ("pdf" in content_type
                             or parsed.path.lower().endswith(".pdf")):
            result = _pdf_result(url, response, size)
        else:
            body = _decode_response(response) if _is_textual(content_type) else ""
            result = format_fetch_response(url=url, final_url=response.url,
                                           status=status,
                                           content_type=content_type,
                                           body=body, size=size)
    finally:
        response.close()
    ctx.fetched[url] = result
    return result


def _pdf_result(url: str, response: Any, size: int) -> str:
    head = _status_url_line(url, response.url or url, response.status_code)
    text = extract_pdf_text(response.content or b"")
    if text is None:
        return (f"{head}\n(a PDF, {size} bytes; no PDF text extractor is "
                f"available on this machine — pypdf or pdftotext would read it)")
    if not text.strip():
        return f"{head}\n(a PDF, {size} bytes; no extractable text — likely a scan)"
    return f"{head}\n(PDF turned into text, {size} bytes)\n{text}"


def _handler_web_search(args: dict[str, Any], ctx: ToolContext) -> str:
    query = str(args.get("query") or args.get("q") or "").strip()
    if not query:
        raise ToolError("missing 'query'")
    cached = ctx.searches.get(query.lower())
    if cached:
        return f"(cached result from earlier in this session)\n{cached}"
    try:
        max_results = int(args.get("max_results") or DEFAULT_SEARCH_RESULTS)
    except (TypeError, ValueError):
        max_results = DEFAULT_SEARCH_RESULTS
    max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
    timeout = _clamp_timeout(args.get("timeout"), min(ctx.timeout, SEARCH_TIMEOUT))
    text = run_web_search(query, max_results, timeout)
    if text is None:
        raise ToolError("web search is unavailable (the ddgs package is not installed)")
    ctx.searches[query.lower()] = text
    return text


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
                timeout: float = 120.0) -> subprocess.CompletedProcess:
    """Queue speech via `tts-remote say`; an older CLI is retried without --json.

    The `--json` report behind the speak hosts note is new, so a tts-remote
    that does not know the flag yet still gets to speak -- it just comes back
    without the report. Raises subprocess.TimeoutExpired when the CLI never
    returns; the caller turns that into its own message.
    """
    command = [binary, "say", "-q", "--json"]
    if markdown:
        command.append("--markdown")
    if voice:
        command += ["--voice", str(voice)]
    if lang:
        command += ["--lang", str(lang)]
    proc = subprocess.run(command, input=text, capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0 and "--json" in (proc.stderr or ""):
        proc = subprocess.run([part for part in command if part != "--json"],
                              input=text, capture_output=True, text=True,
                              timeout=timeout)
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
        proc = run_tts_say(binary, text, voice=voice, lang=lang, timeout=ctx.timeout)
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
HANDLERS: dict[str, Callable[[dict[str, Any], ToolContext], str]] = {
    "shell": _handler_shell,
    "python": _handler_python,
    "read_file": _handler_read_file,
    "write_file": _handler_write_file,
    "edit_file": _handler_edit_file,
    "make_dir": _handler_make_dir,
    "move_file": _handler_move_file,
    "copy_file": _handler_copy_file,
    "delete_file": _handler_delete_file,
    "list_dir": _handler_list_dir,
    "search_sessions": _handler_search_sessions,
    "list_models": _handler_list_models,
    "system_status": _handler_system_status,
    "chem_search": _handler_chem_search,
    "chem_container": _handler_chem_container,
    "chem_locations": _handler_chem_locations,
    "web_search": _handler_web_search,
    "fetch_url": _handler_fetch_url,
    "speak": _handler_speak,
    "list_voices": _handler_list_voices,
}

REQUIREMENTS = {"search", "tts", "chem"}

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
        raise ValueError(f"unknown requirement {requires!r} (search/tts/chem)")
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
                     chem: Optional[bool] = None) -> dict[str, ToolSpec]:
    """The tool set, built from the definition files (defs/tools/*.json).

    `tts` forces the speech tools on/off; None auto-detects a `tts-remote`
    binary (the tools are simply absent when there is none). `search` does
    the same for `web_search` against the `ddgs` package, and `chem` for
    the chemInventoryAPI tools against the CLI.
    """
    if search is None:
        search = search_available()
    if tts is None:
        fallback = Path.home() / "bin" / "tts-remote"
        tts = bool(shutil.which("tts-remote")) \
            or fallback.is_file() or (fallback / "tts-remote").is_file()
    if chem is None:
        chem = chem_inventory_available()
    available = {"search": bool(search), "tts": bool(tts), "chem": bool(chem)}
    registry: dict[str, ToolSpec] = {}
    for definition in tool_definitions():
        if definition.requires and not available.get(definition.requires):
            continue
        registry[definition.name] = ToolSpec(
            definition.name, definition.description, definition.parameters,
            HANDLERS[definition.handler],
            read_only=definition.read_only, always_safe=definition.always_safe)
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
    ctx.title_note = ""
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
    outcome.title_note = ctx.title_note
    return outcome
