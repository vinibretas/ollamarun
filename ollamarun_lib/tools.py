"""Tool registry, safety policy and executor for the agent loop.

`ToolSpec` describes one tool (name, description, JSON-schema parameters and
a handler); `default_registry()` builds the built-in set (shell, python,
files, fetch, and -- when `tts-remote` is available -- speech).
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

from .helpers import now_iso, short_stamp

FETCH_TIMEOUT = 30.0
MAX_LIST_ENTRIES = 200
MAX_READ_CHARS = 200_000

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
PATH_RULE_TOOLS = {"read_file", "write_file", "list_dir"}

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
    # tts-remote lives in its own submodule: ~/bin/tts-remote/tts-remote
    fallback = Path.home() / "bin" / "tts-remote"
    if fallback.is_file():
        return str(fallback)
    entry = fallback / "tts-remote"
    return str(entry) if entry.is_file() else None


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
        fallback = Path.home() / "bin" / "tts-remote"
        available = bool(shutil.which("tts-remote")) \
            or fallback.is_file() or (fallback / "tts-remote").is_file()
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
