"""Approval policy: the shell/python allowlists and the safety analysis.

`shell_safety()` and `python_safety()` answer "is this provably harmless?"
for the unattended allowlist; `approval_decision()` turns that plus the
tool's flags and the session policy into a run/ask decision. The rule engine
that can widen this lives in `rules.py`; the constants here are what both it
and the suggestions share.
"""
from __future__ import annotations

import ast
import os
import re
import shlex
from typing import Any, Optional

from .spec import Decision, ToolSpec


#: Commands `run_shell` may execute unattended (basename of argv[0]).
DEFAULT_SHELL_ALLOW = {
    "ls", "pwd", "cat", "head", "tail", "grep", "rg", "find", "wc", "file",
    "stat", "tree", "du", "df", "sort", "cut", "tr", "diff", "echo", "date",
    "which", "whoami", "uname", "git",
}

#: How a command line is split into segments for per-segment safety checks.
_SHELL_SPLIT = re.compile(r"(?:\|\||&&|[;|&\n])")
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
