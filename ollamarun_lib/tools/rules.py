"""Stored approval rules: matching, suggestions and the dangerous floor.

A session can remember a generalized grant -- `shell: mkdir -p :*`,
`path: ~/proj/**`, `python: imports os,sys` -- and `rule_matches()` decides
whether it covers a call. `suggest_rules()` derives the grants to offer when
a call asks, and `dangerous_command()` is the floor no rule ever covers.
"""
from __future__ import annotations

import ast
import fnmatch
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..helpers import now_iso
from .args import call_fingerprint
from .policy import (_SHELL_SPLIT, _dangerous_shell_args, _python_findings,
                     approval_decision, shell_safety)
from .spec import ToolSpec

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
                   "make_dir", "delete_file", "publish_artifact"}
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


@dataclass
class PolicyCheck:
    """The non-interactive half of the approval gate.

    Shared by the console/TUI prompt path (`Repl._approve_tool_call`) and the
    headless serve approver, so both ask for exactly the calls the policy
    does not already cover. `run` is True with `note` saying why; when it is
    False, `reason` is the policy's explanation and the caller decides
    between parking for a decision and denying outright.
    """

    run: bool
    note: str = ""
    reason: str = ""
    rule: Optional[ToolRule] = None


def policy_check(spec: ToolSpec, args: dict[str, Any], *,
                 policy: str = "allowlist",
                 shell_allow: Optional[set[str]] = None,
                 fingerprints: Optional[list[str]] = None,
                 rules: Optional[list[ToolRule]] = None,
                 fingerprint: str = "", cwd: str = ".") -> PolicyCheck:
    """Classify a call as covered by policy or needing a decision."""
    decision = approval_decision(spec, args, policy=policy,
                                 shell_allow=shell_allow)
    if decision.allowed:
        return PolicyCheck(True, "safe list")
    if fingerprint and fingerprint in (fingerprints or []):
        return PolicyCheck(True, "previously allowed")
    rule = matching_rule(list(rules or []), spec.name, args, cwd=cwd,
                         shell_allow=shell_allow)
    if rule is not None:
        return PolicyCheck(True, f"rule: {rule_label(rule)}", rule=rule)
    return PolicyCheck(False, reason=decision.reason or "needs confirmation")
