"""The `/tools` command: the table, per-tool control, help pages, `call` and the stored approval rules."""
from __future__ import annotations

from typing import TYPE_CHECKING
import json
import os
import shlex
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Optional
from rich import box
from rich.markup import escape
from rich.table import Table
from ...constants import console, tool_color
from ...helpers import coerce_value, now_iso, shortcut
from ...renderer import tool_call_panel, tool_result_panel
from ...tools import (
    REQUIREMENT_HINTS,
    ToolDefinition,
    ToolRule,
    ToolSpec,
    call_detail,
    call_fingerprint,
    execute_tool,
    resolve_tool_alias,
    rule_label,
    tool_definitions,
)
from .tables import TOOLS_HELP_ROWS, TOOLS_SUBCOMMANDS
from ..completion import tool_name_candidates

if TYPE_CHECKING:
    from ..core import Repl


def _rule_rows(repl: Repl) -> list[tuple[str, str, str, str]]:
    """(kind, pattern, base, created) for every stored approval, call-first."""
    rows: list[tuple[str, str, str, str]] = []
    for fingerprint in repl.session.tool_allow or []:
        rows.append(("call", fingerprint, "", ""))
    for entry in repl.session.tool_rules or []:
        rule = ToolRule.from_dict(entry)
        if rule is not None:
            rows.append((rule.kind, rule.pattern, rule.base, rule.created))
    return rows


def _cmd_tools_rules_rm(repl: Repl, parts: list[str]) -> None:
    indices = sorted({int(part) - 1 for part in parts if part.isdigit()})
    if not indices:
        console.print("[err]usage: /tools rules rm N…[/err]")
        return
    allow = list(repl.session.tool_allow or [])
    stored = [entry for entry in (repl.session.tool_rules or [])
              if ToolRule.from_dict(entry) is not None]
    merged: list[tuple[str, int]] = [("call", index) for index in range(len(allow))]
    merged += [("rule", index) for index in range(len(stored))]
    drop = {index for index in indices if 0 <= index < len(merged)}
    if not drop:
        console.print("[warn]no rule with that number[/warn]")
        return
    repl.session.tool_allow = [allow[index] for position, (kind, index)
                               in enumerate(merged)
                               if position not in drop and kind == "call"]
    repl.session.tool_rules = [stored[index] for position, (kind, index)
                               in enumerate(merged)
                               if position not in drop and kind == "rule"]
    repl.session.touch()
    console.print(f"[ok]removed[/ok] {len(drop)} rule(s)")


def _cmd_tools_rules_add(repl: Repl, parts: list[str]) -> None:
    kind = parts[0].lower() if parts else ""
    spec = parts[1:]
    if kind == "shell" and spec:
        pattern = " ".join(spec)
        try:
            tokens = shlex.split(pattern)
        except ValueError as exc:
            console.print(f"[err]unparsable pattern: {exc}[/err]")
            return
        if not tokens or any(char in pattern for char in "$`><(){}&|;"):
            console.print("[err]pattern contains shell metacharacters[/err]")
            return
        if any(token == ":*" for token in tokens[:-1]):
            console.print("[err]':*' is only valid at the end[/err]")
            return
        rule = ToolRule(kind="shell", pattern=pattern, created=now_iso())
    elif kind == "path" and len(spec) >= 2:
        base = Path(spec[0]).expanduser()
        if not base.is_dir():
            console.print(f"[err]not a directory: {escape(str(base))}[/err]")
            return
        pattern = " ".join(spec[1:])
        if pattern.startswith("/") or ".." in Path(pattern).parts:
            console.print("[err]path patterns are relative to BASE and cannot "
                          "escape it[/err]")
            return
        rule = ToolRule(kind="path", pattern=pattern,
                        base=os.path.realpath(base), created=now_iso())
    elif kind == "python" and len(spec) == 1:
        roots = sorted({root.strip() for root in spec[0].split(",") if root.strip()})
        if not roots or any(not root.isidentifier() for root in roots):
            console.print("[err]usage: /tools rules add python mod1,mod2[/err]")
            return
        rule = ToolRule(kind="python", pattern=",".join(roots), created=now_iso())
    elif kind == "call" and spec:
        rule = ToolRule(kind="call", pattern=" ".join(spec), created=now_iso())
    else:
        console.print("[err]usage: /tools rules add shell 'PREFIX :*' | "
                      "path BASE PATTERN | python mod1,mod2 | call FINGERPRINT[/err]")
        return
    repl.session.tool_rules = list(repl.session.tool_rules or []) + [rule.to_dict()]
    repl.session.touch()
    console.print(f"[ok]added[/ok] [key]{escape(rule_label(rule))}[/key]")


def _cmd_tools_rules(repl: Repl, parts: list[str]) -> None:
    topic = parts[0].lower() if parts else ""
    if topic in {"rm", "remove", "delete"}:
        _cmd_tools_rules_rm(repl, parts[1:])
        return
    if topic == "add":
        _cmd_tools_rules_add(repl, parts[1:])
        return
    rows = _rule_rows(repl)
    if not rows:
        console.print("[dim]no stored approval rules — pick a pattern number in "
                      "the confirmation prompt or add one with /tools rules add[/dim]")
        return
    table = Table(title="approval rules", box=box.ROUNDED,
                  header_style="bold magenta", row_styles=["", "dim"])
    table.add_column("#", justify="right")
    table.add_column("kind")
    table.add_column("rule")
    table.add_column("created", style="dim")
    for index, (kind, pattern, base, created) in enumerate(rows, 1):
        if kind == "call":
            label = "exact: " + shortcut(pattern, 60)
        else:
            label = rule_label(ToolRule(kind=kind, pattern=pattern, base=base))
        table.add_row(str(index), kind, escape(label),
                      shortcut(created, 16) if created else "")
    console.print(table)
    console.print("[dim]/tools rules rm N… removes one · changes apply to this "
                  "session[/dim]")


def _find_tool(repl: Repl, name: str
               ) -> tuple[Optional[ToolSpec], Optional[ToolDefinition]]:
    """(registry spec, definition) for a tool name or alias; each may be None.

    The definition exists even when the tool is not installed (its `requires`
    is unmet), so a help page can still explain what it would do.
    """
    key = str(name or "").strip().lower().replace("-", "_")
    canonical = resolve_tool_alias(key)
    spec = repl.tool_registry.get(canonical)
    definition = next((item for item in tool_definitions()
                       if item.name == canonical), None)
    return spec, definition


def _apply_tool_toggle(repl: Repl, name: str, on: bool) -> None:
    """Record NAME on/off in the session (the one place that edits it)."""
    disabled = list(repl.session.tools_disabled)
    if on:
        if name in disabled:
            disabled.remove(name)
    elif name not in disabled:
        disabled.append(name)
    repl.session.tools_disabled = disabled
    repl.session.touch()


def _missing_requirement(definition: Optional[ToolDefinition]) -> str:
    needs = definition.requires if definition is not None else None
    return REQUIREMENT_HINTS.get(needs, needs or "?")


def _tool_state(repl: Repl, spec: Optional[ToolSpec],
                definition: Optional[ToolDefinition]) -> str:
    """The state label a tool's help header and toggle share."""
    if spec is None:
        missing = escape(_missing_requirement(definition))
        return f"[err]not installed[/err] — needs {missing}"
    if not repl.tools_enabled:
        return "[dim]harness off[/dim]"
    if repl.session.format:
        return "[dim]off (format)[/dim]"
    if spec.name in repl.session.tools_disabled:
        return "[warn]disabled for the model[/warn]"
    return "[ok]on[/ok]"


def _tools_off_note(repl: Repl) -> Optional[str]:
    """Why the harness is not carrying tools, or None when it is."""
    if not repl.tools_enabled:
        return ("[dim]note: the tool harness is off globally — /tools on to "
                "bring it back[/dim]")
    if repl.session.format:
        return ("[dim]note: a response format is set — tool calls are not sent; "
                "start a session without --format to use tools[/dim]")
    return None


def _tool_approval_line(spec: ToolSpec) -> str:
    """How a call to SPEC is approved, in the same words as the /tools table."""
    if spec.read_only:
        return "auto — read-only, never asks"
    if spec.always_safe:
        return "auto — runs without confirmation"
    if spec.name == "run_shell":
        return ("safe-command allowlist and stored rules, else asks "
                "(/tools allow, /tools rules)")
    if spec.name == "run_python":
        return "AST-safe snippets and stored rules, else asks (/tools rules)"
    return "stored rules, else asks (/tools rules)"


def _print_tool_help(repl: Repl, spec: Optional[ToolSpec],
                     definition: Optional[ToolDefinition]) -> None:
    """One tool's page: state, description, parameters and approval."""
    source = spec if spec is not None else definition
    if source is None:
        return
    canonical = source.name
    aliases = spec.aliases if spec is not None else definition.aliases
    color = tool_color(canonical)
    header = (f"[{color}]{escape(canonical)}[/{color}] — "
              f"{_tool_state(repl, spec, definition)}")
    if aliases:
        header += " · aliases: " + escape(", ".join(aliases))
    console.print(header)
    console.print(escape(source.description))
    parameters = source.parameters or {}
    properties = parameters.get("properties") or {}
    required = [key for key in (parameters.get("required") or [])]
    order = list(properties)
    signature = ", ".join(key + ("" if key in required else "?") for key in order)
    console.print(f"[key]model calls it with[/key] {canonical}({signature})"
                  + ("" if order else " — no arguments"))
    if order:
        table = Table(box=box.SIMPLE, header_style="bold", pad_edge=False)
        table.add_column("parameter", style=color)
        table.add_column("type", style="dim")
        table.add_column("", style="dim")
        table.add_column("description")
        for key in order:
            prop = properties.get(key) or {}
            table.add_row(escape(key), escape(str(prop.get("type") or "any")),
                          "required" if key in required else "optional",
                          escape(str(prop.get("description") or "")))
        console.print(table)
    if spec is not None:
        console.print(f"[key]approval[/key] {_tool_approval_line(spec)}")
    console.print(f"[dim]/tools {escape(canonical)} "
                  f"{escape('[on|off|help|call ARGS]')}[/dim]")


def _tool_call_args(spec: ToolSpec, raw: str) -> dict[str, Any]:
    """Arguments for `/tools NAME call`: JSON, key=value pairs, or one bare
    value when exactly one parameter is required (`call notes.txt`).

    Everything is checked against the tool's schema: unknown or missing
    parameters are refused with the valid names spelled out.
    """
    text = raw.strip()
    parameters = spec.parameters or {}
    properties = parameters.get("properties") or {}
    required = [key for key in (parameters.get("required") or [])]
    if not text:
        args: dict[str, Any] = {}
    elif text.startswith("{"):
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"invalid JSON arguments: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("JSON arguments must be an object")
        args = value
    else:
        try:
            tokens = shlex.split(text)
        except ValueError as exc:
            raise ValueError(f"unparsable arguments: {exc}") from exc
        keyed = [token for token in tokens if "=" in token]
        if keyed and len(keyed) == len(tokens):
            args = {}
            for token in tokens:
                key, _, value = token.partition("=")
                if not key:
                    raise ValueError(f"missing parameter name in {token!r}")
                args[key] = coerce_value(value)
        elif not keyed and len(required) == 1:
            args = {required[0]: " ".join(tokens)}
        else:
            raise ValueError(
                'usage: /tools NAME call \'{"parameter": value}\' | parameter=value …'
                + (f" | VALUE (for {required[0]})" if len(required) == 1 else ""))
    unknown = [key for key in args if key not in properties]
    if unknown:
        takes = (f"{spec.name} takes: " + ", ".join(properties)
                 if properties else f"{spec.name} takes no arguments")
        raise ValueError(f"unknown parameter(s) {', '.join(unknown)} — {takes}")
    missing = [key for key in required if key not in args]
    if missing:
        raise ValueError(f"missing required parameter(s): {', '.join(missing)}"
                         f" (/tools {spec.name} help)")
    return args


def _cmd_tools_call(repl: Repl, spec: ToolSpec, raw: str) -> None:
    """Run one tool exactly like the model would: approve, execute, render.

    The result is shown but never added to the conversation: a manual call
    must not change what the model sees (or how much context it carries).
    """
    try:
        args = _tool_call_args(spec, raw)
    except ValueError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    detail = call_detail(spec.name, args)
    console.print(tool_call_panel(spec.name, args))
    if spec.name in repl.session.tools_disabled:
        console.print(f"[warn]{escape(spec.name)} is disabled for the model "
                      f"— this manual call does not change that[/warn]")
    try:
        run, note = repl._approve_tool_call(spec, args,
                                            call_fingerprint(spec.name, args))
    except KeyboardInterrupt:
        console.print("[warn]cancelled[/warn]")
        return
    if not run:
        console.print(tool_result_panel(spec.name, detail=detail,
                                        text=f"not run: {note}", ok=False))
        return
    repl.set_state("tool", detail or spec.name)
    repl.ui.notify("tool_start", name=spec.name, detail=detail)
    started = time.monotonic()
    try:
        outcome = execute_tool(spec, args, repl.tool_ctx)
    except KeyboardInterrupt:
        console.print(tool_result_panel(spec.name, detail=detail,
                                        text="interrupted by user", ok=False))
        repl.set_state("ready")
        return
    repl.ui.notify("tool_done", name=spec.name, ok=outcome.ok,
                   duration=outcome.duration or (time.monotonic() - started))
    console.print(tool_result_panel(spec.name, detail=detail, text=outcome.text,
                                    ok=outcome.ok, duration=outcome.duration,
                                    note=outcome.title_note))
    repl.set_state("ready")


def _cmd_tools_named(repl: Repl, name: str, parts: list[str]) -> None:
    """`/tools NAME [on|off|help|call ARGS]` for one tool (aliases resolve)."""
    spec, definition = _find_tool(repl, name)
    if spec is None and definition is None:
        console.print(f"[err]unknown tool {escape(name)}[/err]")
        suggestions = get_close_matches(name.lower(),
                                        tool_name_candidates(repl, ""), n=3)
        if suggestions:
            console.print("[dim]did you mean: " + ", ".join(suggestions) + "?[/dim]")
        console.print("[dim]/tools lists every tool[/dim]")
        return
    canonical = spec.name if spec is not None else definition.name
    action = parts[0].lower() if parts else ""
    if action in {"", "help", "info", "?"}:
        _print_tool_help(repl, spec, definition)
        return
    if action in {"on", "off", "enable", "disable"}:
        if spec is None:
            console.print(f"[err]{escape(canonical)} is not installed[/err] — "
                          f"needs {escape(_missing_requirement(definition))}")
            return
        on = action in {"on", "enable"}
        _apply_tool_toggle(repl, spec.name, on)
        console.print(f"[ok]{escape(spec.name)} {'on' if on else 'off'}[/ok]")
        note = _tools_off_note(repl)
        if note:
            console.print(note)
        return
    if action == "call":
        if spec is None:
            console.print(f"[err]{escape(canonical)} is not installed[/err] — "
                          f"needs {escape(_missing_requirement(definition))}")
            return
        _cmd_tools_call(repl, spec, " ".join(parts[1:]))
        return
    console.print(f"[err]usage: /tools {escape(name)} "
                  f"{escape('[on|off|help|call ARGS]')}[/err]")


def _cmd_tools_help_table() -> None:
    """`/tools help`: every form the command accepts, in one table."""
    table = Table(title="/tools", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("form")
    table.add_column("what it does")
    for form, what in TOOLS_HELP_ROWS:
        table.add_row(escape(form), what)
    console.print(table)
    console.print("[dim]NAME is any tool name or alias (Tab completes them) · "
                  "/tools NAME help shows its parameters[/dim]")


def cmd_tools(repl: Repl, argument: str) -> None:
    parts = argument.split()
    topic = parts[0].lower() if parts else ""
    if not topic or topic in {"status", "list"}:
        table = Table(title="tools", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("tool")
        table.add_column("state")
        table.add_column("approval", style="dim")
        disabled = set(repl.session.tools_disabled)
        for name in sorted(repl.tool_registry):
            spec = repl.tool_registry[name]
            if not repl.tools_active():
                state = "[dim]off[/dim]"
            elif name in disabled:
                state = "[warn]disabled[/warn]"
            else:
                state = "[ok]on[/ok]"
            if spec.read_only:
                approval = "auto (read-only)"
            elif spec.always_safe:
                approval = "auto"
            elif name in {"run_shell", "run_python"}:
                approval = "safe list, rules, else asks"
            else:
                approval = "rules, else asks"
            color = tool_color(name)
            table.add_row(f"[{color}]{escape(name)}[/{color}]", state, approval)
        console.print(table)
        policy = "auto (--yes)" if repl.auto_approve else repl.tools_policy
        console.print(f"[key]policy[/key] {policy} · [key]rules[/key] "
                      f"{len(_rule_rows(repl))} (/tools rules) · [key]max rounds[/key] "
                      f"{repl.tools_max_rounds} · [key]timeout[/key] "
                      f"{repl.tool_ctx.timeout:.0f}s · [key]output cap[/key] "
                      f"{repl.tool_ctx.max_output} chars")
        note = _tools_off_note(repl)
        if note:
            console.print(note)
        console.print("[dim]/tools NAME help shows one tool's parameters · "
                      "/tools help explains every form[/dim]")
        return
    if topic == "help":
        if len(parts) > 1:
            _cmd_tools_named(repl, parts[1], parts[2:])
        else:
            _cmd_tools_help_table()
        return
    if topic == "on":
        repl.tools_enabled = True
        console.print("[ok]tools on[/ok]")
        note = _tools_off_note(repl)
        if note:
            console.print(note)
        return
    if topic == "off":
        repl.tools_enabled = False
        console.print("[ok]tools off[/ok]")
        return
    if topic == "policy":
        mode = parts[1].lower() if len(parts) > 1 else ""
        if mode not in {"auto", "allowlist", "ask"}:
            console.print("[err]usage: /tools policy auto|allowlist|ask[/err]")
            return
        repl.tools_policy = mode
        console.print(f"[ok]policy[/ok] {mode}")
        return
    if topic in {"enable", "disable"}:
        names = parts[1:]
        if not names:
            console.print(f"[err]usage: /tools {topic} NAME…[/err]")
            return
        for name in names:
            spec, definition = _find_tool(repl, name)
            if spec is None:
                hint = (f" — {escape(definition.name)} is not installed"
                        if definition is not None else "")
                console.print(f"[err]unknown tool {escape(name)}[/err]{hint}")
                continue
            _apply_tool_toggle(repl, spec.name, topic == "enable")
        console.print(f"[ok]{topic}d[/ok] active: "
                      + (", ".join(sorted(set(repl.tool_registry)
                                         - set(repl.session.tools_disabled)))
                         or "[dim]none[/dim]"))
        note = _tools_off_note(repl)
        if note:
            console.print(note)
        return
    if topic == "allow":
        console.print("safe shell commands: " + ", ".join(sorted(repl.shell_allow)))
        return
    if topic == "rules":
        _cmd_tools_rules(repl, parts[1:])
        return
    # any other first token is a tool name or an alias for one
    spec, definition = _find_tool(repl, parts[0])
    if spec is not None or definition is not None:
        _cmd_tools_named(repl, parts[0], parts[1:])
        return
    console.print(f"[err]unknown tool or subcommand {escape(parts[0])}[/err]")
    suggestions = get_close_matches(
        parts[0].lower(),
        list(TOOLS_SUBCOMMANDS) + tool_name_candidates(repl, ""), n=3)
    if suggestions:
        console.print("[dim]did you mean: " + ", ".join(suggestions) + "?[/dim]")
    console.print("[err]usage: /tools "
                  + escape("[on|off|policy MODE|enable|disable NAME…|"
                           "NAME on|off|NAME help|NAME call ARGS|allow|rules|help]")
                  + "[/err]")


