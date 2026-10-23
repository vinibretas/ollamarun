"""Session settings: `/set`, `/system`, `/think(ing)`, `/render`, `/math`, `/pager`, `/tps`, `/tts` and `/show`."""
from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ...client import OllamaError
from ...constants import console
from ...helpers import (coerce_value, human_chars, parse_bool, parse_think,
                        shortcut)
from ...interactive import model_context_length
from ...renderer import thinking_panel
from ...tools import execute_tool, find_tts_remote

if TYPE_CHECKING:
    from ..core import Repl

from ..view import _last_assistant


def cmd_set(repl: Repl, argument: str) -> None:
    try:
        parts = shlex.split(argument)
    except ValueError as exc:
        console.print(f"[err]{exc}[/err]")
        return
    if not parts:
        cmd_params(repl, "")
        return
    kind = parts[0]
    if kind == "parameter":
        if len(parts) < 2:
            console.print("[err]usage: /set parameter NAME [VALUE][/err]")
            return
        if len(parts) == 2:
            repl.session.parameters.pop(parts[1], None)
            console.print(f"[ok]cleared[/ok] parameter {escape(parts[1])}")
        else:
            repl.session.parameters[parts[1]] = coerce_value(" ".join(parts[2:]))
            console.print(f"[ok]set[/ok] {escape(parts[1])} = "
                          f"{escape(json.dumps(repl.session.parameters[parts[1]]))}")
            repl.session.touch()
    elif kind == "system":
        _apply_system(repl, argument[len("system"):].strip())
    else:
        console.print("[err]usage: /set parameter NAME VALUE | /set system TEXT[/err]")


def cmd_params(repl: Repl, argument: str) -> None:
    table = Table(title="session parameters", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("parameter")
    table.add_column("value")
    if repl.session.parameters:
        for name, value in repl.session.parameters.items():
            table.add_row(name, escape(json.dumps(value)))
    else:
        table.add_row("[dim]none[/dim]", "")
    table.add_row("think", escape(str(repl.session.think)))
    table.add_row("keep_alive", escape(str(repl.session.keepalive or "")))
    table.add_row("gpu layers", escape(repl.gpu_layers_summary()))
    table.add_row("format", escape(str(repl.session.format or "")))
    table.add_row("system", escape(system_label(repl)))
    console.print(table)



def cmd_tts(repl: Repl, argument: str) -> None:
    parts = argument.split()
    topic = parts[0].lower() if parts else ""
    binary = find_tts_remote(repl.tool_ctx)
    if not topic or topic == "status":
        table = Table(title="tts-remote", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("setting")
        table.add_column("value")
        table.add_row("binary", escape(binary) if binary else "[err]not found[/err]")
        table.add_row("auto-speak", "[ok]on[/ok]" if repl.tts_auto else "off")
        table.add_row("voice", escape(repl.tts_voice or "(default)"))
        table.add_row("language", escape(repl.tts_lang or "(default)"))
        states = []
        for name in ("speak", "list_voices"):
            if name not in repl.tool_registry:
                states.append(f"{name}: not installed")
            elif name in repl.session.tools_disabled:
                states.append(f"{name}: disabled")
            else:
                states.append(f"{name}: on")
        table.add_row("model access", escape(", ".join(states)))
        console.print(table)
        return
    if topic in {"on", "off"}:
        repl.tts_auto = topic == "on"
        repl.session.tts_auto = repl.tts_auto
        repl.session.touch()
        console.print(f"[ok]auto-speak[/ok] {'on' if repl.tts_auto else 'off'}")
        return
    if topic in {"voice", "lang"}:
        value = " ".join(parts[1:]).strip()
        if topic == "voice":
            repl.tts_voice = value or None
            repl.session.tts_voice = repl.tts_voice
        else:
            repl.tts_lang = value or None
            repl.session.tts_lang = repl.tts_lang
        repl.session.touch()
        console.print(f"[ok]{topic}[/ok] {escape(value or '(default)')}")
        return
    if topic == "tools":
        action = parts[1].lower() if len(parts) > 1 else ""
        if action not in {"on", "off"}:
            console.print("[err]usage: /tts tools on|off[/err]")
            return
        if not {"speak", "list_voices"} & set(repl.tool_registry):
            console.print("[warn]tts-remote not found; nothing to toggle[/warn]")
            return
        disabled = list(repl.session.tools_disabled)
        for name in ("speak", "list_voices"):
            if name not in repl.tool_registry:
                continue
            if action == "off" and name not in disabled:
                disabled.append(name)
            if action == "on" and name in disabled:
                disabled.remove(name)
        repl.session.tools_disabled = disabled
        repl.session.touch()
        console.print(f"[ok]tts tools {action}[/ok]")
        return
    if topic == "voices":
        spec = repl.tool_registry.get("list_voices")
        if spec is None:
            console.print("[err]tts-remote not found[/err]")
            return
        outcome = execute_tool(spec, {}, repl.tool_ctx)
        console.print(Panel(escape(outcome.text), title="tts voices",
                            border_style="dim" if outcome.ok else "red",
                            title_align="left"))
        return
    console.print("[err]usage: /tts [on|off|voice NAME|lang CODE|tools on|off|"
                  "voices][/err]")


def cmd_show(repl: Repl, argument: str) -> None:
    topic = (argument or "info").split()[0]
    try:
        data = repl.client.show(repl.session.model)
    except OllamaError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    if topic in {"info"}:
        details = data.get("details") or {}
        table = Table(title=f"model · {repl.session.model}", box=box.ROUNDED,
                      header_style="bold magenta")
        table.add_column("field")
        table.add_column("value")
        for field_name, value in (
            ("family", details.get("family")),
            ("parameter size", details.get("parameter_size")),
            ("quantization", details.get("quantization_level")),
            ("format", details.get("format")),
            ("capabilities", ", ".join(data.get("capabilities") or [])),
            ("context length", model_context_length(data, repl.session.parameters)),
            ("modified", data.get("modified_at")),
        ):
            table.add_row(field_name, escape(str(value if value is not None else "")))
        console.print(table)
        cmd_params(repl, "")
        return
    key_map = {"license": "license", "modelfile": "modelfile", "parameters": "parameters",
               "system": "system", "template": "template"}
    if topic not in key_map:
        console.print("[err]usage: /show [info|license|modelfile|parameters|system|template][/err]")
        return
    value = data.get(key_map[topic]) or "(empty)"
    if topic in {"modelfile", "template"}:
        from rich.syntax import Syntax
        console.print(Panel(Syntax(str(value), "dockerfile" if topic == "modelfile" else "text"),
                            title=topic, border_style="dim", title_align="left"))
    else:
        console.print(Panel(escape(str(value)), title=topic, border_style="dim",
                            title_align="left"))



def cmd_think(repl: Repl, argument: str) -> None:
    if not argument:
        console.print(f"think: [key]{repl.session.think if repl.session.think is not None else 'default'}[/key]")
        return
    try:
        repl.session.think = parse_think(argument)
    except ValueError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    repl.session.touch()
    console.print(f"[ok]think[/ok] {repl.session.think}")


def cmd_thinking(repl: Repl, argument: str) -> None:
    """Display preference for the model's thinking (not the model's think flag)."""
    topic = argument.split()[0].lower() if argument.split() else ""
    if topic in {"full", "show-full", "last"}:
        message = _last_assistant(repl)
        thinking = (message or {}).get("thinking")
        if not thinking:
            console.print("[dim]the last response has no stored thinking[/dim]")
            return
        console.print(thinking_panel(thinking, f"thinking · {human_chars(len(thinking))}"))
        return
    if topic in {"", "status"}:
        console.print(f"thinking display: [key]{repl.thinking_mode}[/key] "
                      "(collapsed|expanded|hidden)")
        return
    if topic in {"expand", "expanded", "show", "on"}:
        repl.thinking_mode = "expanded"
    elif topic in {"collapse", "collapsed"}:
        repl.thinking_mode = "collapsed"
    elif topic in {"hide", "hidden", "off"}:
        repl.thinking_mode = "hidden"
    elif topic == "toggle":
        repl.thinking_mode = "collapsed" if repl.thinking_mode == "expanded" else "expanded"
    else:
        console.print("[err]usage: /thinking [show|collapse|hide|toggle|full][/err]")
        return
    console.print(f"[ok]thinking[/ok] {repl.thinking_mode}")


def _apply_system(repl: Repl, text: str) -> None:
    """One place for the /system and /set system semantics.

    None means "use the built-in default", an empty string means "send no
    system prompt at all", any other text is the session's own prompt.
    """
    if text in ("off", "none"):
        repl.session.system = ""
        console.print("[ok]system prompt disabled[/ok]")
    elif text in ("", "clear", "default", "reset"):
        repl.session.system = None
        console.print("[ok]system prompt reset to the built-in default[/ok]")
    else:
        repl.session.system = text
        console.print("[ok]system prompt set[/ok]")
    repl.session.touch()


def _show_system(repl: Repl) -> None:
    if repl.session.system:
        console.print(Panel(escape(repl.session.system), title="system",
                            border_style="dim", title_align="left"))
        return
    if repl.session.system == "":
        console.print("[dim]no system prompt (disabled; /system clear re-enables "
                      "the built-in default)[/dim]")
        return
    text = repl.effective_system() or ""
    if not text:
        console.print("[warn]the built-in system prompt file is missing "
                      "(defs/prompts/system.md)[/warn]")
        return
    console.print(Panel(escape(text), title="system (built-in default)",
                        border_style="dim", title_align="left"))


def system_label(repl: Repl) -> str:
    """One-line system-prompt state for the parameter/session tables."""
    if repl.session.system:
        return shortcut(repl.session.system, 60)
    if repl.session.system == "":
        return "disabled"
    text = repl.effective_system()
    return f"built-in default ({len(text)} chars)" if text else "none"


def cmd_system(repl: Repl, argument: str) -> None:
    if not argument:
        _show_system(repl)
        return
    _apply_system(repl, argument.strip())



def cmd_render(repl: Repl, argument: str) -> None:
    if not argument:
        console.print(f"render: [key]{repl.render_mode}[/key]")
        return
    if argument not in {"markdown", "plain", "raw", "auto"}:
        console.print("[err]usage: /render [markdown|plain|raw|auto][/err]")
        return
    repl.render_mode = argument
    console.print(f"[ok]render[/ok] {argument}")


def cmd_math(repl: Repl, argument: str) -> None:
    if not argument:
        console.print(f"math: [key]{'on' if repl.math else 'off'}[/key]")
        return
    repl.math = parse_bool(argument, repl.math)
    console.print(f"[ok]math {'on' if repl.math else 'off'}[/ok]")


def cmd_pager(repl: Repl, argument: str) -> None:
    if not argument:
        console.print(f"pager: [key]{'on' if repl.pager else 'off'}[/key]")
        return
    repl.pager = parse_bool(argument, repl.pager)
    console.print(f"[ok]pager {'on' if repl.pager else 'off'}[/ok]")


def cmd_tps(repl: Repl, argument: str) -> None:
    if not argument:
        console.print(f"tps footer: [key]{'on' if repl.tps_footer else 'off'}[/key]")
        return
    repl.tps_footer = parse_bool(argument, repl.tps_footer)
    console.print(f"[ok]tps footer[/ok] {'on' if repl.tps_footer else 'off'}")


