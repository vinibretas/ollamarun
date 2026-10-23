"""`session_info`: date, session identity, model, settings and tool state."""
from __future__ import annotations

import socket
import sys
from datetime import datetime
from typing import Any

from ..helpers import short_stamp
from .spec import ToolContext, ToolError


def run(args: dict[str, Any], ctx: ToolContext) -> str:
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
