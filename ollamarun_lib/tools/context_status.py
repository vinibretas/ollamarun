"""`context_status`: the context budget the model actually runs with."""
from __future__ import annotations

from typing import Any

from .spec import ToolContext, ToolError


def run(args: dict[str, Any], ctx: ToolContext) -> str:
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
        note = ("model max " + str(model_max)) if model_max else "no /api/show answer"
        lines.append(f"context window: {data.get('source') or 'unknown'} ({note})")
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
