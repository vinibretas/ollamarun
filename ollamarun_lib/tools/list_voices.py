"""`list_voices`: the tts-remote voice catalogue."""
from __future__ import annotations

import json
import subprocess
from typing import Any

from ..helpers import run_cancellable
from .spec import ToolContext, ToolError
from .tts import find_tts_remote


def run(args: dict[str, Any], ctx: ToolContext) -> str:
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
