"""`speak`: queue text through tts-remote (the auto-speak path is the REPL's)."""
from __future__ import annotations

import subprocess
from typing import Any

from .spec import ToolContext, ToolError
from .tts import find_tts_remote, run_tts_say, speech_hosts


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ToolError("missing 'text'")
    binary = find_tts_remote(ctx)
    if not binary:
        raise ToolError("tts-remote not found on PATH")
    voice = args.get("voice")
    lang = args.get("lang")
    try:
        proc = run_tts_say(binary, text, voice=voice, lang=lang, timeout=ctx.timeout,
                           cancel=getattr(ctx, "cancel", None))
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
