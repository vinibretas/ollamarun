"""The tts-remote subprocess plumbing shared by the speech tools."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from ..helpers import run_cancellable
from .spec import ToolContext


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
                timeout: float = 120.0, cancel: Optional[Any] = None
                ) -> subprocess.CompletedProcess:
    """Queue speech via `tts-remote say`; an older CLI is retried without --json.

    The `--json` report behind the speak hosts note is new, so a tts-remote
    that does not know the flag yet still gets to speak -- it just comes back
    without the report. Raises subprocess.TimeoutExpired when the CLI never
    returns; the caller turns that into its own message. `cancel` makes the
    synthesis killable by a stopped turn.
    """
    command = [binary, "say", "-q", "--json"]
    if markdown:
        command.append("--markdown")
    if voice:
        command += ["--voice", str(voice)]
    if lang:
        command += ["--lang", str(lang)]
    proc = run_cancellable(command, input_text=text, timeout=timeout,
                           cancel=cancel)
    if proc.returncode != 0 and "--json" in (proc.stderr or ""):
        proc = run_cancellable([part for part in command if part != "--json"],
                               input_text=text, timeout=timeout, cancel=cancel)
    return proc
