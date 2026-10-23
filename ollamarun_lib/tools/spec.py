"""Tool-harness types and output plumbing.

`ToolSpec` describes one tool (name, description, JSON-schema parameters and a
handler); `ToolContext` carries the per-session runtime seams a handler may
use; `ToolOutcome`/`ToolHandlerResult` are what comes back out. Nothing here
imports a tool module: every handler and the registry sit on top of this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..helpers import CommandCancelled

#: One `read_pdf` call rasterizes at most this many pages (config
#: `pdf_vision_pages` can lower it, never raise it).
MAX_PDF_VISION_PAGES = 10


TRUNCATE_TAIL = 160


class ToolError(Exception):
    """A tool could not run; the message goes back to the model as-is."""


#: The tools-level name for a subprocess call a cancelled turn killed. The
#: generic implementation lives in helpers.py so the ssh probes can cancel too.
ToolCancelled = CommandCancelled


@dataclass
class ToolContext:
    """Runtime settings shared by every handler of one session."""
    cwd: str = "."
    #: The API workspace root (`ollamarun serve`). When set, path arguments
    #: resolve under it and escapes (symlinks followed) are refused; the
    #: CLI leaves it None and keeps its previous behavior. This governs
    #: path arguments and default working directories — it is never a
    #: claim that a subprocess can only reach the root.
    workspace: Optional[str] = None
    timeout: float = 245.0
    max_output: int = 40000
    tts_remote: Optional[str] = None
    #: Live handles injected by the REPL: the Ollama client (`list_models`,
    #: `system_status`) and the session store (`search_sessions`). Typed Any,
    #: so this module keeps its one-way dependency graph; None outside a REPL.
    client: Any = None
    store: Any = None
    #: `--remote` details for probing the machine running Ollama: the ssh
    #: target/user, or None on the local machine.
    ssh_target: Optional[str] = None
    ssh_user: Optional[str] = None
    #: Path/command of the chemInventoryAPI CLI (config `chem_inventory`).
    chem_inventory: Optional[str] = None
    #: Environment reports the REPL injects: `context_report()` returns the
    #: context window/usage data and `session_report()` the identity, settings
    #: and tool state. None outside a REPL (the tools then say so).
    context_report: Optional[Callable[[], dict[str, Any]]] = None
    session_report: Optional[Callable[[], dict[str, Any]]] = None
    #: URL -> formatted result, so a repeated fetch is answered from the cache
    #: (the model is told instead of silently re-downloading the same page).
    fetched: dict[str, str] = field(default_factory=dict)
    #: lowercased query -> formatted web_search results, for the same reason.
    searches: dict[str, str] = field(default_factory=dict)
    #: Free-form note a handler wants in the result panel title (the speak
    #: tool sets `optiplex → g3` there); execute_tool() moves it onto the
    #: ToolOutcome, so callers never read this directly.
    title_note: str = ""
    #: The turn's `CancelToken`, when one exists. Subprocess handlers pass it
    #: to `run_cancellable()` so a stop kills the child's process group; the
    #: ToolRunner abandons network calls on it. None outside a turn.
    cancel: Optional[Any] = None
    #: The turn-bound artifact publisher (`artifacts.TurnArtifacts`) when the
    #: host runs a store (`ollamarun serve`); file-producing tools return
    #: `ToolHandlerResult.artifact_paths` and `execute_tool()` registers them
    #: here. Typed Any for the same one-way reason as `client`/`store`.
    artifacts: Any = None
    #: The session's model right now, re-read per call so `/model` switches
    #: are seen (`read_pdf` decides whether it can see images, and what it
    #: would have to unload for a scan). None outside a REPL.
    current_model: Optional[Callable[[], Optional[str]]] = None
    #: Raw `/api/show` through the host's run-long cache, for capabilities.
    #: None falls back to a direct `client.show()`.
    model_show: Optional[Callable[[str], dict[str, Any]]] = None
    #: Scanned-PDF vision settings, from the `pdf_vision*` config keys.
    pdf_vision: bool = True
    pdf_vision_switch: bool = True
    pdf_vision_model: Optional[str] = None
    pdf_vision_pages: int = MAX_PDF_VISION_PAGES
    #: The transcription prompt (loaded from `defs/prompts/pdf_vision.md`);
    #: empty means the built-in fallback.
    vision_prompt: str = ""
    #: "label#pages#mode" -> formatted `read_pdf` result, so a repeated read
    #: is answered from the session instead of re-running vision.
    pdf_reads: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolOutcome:
    text: str = ""
    ok: bool = False
    duration: float = 0.0
    error: Optional[str] = None
    #: Extra title text for the result panel (empty for most tools).
    title_note: str = ""
    #: True when the turn was stopped while this call was queued or running;
    #: `text` says whether it was interrupted or abandoned.
    cancelled: bool = False
    #: Refs (`ArtifactRef.to_dict()`) registered for this call, in order.
    artifacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolHandlerResult:
    """A handler's structured result: text plus the files it produced.

    Every existing handler may keep returning a plain string; handlers that
    create files return this instead and `execute_tool()` registers the
    paths through the turn's artifact publisher (regular files only --
    directories are skipped, nothing is scanned recursively).
    """

    text: str
    artifact_paths: list[str] = field(default_factory=list)


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
    #: capability the tool needed to appear (None, "search", "tts" or "chem")
    requires: Optional[str] = None
    #: names models tend to invent for this tool, from the definition file
    aliases: list[str] = field(default_factory=list)


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
