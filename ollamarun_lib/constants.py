"""Shared imports, paths and constants used across the ollamarun_lib package.

Every other module in this package imports from here rather than repeating
these definitions -- this is the module every other stage of the pipeline
(see the top-level package docstring) sits on top of.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import hashlib
import io
import json
import logging
import os
import random
import re
import select
import shlex
import shutil
import subprocess
import sys
import tempfile
import termios
import time
import tty
import urllib.parse
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import requests
import rich
import rich.traceback
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.errors import StyleSyntaxError
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.patch_stdout import patch_stdout
    HAS_PROMPT_TOOLKIT = True
except ImportError:  # pragma: no cover - fallback path
    HAS_PROMPT_TOOLKIT = False

    class _MissingPromptToolkit:
        """Stand-in for the prompt_toolkit classes when it is not installed.

        `repl.py` subclasses `Completer` at module level (`SlashCompleter`),
        so the fallback must bind subclassable classes here -- with the
        names bound to `None`, importing the package died with `TypeError:
        NoneType takes no arguments` before any command could run. Every
        real use of these names sits behind a `HAS_PROMPT_TOOLKIT` check
        (the inline REPL falls back to plain `input()` prompts), so a stub
        that is ever instantiated means a guard went missing and raises
        instead of misbehaving.
        """

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("prompt_toolkit is not installed")

    (PromptSession, Completer, Completion, FileHistory,
     KeyBindings) = (_MissingPromptToolkit,) * 5

    def patch_stdout(*args, **kwargs) -> Any:
        """No-op stand-in (the real one is only used with a live session)."""
        return contextlib.nullcontext()

try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:  # pragma: no cover - fallback path
    HAS_PYPERCLIP = False
    pyperclip = None

try:
    from pylatexenc.latex2text import LatexNodes2Text
    HAS_PYLATEXENC = True
except ImportError:  # pragma: no cover - fallback path
    HAS_PYLATEXENC = False
    LatexNodes2Text = None

# The Textual TUI (`--ui tui`) is optional: without it the inline
# prompt_toolkit REPL is used and the flag degrades with a note. Only the
# names re-exported for the package are bound here; `tui.py` imports the rest.
TEXTUAL_MIN_VERSION = "1.0"
try:
    import textual
    from textual.app import App as TextualApp  # noqa: F401
    from textual.widgets import Input as _TextualInput  # noqa: F401

    HAS_TEXTUAL = True
    TEXTUAL_VERSION = getattr(textual, "__version__", "?")
except ImportError:  # pragma: no cover - fallback path
    HAS_TEXTUAL = False
    TEXTUAL_VERSION = ""
    TextualApp = None

VERSION = "2.10"
SCHEMA_VERSION = 1

HOME = Path.home()
CONFIG_PATH = HOME / "etc/config_files/ollamarun.json"
DATA_DIR = HOME / "etc/data/ollamarun"
SESSIONS_DIR = DATA_DIR / "sessions"
INDEX_PATH = DATA_DIR / "index.json"
HISTORY_PATH = DATA_DIR / "history"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
LOG_PATH = HOME / "etc/logs/ollamarun.log"
#: Background session hosts (`session_host.py`): one Unix socket + metadata
#: file per session, under a per-user cache directory. The directory is
#: created 0700 and the socket 0600; nothing here is durable (removing it
#: only loses a live host, never a saved session).
HOST_DIR = HOME / ".cache/ollamarun/session-hosts"
HOST_LOG_PATH = HOME / "etc/logs/ollamarun-session-host.log"

# Model-facing definitions live outside the code: one JSON per tool and one
# file per prompt, shipped in `ollamarun_lib/defs/`. A same-named file in the
# override directory replaces the shipped one, so prompts and tool text can
# be edited without touching Python.
DEFS_DIR = Path(__file__).resolve().parent / "defs"
DEFS_OVERRIDE_DIR = HOME / "etc/config_files/ollamarun-defs"

DEFAULT_HOST = "127.0.0.1:11434"
DEFAULT_PORT = 11434
DEFAULT_TIMEOUT = 600
#: `ollamarun serve`: the API listener. LAN-accessible without
#: authentication is the locked contract; `--bind` narrows it.
DEFAULT_SERVE_BIND = "0.0.0.0"
DEFAULT_SERVE_PORT = 11440
#: `ollamarun serve` transport and input limits (`helpers.serve_settings`):
#: the body cap bounds the JSON request; the aggregate input caps bound what
#: resolving content parts (remote images, artifact inlines) may produce on
#: top of it. All are validated at startup — see the README matrix.
DEFAULT_SERVE_BODY_LIMIT = 32 * 1024 * 1024
DEFAULT_SERVE_REQUEST_TIMEOUT = 30.0
DEFAULT_SERVE_PING_INTERVAL = 15.0
DEFAULT_SERVE_SHUTDOWN_TIMEOUT = 10.0
DEFAULT_SERVE_MAX_MESSAGES = 16384
DEFAULT_SERVE_MAX_CONTENT_PARTS = 4096
DEFAULT_SERVE_MAX_MESSAGE_BYTES = 8 * 1024 * 1024
DEFAULT_SERVE_MAX_TEXT_BYTES = 32 * 1024 * 1024
DEFAULT_SERVE_MAX_IMAGES = 64
DEFAULT_SERVE_MAX_IMAGE_BYTES = 64 * 1024 * 1024
#: Artifacts (`artifacts.py`): one snapshot cap, one store quota and one
#: retention TTL. The per-file cap must fit base64-in-JSON under the
#: request body cap (32 MiB), which leaves ~24 MiB of binary headroom.
DEFAULT_ARTIFACT_MAX_BYTES = 16 * 1024 * 1024
DEFAULT_ARTIFACT_QUOTA_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_ARTIFACT_TTL = 7 * 86400

# `--remote NAME` names for the tailnet machines: alias -> tailscale hostname
# (resolved to its tailnet IP at run time; `remote_hosts` in the config file
# adds or overrides entries).
REMOTE_NODES: dict[str, str] = {
    "pi": "vinipi",
    "kiddo": "kiddo",
    "g3": "g3",
    "optiplex7070": "optiplex7070",
    "optiplex": "optiplex7070",
    "guri": "guri",
}
REMOTE_PROBE_TIMEOUT = 4.0

THEME = Theme({
    "info": "cyan",
    "ok": "green",
    "warn": "yellow",
    "err": "bold red",
    "key": "bold magenta",
    "you": "light_pink3",
    "ai": "dark_sea_green2",
    "answer": "bold dark_sea_green2",
    "math": "bright_cyan",
    "dim": "dim",
})

#: A background session host has no terminal of its own, but the client
#: attached to it does. The host is spawned with this variable set so the
#: shared consoles are built as if they were on a terminal; their output is
#: then routed to the attached client (see `session_host.OutputRouter`).
_FORCE_TERMINAL = os.environ.get("OLLAMARUN_FORCE_TERMINAL") == "1"
console = Console(theme=THEME, force_terminal=True if _FORCE_TERMINAL else None)
#: The shared stderr sink for messages that must not pollute stdout (the
#: `--raw` error line, the quiet unload warning); `rich` has no `stderr=`
#: print keyword -- a second Console bound to stderr is the supported way.
err_console = Console(stderr=True, theme=THEME,
                      force_terminal=True if _FORCE_TERMINAL else None)
rich.traceback.install()

#: One colour per built-in tool: the arguments panel of a call, the
#: "running NAME…" line and the "tool · NAME" result title all share it, so
#: a mixed run of calls (shell, files, speech, ...) stays readable at a
#: glance. A tool with no entry here gets a random pick from
#: TOOL_COLOR_PALETTE through `tool_color()`; config `tool_colors` overrides
#: entries by tool name.
TOOL_COLORS: dict[str, str] = {
    "run_shell": "cyan",
    "run_python": "yellow",
    "read_file": "deep_sky_blue1",
    "write_file": "orange1",
    "edit_file": "light_goldenrod2",
    "make_dir": "dark_sea_green2",
    "move_file": "cornflower_blue",
    "copy_file": "sky_blue2",
    "delete_file": "indian_red",
    "list_dir": "magenta",
    "search_sessions": "medium_turquoise",
    "list_models": "wheat1",
    "system_status": "steel_blue1",
    "context_status": "light_cyan3",
    "session_info": "medium_spring_green",
    "chem_search": "rosy_brown",
    "chem_container": "pale_turquoise4",
    "chem_locations": "plum2",
    "fetch_url": "spring_green2",
    "web_search": "hot_pink2",
    "speak": "salmon1",
    "list_voices": "turquoise2",
}

#: Fallback pool for tools without a fixed colour -- disjoint from
#: TOOL_COLORS so a random pick can never shadow a mapped tool.
TOOL_COLOR_PALETTE: tuple[str, ...] = (
    "gold1", "medium_purple1", "chartreuse2", "deep_pink2",
    "dodger_blue1", "dark_orange", "orchid", "khaki1",
)

_TOOL_COLOR_OVERRIDES: dict[str, str] = {}
_TOOL_COLOR_FALLBACKS: dict[str, str] = {}


def tool_color(name: str) -> str:
    """Panel colour for one tool: fixed when mapped, random (per run) otherwise.

    Unknown tools (invented names, future tools) pick a random colour from
    TOOL_COLOR_PALETTE -- memoised, so every panel of the same tool matches
    for the rest of the run, and drawn from colours not already in use so a
    random pick never collides with another tool while the palette lasts.
    """
    key = str(name or "tool")
    fixed = _TOOL_COLOR_OVERRIDES.get(key) or TOOL_COLORS.get(key)
    if fixed:
        return fixed
    if key not in _TOOL_COLOR_FALLBACKS:
        used = (set(TOOL_COLORS.values()) | set(_TOOL_COLOR_OVERRIDES.values())
                | set(_TOOL_COLOR_FALLBACKS.values()))
        free = [color for color in TOOL_COLOR_PALETTE if color not in used]
        _TOOL_COLOR_FALLBACKS[key] = random.choice(free or list(TOOL_COLOR_PALETTE))
    return _TOOL_COLOR_FALLBACKS[key]


def set_tool_colors(overrides: Optional[dict[str, Any]] = None) -> None:
    """Apply config `tool_colors` (tool name -> rich colour) over the built-ins.

    A named colour wins over TOOL_COLORS for that tool; a value rich cannot
    parse as a style is ignored with a warning (a typo must not crash panel
    rendering). Also re-rolls the random fallbacks, so calling it again gives
    unknown tools new colours (the CLI calls it once at startup, which is
    what makes the picks stable for a whole run).
    """
    _TOOL_COLOR_OVERRIDES.clear()
    for name, color in (overrides or {}).items():
        if not isinstance(color, str) or not color.strip():
            continue
        value = color.strip()
        try:
            Style.parse(value)
        except StyleSyntaxError:
            console.print(f"[warn]ignoring invalid tool colour for "
                          f"{str(name)!r}: {value!r}[/warn]")
            continue
        _TOOL_COLOR_OVERRIDES[str(name)] = value
    _TOOL_COLOR_FALLBACKS.clear()

DEFAULT_CONFIG: dict[str, Any] = {
    "host": None,
    "remote": None,
    "remote_hosts": {},
    "remote_ssh_user": None,
    "model": None,
    # Background session host (`session_host.py`). `background_sessions`
    # switches the inline REPL to its own detached process; closing the
    # terminal or double-Ctrl+C detaches without stopping it, and only
    # /bye, Ctrl+D, --close-session or --delete end it. `approval_timeout`
    # is how long a tool approval may wait for a client to reattach
    # (0 = wait forever); the event buffer/TTL size the replay window.
    "background_sessions": True,
    "background_approval_timeout": 0,
    "background_event_buffer": 20000,
    "background_event_ttl": 300,
    "system": None,
    "options": {},
    "render": "markdown",
    "math": True,
    "thinking": "collapsed",
    "ui": "inline",
    "show_stats": True,
    "auto_title": False,
    "auto_compact": "ask",
    "image_storage": "embed",
    "max_file_kb": 256,
    "pager": False,
    "think": None,
    "keepalive": None,
    "gpu_layers": None,
    "max_gpu_margin_mb": 1024,
    "max_gpu_overhead_mb": 0,
    "max_gpu_probes": 10,
    "max_cpu_overhead_mb": 256,
    "max_cpu_probes": 10,
    "unload_on_exit": True,
    "tps_footer": True,
    "timeout": DEFAULT_TIMEOUT,
    "max_render_chars": 12000,
    "tools": True,
    "tools_policy": "allowlist",
    "tools_shell_allow": [],
    "tools_timeout": 120,
    "tools_max_output": 4000,
    "tools_max_rounds": 8,
    "tool_colors": {},
    "tts_auto": False,
    "tts_voice": None,
    "tts_lang": None,
    "tts_remote": "tts-remote",
    "tts_tools": True,
    "chem_inventory": "chemInventoryAPI",
    "chem_tools": True,
    # Turn manager (`turns.py`): the serve front end's event buffers. The
    # HTTP listener keys land with the server phase.
    "serve_max_active_turns": 4,
    "serve_turn_event_buffer": 10000,
    "serve_turn_event_ttl": 86400,
    # `ollamarun serve`: the API listener (LAN, no authentication; the
    # trusted-network boundary is the bind address). `serve_agent` switches
    # the native agent routes; `serve_models` is an optional fnmatch
    # allowlist ([] = every installed model); `serve_workspace` jails the
    # API tool context to a directory (unset = the startup cwd).
    "serve_bind": DEFAULT_SERVE_BIND,
    "serve_port": DEFAULT_SERVE_PORT,
    "serve_agent": True,
    "serve_models": [],
    "serve_workspace": None,
    # Transport (`server.py`) and aggregate input limits (`artifacts.py`).
    # `helpers.serve_settings()` validates every serve key at startup: an
    # omitted (null) key uses the default, a provided value is validated,
    # and a zero only means what the key defines (see the README matrix).
    "serve_body_limit": DEFAULT_SERVE_BODY_LIMIT,
    "serve_request_timeout": DEFAULT_SERVE_REQUEST_TIMEOUT,
    "serve_sse_ping_interval": DEFAULT_SERVE_PING_INTERVAL,
    "serve_shutdown_timeout": DEFAULT_SERVE_SHUTDOWN_TIMEOUT,
    "serve_unload_on_exit": True,
    "serve_max_messages": DEFAULT_SERVE_MAX_MESSAGES,
    "serve_max_content_parts": DEFAULT_SERVE_MAX_CONTENT_PARTS,
    "serve_max_message_bytes": DEFAULT_SERVE_MAX_MESSAGE_BYTES,
    "serve_max_text_bytes": DEFAULT_SERVE_MAX_TEXT_BYTES,
    "serve_max_images": DEFAULT_SERVE_MAX_IMAGES,
    "serve_max_image_bytes": DEFAULT_SERVE_MAX_IMAGE_BYTES,
    # Native agent turns (`agent_api.py`): how long an interactive approval
    # may stay pending before it resolves as denied, and a wall-clock cap on
    # one turn (0 = no cap).
    "serve_approval_timeout": 300,
    "serve_turn_timeout": 0,
    # Artifacts (`artifacts.py`): uploads and tool outputs share one store.
    # `serve_artifact_root` overrides the directory; the per-file cap must
    # stay under the base64-in-JSON body ceiling; the TTL is in seconds
    # (0 = keep forever), the quota in bytes (0 = unlimited).
    "serve_artifact_root": str(ARTIFACTS_DIR),
    "serve_artifact_max_bytes": DEFAULT_ARTIFACT_MAX_BYTES,
    "serve_artifact_quota_bytes": DEFAULT_ARTIFACT_QUOTA_BYTES,
    "serve_artifact_ttl": DEFAULT_ARTIFACT_TTL,
    # Context fits (`fits.py` / `agent_api.py`): `serve_fit` gates the
    # estimate/probe surface ("off"/"estimate"/"probe"), the cap bounds
    # concurrent probing, the TTL is completed-job retention (0 = forever).
    "serve_fit": "probe",
    "serve_max_active_fits": 2,
    "serve_fit_ttl": 86400,
}

THINKING_MODES = {"collapsed", "expanded", "hidden"}
UI_MODES = {"inline", "tui", "auto"}
CTRL_D = 0x04
CTRL_Q = 0x11
CTRL_T = 0x14

EMBED_MODEL_HINTS = {"bert", "nomic-embed", "mxbai-embed", "all-minilm", "snowflake-arctic-embed"}
