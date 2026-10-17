"""Shared imports, paths and constants used across the ollamarun_lib package.

Every other module in this package imports from here rather than repeating
these definitions -- this is the module every other stage of the pipeline
(see the top-level package docstring) sits on top of.
"""
from __future__ import annotations

import argparse
import base64
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
    (PromptSession, Completer, Completion, FileHistory, KeyBindings,
     patch_stdout) = (None,) * 6

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

VERSION = "1.37"
SCHEMA_VERSION = 1

HOME = Path.home()
CONFIG_PATH = HOME / "etc/config_files/ollamarun.json"
DATA_DIR = HOME / "etc/data/ollamarun"
SESSIONS_DIR = DATA_DIR / "sessions"
INDEX_PATH = DATA_DIR / "index.json"
HISTORY_PATH = DATA_DIR / "history"
LOG_PATH = HOME / "etc/logs/ollamarun.log"

# Model-facing definitions live outside the code: one JSON per tool and one
# file per prompt, shipped in `ollamarun_lib/defs/`. A same-named file in the
# override directory replaces the shipped one, so prompts and tool text can
# be edited without touching Python.
DEFS_DIR = Path(__file__).resolve().parent / "defs"
DEFS_OVERRIDE_DIR = HOME / "etc/config_files/ollamarun-defs"

DEFAULT_HOST = "127.0.0.1:11434"
DEFAULT_PORT = 11434
DEFAULT_TIMEOUT = 600

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

console = Console(theme=THEME)
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
    "max_gpu_overhead_mb": 256,
    "max_gpu_probes": 8,
    "max_cpu_overhead_mb": 256,
    "max_cpu_probes": 8,
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
}

THINKING_MODES = {"collapsed", "expanded", "hidden"}
UI_MODES = {"inline", "tui", "auto"}
CTRL_T = 0x14

EMBED_MODEL_HINTS = {"bert", "nomic-embed", "mxbai-embed", "all-minilm", "snowflake-arctic-embed"}
