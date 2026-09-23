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
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    HAS_PROMPT_TOOLKIT = True
except ImportError:  # pragma: no cover - fallback path
    HAS_PROMPT_TOOLKIT = False
    PromptSession = Completer = Completion = FileHistory = KeyBindings = None

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

VERSION = "1.4"
SCHEMA_VERSION = 1

HOME = Path.home()
CONFIG_PATH = HOME / "etc/config_files/ollamarun.json"
DATA_DIR = HOME / "etc/data/ollamarun"
SESSIONS_DIR = DATA_DIR / "sessions"
INDEX_PATH = DATA_DIR / "index.json"
HISTORY_PATH = DATA_DIR / "history"
LOG_PATH = HOME / "etc/logs/ollamarun.log"

DEFAULT_HOST = "127.0.0.1:11434"
DEFAULT_PORT = 11434
DEFAULT_TIMEOUT = 600

THEME = Theme({
    "info": "cyan",
    "ok": "green",
    "warn": "yellow",
    "err": "bold red",
    "key": "bold magenta",
    "you": "bold cyan",
    "ai": "bold green",
    "math": "bright_cyan",
    "dim": "dim",
})

console = Console(theme=THEME)
rich.traceback.install()

DEFAULT_CONFIG: dict[str, Any] = {
    "host": None,
    "model": None,
    "system": None,
    "options": {},
    "render": "markdown",
    "math": True,
    "thinking": "collapsed",
    "show_stats": False,
    "auto_title": False,
    "auto_compact": "ask",
    "image_storage": "embed",
    "max_file_kb": 256,
    "pager": False,
    "think": None,
    "keepalive": None,
    "unload_on_exit": True,
    "tps_footer": True,
    "timeout": DEFAULT_TIMEOUT,
    "max_render_chars": 12000,
}

THINKING_MODES = {"collapsed", "expanded", "hidden"}
CTRL_T = 0x14

EMBED_MODEL_HINTS = {"bert", "nomic-embed", "mxbai-embed", "all-minilm", "snowflake-arctic-embed"}
