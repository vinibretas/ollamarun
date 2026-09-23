"""Small helpers: timestamps, ids, coercion, config load, logging setup."""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .constants import CONFIG_PATH, DEFAULT_CONFIG, DEFAULT_HOST, DEFAULT_PORT, LOG_PATH, VERSION, console


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def short_stamp(value: Optional[str]) -> str:
    return (value or "")[:19].replace("T", " ")


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def parse_bool(value: Any, default: bool = False) -> bool:
    """Accepts real bools plus the strings ollama CLI users write."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return default


def normalize_host(value: Optional[str]) -> str:
    """'localhost:11434' / 'http://host' / '[::1]:11434' -> http://host:port."""
    value = (value or DEFAULT_HOST).strip()
    if "://" not in value:
        value = "http://" + value
    parsed = urllib.parse.urlsplit(value)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or DEFAULT_PORT
    scheme = parsed.scheme or "http"
    netloc = f"[{host}]" if ":" in host else host
    netloc += f":{port}"
    return f"{scheme}://{netloc}"


def parse_think(value: Any) -> Optional[Any]:
    """None when unset; True/False or low/medium/high otherwise."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    if text in {"low", "medium", "high"}:
        return text
    raise ValueError(f"invalid --think value {value!r} (true/false/low/medium/high)")


def coerce_value(text: str) -> Any:
    """0.7 -> float, true -> bool, otherwise the raw string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def canonical_model_name(name: str) -> str:
    """'qwen3' -> 'qwen3:latest' (tags only matter in the last path segment)."""
    name = (name or "").strip()
    if not name:
        return name
    if ":" not in name.rsplit("/", 1)[-1]:
        name += ":latest"
    return name


def human_bytes(num: Optional[float]) -> str:
    if not num:
        return "?"
    step = 1024.0
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < step:
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= step
    return f"{num:.1f} PiB"


def human_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def human_chars(count: int) -> str:
    if count < 1000:
        return f"{count} chars"
    if count < 1_000_000:
        return f"{count / 1000:.1f}k chars"
    return f"{count / 1_000_000:.1f}M chars"


def shortcut(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def naive_title(text: str) -> str:
    for line in text.splitlines():
        line = re.sub(r"^[#>*\-\s]+", "", line).strip()
        if line:
            return shortcut(line, 60)
    return "untitled"


def parse_format(value: Optional[str]) -> Optional[Any]:
    if not value:
        return None
    stripped = value.strip()
    if stripped in {"json", "JSON"}:
        return "json"
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except ValueError as exc:
            raise ValueError(f"--format is neither 'json' nor valid JSON: {exc}") from exc
    return stripped


def parse_options(pairs: Optional[list[str]]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--option expects key=value, got {item!r}")
        key, _, value = item.partition("=")
        options[key.strip()] = coerce_value(value.strip())
    return options


def load_config(path: Optional[Path] = None) -> dict[str, Any]:
    path = Path(path) if path else CONFIG_PATH
    config = dict(DEFAULT_CONFIG)
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                config.update(stored)
        except (ValueError, OSError) as exc:
            console.print(f"[warn]ignoring unreadable config {path}: {exc}[/warn]")
    else:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
    return config


def setup_logging(enabled: bool) -> None:
    if not enabled:
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(LOG_PATH), level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s")
        logging.debug("ollamarun %s starting", VERSION)
    except OSError:
        pass
