"""Small helpers: timestamps, ids, coercion, config load, logging setup, and
the `CancelToken` the UI threads use to stop a running turn.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .constants import (
    CONFIG_PATH,
    DEFAULT_CONFIG,
    DEFAULT_HOST,
    DEFAULT_PORT,
    LOG_PATH,
    REMOTE_NODES,
    VERSION,
    console,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def short_stamp(value: Optional[str]) -> str:
    return (value or "")[:19].replace("T", " ")


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


class CancelToken:
    """Thread-safe "stop this turn" flag shared between UI and worker threads.

    The streaming worker checks it between chunks; the UI thread sets it (and
    closes the in-flight HTTP response through `OllamaClient.abort_stream()`)
    so a cancel lands even while the worker is blocked waiting for the next
    chunk -- which can be a long time away while a model is being loaded.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    # `is_set` is the `threading.Event` spelling, so a plain Event works
    # anywhere a CancelToken is accepted.
    def is_set(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()


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


_TAILSCALE_CACHE: Optional[dict[str, dict[str, Any]]] = None
_TAILSCALE_SELF_IPS: set[str] = set()


def _tailscale_records() -> dict[str, dict[str, Any]]:
    """Tailnet peers keyed by hostname(lower), from `tailscale status --json`.

    Cached for the process; {} when tailscale is missing or the call fails.
    """
    global _TAILSCALE_CACHE
    if _TAILSCALE_CACHE is not None:
        return _TAILSCALE_CACHE
    records: dict[str, dict[str, Any]] = {}
    if shutil.which("tailscale"):
        try:
            raw = subprocess.check_output(
                ["tailscale", "status", "--json"], text=True, timeout=10,
                stderr=subprocess.DEVNULL)
            data = json.loads(raw)
            peers = list((data.get("Peer") or {}).values())
            self_peer = data.get("Self") or {}
            if self_peer:
                peers.append(self_peer)
                for ip in self_peer.get("TailscaleIPs") or []:
                    _TAILSCALE_SELF_IPS.add(str(ip).lower())
            for peer in peers:
                name = (peer.get("HostName") or "").lower()
                if name:
                    records[name] = peer
        except (OSError, ValueError, subprocess.SubprocessError):
            records = {}
    _TAILSCALE_CACHE = records
    return records


def tailscale_map() -> dict[str, str]:
    """hostname(lower) -> first Tailscale IP, from `tailscale status --json`."""
    out: dict[str, str] = {}
    for name, peer in _tailscale_records().items():
        ips = peer.get("TailscaleIPs") or []
        if ips:
            out[name] = ips[0]
    return out


def tailscale_peer_state(name: str) -> str:
    """'online' / 'offline, last seen …' for a tailnet node; '' when unknown."""
    peer = _tailscale_records().get((name or "").lower())
    if peer is None:
        return ""
    if peer.get("Online"):
        return "online"
    seen = short_stamp(peer.get("LastSeen"))[:16]
    return f"offline, last seen {seen}" if seen else "offline"


def _remote_parts(value: str, config: Optional[dict[str, Any]]) -> tuple[str, Optional[int]]:
    """(node, port) for a `--remote` value, before tailnet-IP resolution.

    Known names/aliases map through `config["remote_hosts"]` then
    REMOTE_NODES; anything else is taken as a hostname. The port is only set
    when the value carries one explicitly.
    """
    config = config or {}
    raw = (value or "").strip()
    if not raw:
        raise ValueError("--remote needs a machine name (pi, kiddo, g3, optiplex7070, guri)")
    if "://" in raw:
        parsed = urllib.parse.urlsplit(raw)
        return (parsed.hostname or raw).lower(), parsed.port
    parsed = urllib.parse.urlsplit("http://" + raw)
    name = (parsed.hostname or raw).lower()
    override = (config.get("remote_hosts") or {}).get(name)
    if override:
        seen = urllib.parse.urlsplit("http://" + str(override))
        node = (seen.hostname or str(override)).lower()
        return node, parsed.port or seen.port
    return str(REMOTE_NODES.get(name) or name).lower(), parsed.port


def remote_node(value: str, config: Optional[dict[str, Any]] = None) -> str:
    """Machine (tailnet hostname or literal address) behind a `--remote` value."""
    return _remote_parts(value, config)[0]


def resolve_remote(value: str, config: Optional[dict[str, Any]] = None,
                   ts_map: Optional[dict[str, str]] = None) -> str:
    """`--remote` value -> base URL, e.g. 'kiddo' -> 'http://100.64.0.2:11434'.

    Known names map through `config["remote_hosts"]` then REMOTE_NODES and are
    resolved to their tailnet IP through `tailscale status --json` when
    available (falling back to the name itself, which MagicDNS resolves);
    every other value goes through `normalize_host` exactly like `--host`.
    `ts_map` is injectable for tests.
    """
    raw = (value or "").strip()
    if not raw:
        raise ValueError("--remote needs a machine name (pi, kiddo, g3, optiplex7070, guri)")
    if "://" in raw:
        return normalize_host(raw)
    node, port = _remote_parts(value, config)
    if ts_map is None:
        ts_map = tailscale_map()
    node = ts_map.get(node) or node
    if not port:
        port = DEFAULT_PORT
    bracketed = f"[{node}]" if ":" in node else node
    return normalize_host(f"{bracketed}:{port}")


def is_local_host(base: str) -> bool:
    """True when a base URL points at this machine's default local Ollama."""
    try:
        hostname = urllib.parse.urlsplit(normalize_host(base)).hostname or ""
    except ValueError:
        return False
    return hostname.lower() in {"127.0.0.1", "localhost", "::1"}


_SELF_ADDRESSES: Optional[set[str]] = None


def _self_addresses() -> set[str]:
    """Hostnames and addresses that mean 'this machine' (cached)."""
    global _SELF_ADDRESSES
    if _SELF_ADDRESSES is not None:
        return _SELF_ADDRESSES
    addresses = {"localhost", "127.0.0.1", "::1"}
    try:
        name = socket.gethostname().lower()
        addresses.add(name)
        addresses.add(name.split(".")[0])
        for info in socket.getaddrinfo(name, None):
            addresses.add(str(info[4][0]).lower())
    except OSError:
        pass
    _tailscale_records()
    addresses.update(_TAILSCALE_SELF_IPS)
    _SELF_ADDRESSES = addresses
    return addresses


def host_is_self(value: str) -> bool:
    """True when a host/base URL points at an address or name of this machine.

    Broader than `is_local_host`: also matches the machine's hostname, its
    resolved addresses and its own tailnet IP (ollama may be bound to the
    tailscale address instead of localhost, as on this fleet).
    """
    try:
        host = (urllib.parse.urlsplit(normalize_host(value)).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    known = _self_addresses()
    if host in known:
        return True
    try:
        resolved = {str(info[4][0]).lower() for info in socket.getaddrinfo(host, None)}
    except OSError:
        return False
    return bool(resolved & known)


SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
            "-o", "ConnectionAttempts=1"]

# Read OLLAMA_MODELS from the ollama unit, then df the nearest accessible
# ancestor (the model directory is often 750 ollama:ollama and untraversable
# for the login user); prints "total_kb free_kb mount".
SSH_DISK_COMMAND = (
    "m=$(systemctl show ollama -p Environment 2>/dev/null "
    "| tr ' ' '\\n' | sed -n 's/^OLLAMA_MODELS=//p' | tail -n 1); "
    "[ -n \"$m\" ] || m=/usr/share/ollama/.ollama/models; "
    "p=$m; while [ ! -e \"$p\" ] && [ \"$p\" != / ]; do p=$(dirname \"$p\"); done; "
    "df -Pk \"$p\" | tail -n 1 | awk '{print $2, $4, $6}'"
)


def ssh_free_disk(target: str, *, user: Optional[str] = None, timeout: float = 8.0,
                  runner: Any = None) -> Optional[tuple[int, int, str]]:
    """(free_bytes, total_bytes, mount) on a remote machine over SSH, or None.

    Best-effort: a missing/unreachable host, no key or a bad response all
    return None; the caller degrades to "unknown". `runner` is injectable for
    tests (defaults to `subprocess.run`).
    """
    if not target or shutil.which("ssh") is None:
        return None
    dest = f"{user}@{target}" if user else target
    run = runner or subprocess.run
    try:
        proc = run(["ssh", *SSH_OPTS, dest, SSH_DISK_COMMAND],
                   capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    parts = lines[-1].split()
    if len(parts) < 3:
        return None
    try:
        total_kb, free_kb = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return free_kb * 1024, total_kb * 1024, parts[2]


GPU_ENV_MARKER = "---ENV---"

# One ssh round trip for the context-fit probe: per-GPU total/free VRAM (MiB)
# first, then the ollama unit's environment (LLAMA_ARG_FIT_TARGET and
# OLLAMA_NUM_PARALLEL/OLLAMA_KV_CACHE_TYPE feed the estimate).
SSH_GPU_ENV_COMMAND = (
    "nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits "
    f"2>/dev/null; echo {GPU_ENV_MARKER}; "
    "systemctl show ollama -p Environment 2>/dev/null"
)


def parse_gpu_memory(text: str) -> list[tuple[int, int]]:
    """[(total_mb, free_mb)] from nvidia-smi CSV lines; junk lines are skipped."""
    devices: list[tuple[int, int]] = []
    for line in (text or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            devices.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return devices


def parse_service_env(text: str) -> dict[str, str]:
    """KEY=VALUE pairs from `systemctl show ollama -p Environment` output."""
    env: dict[str, str] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("Environment="):
            line = line[len("Environment="):]
        for match in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|\S+)', line):
            env[match.group(1)] = match.group(2).strip('"')
    return env


def local_gpu_probe(runner: Any = None) -> tuple[list[tuple[int, int]], dict[str, str]]:
    """(devices, ollama service env) on this machine; both empty when unavailable."""
    run = runner or subprocess.run
    devices: list[tuple[int, int]] = []
    env: dict[str, str] = {}
    if shutil.which("nvidia-smi"):
        try:
            proc = run(["nvidia-smi", "--query-gpu=memory.total,memory.free",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=8)
            if proc.returncode == 0:
                devices = parse_gpu_memory(proc.stdout)
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        proc = run(["systemctl", "show", "ollama", "-p", "Environment"],
                   capture_output=True, text=True, timeout=8)
        if proc.returncode == 0:
            env = parse_service_env(proc.stdout)
    except (OSError, subprocess.SubprocessError):
        pass
    return devices, env


def ssh_gpu_probe(target: str, *, user: Optional[str] = None, timeout: float = 8.0,
                  runner: Any = None) -> Optional[tuple[list[tuple[int, int]], dict[str, str]]]:
    """(devices, ollama service env) on a remote machine over SSH, or None on failure.

    Best-effort exactly like `ssh_free_disk`: a missing/unreachable host, no
    key, a machine without nvidia-smi or a bad response all return None.
    """
    if not target or shutil.which("ssh") is None:
        return None
    dest = f"{user}@{target}" if user else target
    run = runner or subprocess.run
    try:
        proc = run(["ssh", *SSH_OPTS, dest, SSH_GPU_ENV_COMMAND],
                   capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout or ""
    gpu_text, _, env_text = text.partition(GPU_ENV_MARKER)
    devices = parse_gpu_memory(gpu_text)
    if not devices:
        return None
    return devices, parse_service_env(env_text)


def _first_existing_path(path: str) -> str:
    current = path or "/"
    while current != "/" and not os.path.exists(current):
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break
        current = parent
    return current or "/"


def local_free_disk(candidates: Optional[list[str]] = None) -> Optional[tuple[int, int, str]]:
    """(free_bytes, total_bytes, path) for the filesystem holding the models.

    Probes OLLAMA_MODELS, the systemd default store and ~/.ollama/models,
    climbing to the nearest accessible ancestor of each (the model store
    itself may be unreadable when ollama runs as another user), and falls
    back to `/`. `candidates` is injectable for tests.
    """
    if candidates is None:
        candidates = []
        if os.environ.get("OLLAMA_MODELS"):
            candidates.append(os.environ["OLLAMA_MODELS"])
        candidates += ["/usr/share/ollama/.ollama/models",
                       str(Path.home() / ".ollama/models"), "/"]
    for candidate in candidates:
        probe = _first_existing_path(candidate)
        try:
            usage = shutil.disk_usage(probe)
        except OSError:
            continue
        return usage.free, usage.total, probe
    return None


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


def human_tokens(value: Optional[int]) -> str:
    """A context token count: raw below 1k, 3-decimal k above.

    Three decimals make a single streamed token visible (`12.345k` ->
    `12.346k`) -- the point of the live counter -- while the fixed width
    keeps the footer from jittering; a 1-decimal `12.3k` only moves once
    every 100 tokens.
    """
    if value is None:
        return "?"
    return str(value) if value < 1000 else f"{value / 1000:.3f}k"


def format_context(used: Optional[int], total: Optional[int]) -> str:
    """The `ctx 12.345k/32.768k (38.123%)` piece of the footer and status bar.

    Missing numbers degrade like the status bar always did: no usage yet
    reads `ctx ?` (or `ctx 0/total`), an unknown window drops the share.
    """
    if not used:
        return "ctx ?" if not total else f"ctx 0/{human_tokens(total)}"
    if not total:
        return f"ctx {human_tokens(used)}"
    return (f"ctx {human_tokens(used)}/{human_tokens(total)} "
            f"({used * 100 / total:.3f}%)")


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
