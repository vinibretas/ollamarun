"""Background session host: one long-lived process owned by an inline session.

The inline REPL normally lives in the terminal. This module flips that: the
REPL (session, commands, turns, approvals, pulls, compaction) runs in a
detached process and the terminal is a thin client over a Unix socket. The
host is started on demand, survives the terminal closing, and only ends on an
explicit close (`/bye`, Ctrl+D, `--close-session`, `--delete`) or SIGTERM.

The pieces:

* `HostSpec` -- everything the child needs to reconstruct the session: the
  resolved `Session`, the parent's `argparse` namespace, the resolved Ollama
  base, the launch directory, pending `--image` payloads and the explicit
  launch overrides a resumed session should apply.
* `SessionHostServer` -- binds the per-session socket, owns the `Repl`, turns
  its `console` output into ANSI `output` messages, runs turns through
  `TurnManager` (so a reattaching client can replay them), parks approvals in
  `pending_question`, and rekeys the socket when `/clear`, `/new`, `/branch`
  or `/load` changes the session id.
* `HostUi` -- the `Ui` seam the Repl talks to: prompts, confirmations,
  approvals, the editor and clipboard are forwarded to the attached client.
* `LineConnection` -- newline-delimited JSON framing shared with the client.
* `ensure_host` / `spawn_host` / `stop_host` / `host_is_running` -- discovery
  and lifecycle helpers used by `cli.py`.
* `daemon_main` -- the hidden `__session-host` entry point.

The client side lives in `host_client.py`. Nothing here imports the client.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:  # POSIX start lock; Unix sockets already require POSIX, kept optional
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from .client import OllamaClient
from .constants import (
    DEFAULT_TIMEOUT,
    HOST_DIR,
    HOST_LOG_PATH,
    THINKING_MODES,
    VERSION,
    console,
    err_console,
    set_tool_colors,
)
from .events import (
    MESSAGE_DELTA,
    MESSAGE_USER,
    TERMINAL_EVENTS,
    RuntimeEvent,
)
from .helpers import CancelToken, load_config, now_iso, setup_logging
from .repl import Repl
from .session import Session, SessionStore
from .turns import ManagerClosed, SessionBusy, TurnManager, TurnResult
from .ui import Ui

logger = logging.getLogger("ollamarun.host")

#: Wire protocol version. Bump on any incompatible message-shape change.
HOST_PROTOCOL = 1
#: Additive capabilities advertised in `hello`; a client checks these before
#: using an op so a newer terminal can attach to an older live host.
HOST_FEATURES = ("cancel", "thinking")
#: One JSON message may be this large (a full session snapshot with images
#: travels in the launch spec, not over the live socket; this is slack).
MAX_LINE = 64 * 1024 * 1024
#: How much command output is replayed to a client that attaches late.
OUTPUT_BUFFER_BYTES = 1_000_000


class SessionHostError(RuntimeError):
    """The host could not be started or answered with an error."""


class SessionHostUnavailable(SessionHostError):
    """No host could be spawned/connected; the caller may fall back."""


class HostClosed(RuntimeError):
    """The host is shutting down under a pending UI request."""


# ── paths ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HostPaths:
    session_id: str
    root: Path
    socket: Path
    meta: Path
    lock: Path
    spec: Path


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return cleaned or "session"


def host_paths(session_id: str, root: Optional[Path] = None) -> HostPaths:
    """Deterministic socket/metadata paths for one session id."""
    root = Path(root or HOST_DIR)
    safe = _safe_name(session_id)
    socket_path = root / f"{safe}.sock"
    # Unix socket paths are limited to ~107 bytes. A very deep $HOME falls
    # back to a short per-user runtime directory; the name is deterministic
    # so the client computes the same path.
    if len(str(socket_path)) > 100:
        import hashlib
        import tempfile

        digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:16]
        socket_path = (Path(tempfile.gettempdir())
                       / f"ollamarun-{os.getuid()}" / f"{digest}.sock")
    return HostPaths(session_id=session_id, root=root, socket=socket_path,
                     meta=root / f"{safe}.json",
                     lock=root / f"{safe}.lock",
                     spec=root / f"{safe}.spec.json")


# ── launch spec ────────────────────────────────────────────────────────────


@dataclass
class HostSpec:
    """What the child needs to rebuild the parent's interactive session."""

    protocol: int
    version: str
    session: dict[str, Any]
    args: dict[str, Any]
    base: str
    insecure: bool
    cwd: str
    session_id: str
    pending_images: list[dict[str, Any]] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)
    config_path: Optional[str] = None
    debug: bool = False
    #: Test/embedding override for the socket directory; the CLI leaves it
    #: unset so the default `HOST_DIR` is used.
    host_root: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol, "version": self.version,
            "session": self.session, "args": self.args, "base": self.base,
            "insecure": self.insecure, "cwd": self.cwd,
            "session_id": self.session_id,
            "pending_images": self.pending_images,
            "overrides": self.overrides, "config_path": self.config_path,
            "debug": self.debug, "host_root": self.host_root,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HostSpec":
        return cls(
            protocol=int(data.get("protocol") or 0),
            version=str(data.get("version") or ""),
            session=dict(data.get("session") or {}),
            args=dict(data.get("args") or {}),
            base=str(data.get("base") or ""),
            insecure=bool(data.get("insecure")),
            cwd=str(data.get("cwd") or os.getcwd()),
            session_id=str(data.get("session_id") or ""),
            pending_images=list(data.get("pending_images") or []),
            overrides=dict(data.get("overrides") or {}),
            config_path=(str(data["config_path"])
                         if data.get("config_path") else None),
            debug=bool(data.get("debug")),
            host_root=(str(data["host_root"])
                       if data.get("host_root") else None),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        text = json.dumps(self.to_dict(), ensure_ascii=False, default=str)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    @classmethod
    def read(cls, path: Path) -> "HostSpec":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ── line framing ───────────────────────────────────────────────────────────


def _peer_credentials(sock: socket.socket) -> tuple[Optional[int], Optional[int]]:
    """(pid, uid) of the peer on Linux; (None, None) when unavailable."""
    try:
        raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                              struct.calcsize("3i"))
        pid, uid, _gid = struct.unpack("3i", raw)
        return pid, uid
    except (AttributeError, OSError, struct.error):
        return None, None


class LineConnection:
    """One newline-delimited JSON socket, safe for one reader and many writers."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.reader = sock.makefile("rb")
        self.writer = sock.makefile("wb")
        self.lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.closed = False
        # Bound a write to a wedged client without touching the read side:
        # SO_SNDTIMEO is write-only (SO_RCVTIMEO would disconnect idle, healthy
        # terminals). A failed send marks the connection dead.
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO,
                                 struct.pack("ll", 5, 0))
        except (AttributeError, OSError, struct.error):
            pass
        self.pid, self.uid = _peer_credentials(sock)

    @classmethod
    def connect(cls, path: Path, timeout: float = 10.0) -> "LineConnection":
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(str(path))
        except OSError:
            sock.close()
            raise
        sock.settimeout(None)
        return cls(sock)

    def send(self, payload: dict[str, Any]) -> bool:
        try:
            data = (json.dumps(payload, ensure_ascii=False, default=str)
                    + "\n").encode("utf-8")
        except (TypeError, ValueError):
            logger.debug("host message is not JSON-ready", exc_info=True)
            return False
        if len(data) > MAX_LINE:
            logger.warning("host message too large (%d bytes)", len(data))
            return False
        with self.lock:
            if self.closed:
                return False
            try:
                self.writer.write(data)
                self.writer.flush()
                return True
            except (OSError, ValueError):
                self.closed = True
                return False

    def recv(self) -> Optional[dict[str, Any]]:
        try:
            line = self.reader.readline(MAX_LINE + 1)
        except (OSError, ValueError):
            return None
        if not line:
            return None
        if len(line) > MAX_LINE:
            logger.warning("oversized host line; closing")
            return None
        try:
            data = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def close(self) -> None:
        # Never wait for the write lock here: a stuck send must not hold up a
        # takeover or shutdown. Shutting the socket down below wakes it.
        with self._state_lock:
            if self.closed:
                return
            self.closed = True
        # Wake a reader blocked in recv() before touching the buffered file
        # objects: closing a makefile from another thread waits for the
        # reader's lock, and without this a takeover of a connection parked
        # in recv() would deadlock the host.
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        for handle in (self.reader, self.writer):
            try:
                handle.close()
            except (OSError, ValueError):
                pass
        try:
            self.sock.close()
        except OSError:
            pass


# ── discovery / lifecycle helpers ──────────────────────────────────────────


def _read_meta(paths: HostPaths) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(paths.meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_meta(paths: HostPaths, data: dict[str, Any]) -> None:
    try:
        paths.root.mkdir(parents=True, exist_ok=True)
        tmp = paths.meta.with_name(f"{paths.meta.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str) + "\n",
                       encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, paths.meta)
    except OSError:
        logger.debug("could not write host metadata", exc_info=True)


def _wake_pid(pid: Optional[int]) -> None:
    """Interrupt another terminal process; never signal our own test process."""
    if os.environ.get("OLLAMARUN_HOST_NO_SIGNAL"):
        return
    if not pid or int(pid) == os.getpid():
        return
    try:
        os.kill(int(pid), signal.SIGINT)
    except (OSError, ValueError, TypeError):
        pass


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def ping_host(session_id: str, root: Optional[Path] = None,
              timeout: float = 0.6) -> bool:
    """True when a host for `session_id` answers a ping (does not attach)."""
    paths = host_paths(session_id, root)
    if not paths.socket.exists():
        return False
    conn: Optional[LineConnection] = None
    try:
        conn = LineConnection.connect(paths.socket, timeout=timeout)
        if not conn.send({"op": "ping"}):
            return False
        conn.sock.settimeout(timeout)
        reply = conn.recv()
        return bool(reply and reply.get("op") == "pong")
    except (OSError, ValueError):
        return False
    finally:
        if conn is not None:
            conn.close()


def host_is_running(session_id: str, root: Optional[Path] = None) -> bool:
    return ping_host(session_id, root)


def host_status(session_id: str, root: Optional[Path] = None
                ) -> Optional[dict[str, Any]]:
    paths = host_paths(session_id, root)
    meta = _read_meta(paths)
    if meta is None or not ping_host(session_id, root):
        return None
    return meta


def stop_host(session_id: str, root: Optional[Path] = None,
              timeout: float = 20.0) -> bool:
    """Explicitly close the host (the non-interactive `/bye`)."""
    paths = host_paths(session_id, root)
    if not paths.socket.exists():
        return False
    conn: Optional[LineConnection] = None
    try:
        conn = LineConnection.connect(paths.socket, timeout=5.0)
        if not conn.send({"op": "close"}):
            return False
        conn.sock.settimeout(timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            reply = conn.recv()
            if reply is None:
                break
            if reply.get("op") == "shutdown":
                break
        # The reply is only an acknowledgement: wait for the listener and
        # metadata to disappear, which happens after the Repl has saved and
        # unloaded. This keeps --delete from trashing a session the host is
        # still writing.
        while time.monotonic() < deadline:
            if not paths.socket.exists():
                return True
            time.sleep(0.05)
        return True
    except (OSError, ValueError):
        return False
    finally:
        if conn is not None:
            conn.close()


def _cleanup_stale(paths: HostPaths) -> None:
    if ping_host(paths.session_id, paths.root):
        return
    for path in (paths.socket, paths.meta, paths.spec):
        try:
            path.unlink()
        except OSError:
            pass


@contextlib.contextmanager
def _start_lock(paths: HostPaths):
    """Serialize two terminals racing to start the same session host."""
    paths.root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(paths.root, 0o700)
    except OSError:
        pass
    handle = open(paths.lock, "a+")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def _entry_script() -> str:
    candidate = Path(sys.argv[0]).resolve()
    if candidate.name == "ollamarun" and candidate.exists():
        return str(candidate)
    return str(Path(__file__).resolve().parents[1] / "ollamarun")


def _log_tail(lines: int = 20) -> str:
    try:
        text = HOST_LOG_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def spawn_host(spec: HostSpec, paths: HostPaths) -> subprocess.Popen:
    """Write the spec and start the detached child. Raises on failure."""
    spec.write(paths.spec)
    HOST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        log = open(HOST_LOG_PATH, "a", encoding="utf-8", buffering=1)
    except OSError as exc:
        raise SessionHostUnavailable(f"cannot open {HOST_LOG_PATH}: {exc}") from exc
    env = os.environ.copy()
    env["OLLAMARUN_FORCE_TERMINAL"] = "1"
    env["OLLAMARUN_SESSION_HOST"] = "1"
    command = [sys.executable, _entry_script(), "__session-host",
               "--spec", str(paths.spec)]
    try:
        return subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            cwd=spec.cwd or None, env=env, close_fds=True,
            start_new_session=True)
    except OSError as exc:
        log.close()
        raise SessionHostUnavailable(f"cannot start the session host: {exc}") from exc


def ensure_host(spec: HostSpec, paths: HostPaths,
                timeout: float = 30.0) -> tuple[Optional[subprocess.Popen], bool]:
    """Return (spawned process or None, spawned) with a live host answering.

    The caller still connects and sends `attach`; this only guarantees the
    socket is up. A running host is reused, never restarted.
    """
    if ping_host(spec.session_id, paths.root):
        return None, False
    with _start_lock(paths):
        if ping_host(spec.session_id, paths.root):
            return None, False
        _cleanup_stale(paths)
        popen = spawn_host(spec, paths)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if ping_host(spec.session_id, paths.root, timeout=0.3):
                return popen, True
            if popen.poll() is not None:
                tail = _log_tail()
                try:
                    paths.spec.unlink()
                except OSError:
                    pass
                raise SessionHostUnavailable(
                    "the session host exited during startup"
                    + (f":\n{tail}" if tail else ""))
            time.sleep(0.05)
        try:
            popen.terminate()
            popen.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        tail = _log_tail()
        try:
            paths.spec.unlink()
        except OSError:
            pass
        raise SessionHostUnavailable(
            "the session host did not answer"
            + (f":\n{tail}" if tail else ""))


# ── the host UI seam ───────────────────────────────────────────────────────


class HostUi(Ui):
    """The `Ui` implementation a background host uses.

    Every question becomes a `question` message to the attached client and
    blocks the Repl thread until it answers. No attached client simply means
    the question waits (the turn is parked), which is the contract for a
    detached approval. `interactive` is always true: a client can attach.
    """

    interactive = True

    def __init__(self, host: "SessionHostServer"):
        self.host = host

    def read_line(self, prompt: str) -> str:
        return self.host.read_line(prompt)

    def ask(self, question: str, *, choices: Optional[list[str]] = None,
            default: Optional[str] = None) -> Optional[str]:
        answer = self.host.request_question(
            "ask", {"question": question,
                    "choices": list(choices) if choices else None,
                    "default": default})
        return answer.get("value")

    def confirm(self, question: str, *, default: bool = False) -> bool:
        answer = self.host.request_question(
            "confirm", {"question": question, "default": bool(default)})
        return bool(answer.get("value"))

    def editor(self, initial: str = "") -> Optional[str]:
        answer = self.host.request_question(
            "editor", {"initial": initial}, interrupt="cancel")
        if answer.get("cancelled"):
            return None
        return answer.get("text")

    def approval(self, suggestions: list[Any], *, name: str = "",
                 detail: str = "", reason: str = "") -> tuple[str, Any]:
        self.host.mark_waiting_approval(True)
        try:
            answer = self.host.request_question(
                "approval",
                {"name": name, "detail": detail, "reason": reason,
                 "suggestions": [rule.to_dict() if hasattr(rule, "to_dict")
                                 else rule for rule in suggestions]},
                interrupt="detach")
        finally:
            self.host.mark_waiting_approval(False)
        action = str(answer.get("action") or "deny")
        from .tools import ToolRule

        rule = ToolRule.from_dict(answer.get("rule"))
        return action, rule

    def copy(self, text: str) -> tuple[bool, str]:
        answer = self.host.request_question("copy", {"text": text},
                                            interrupt="cancel")
        return bool(answer.get("ok")), str(answer.get("message") or "")

    def notify(self, event: str, **data: Any) -> None:
        return


# ── console routing ────────────────────────────────────────────────────────


class OutputRouter:
    """A file-like object that sends rich's ANSI output to the attached client."""

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, host: "SessionHostServer"):
        self.host = host

    def write(self, text: str) -> int:
        self.host.write_output(text)
        return len(text)

    def flush(self) -> None:
        return

    def isatty(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:  # pragma: no cover - rich never needs it here
        raise OSError("the session host output has no fileno")


class _Coalescer:
    """Merge per-token deltas into ~20/s events before the manager buffers them."""

    def __init__(self, sink: Callable[[RuntimeEvent], None], *,
                 interval: float = 0.05, max_chars: int = 128):
        self.sink = sink
        self.interval = interval
        self.max_chars = max_chars
        self.buffers: dict[str, list[str]] = {}
        self.counts: dict[str, int] = {}
        self.last = 0.0

    def __call__(self, event: RuntimeEvent) -> None:
        if event.type != MESSAGE_DELTA:
            self.flush()
            self.sink(event)
            return
        channel = str(event.data.get("channel") or "content")
        text = str(event.data.get("text") or "")
        if not text:
            return
        self.buffers.setdefault(channel, []).append(text)
        self.counts[channel] = self.counts.get(channel, 0) + 1
        now = time.monotonic()
        size = sum(len(part) for part in self.buffers[channel])
        if now - self.last >= self.interval or size >= self.max_chars:
            self.flush()

    def flush(self) -> None:
        for channel, parts in list(self.buffers.items()):
            text = "".join(parts)
            count = self.counts.get(channel, 1)
            self.buffers[channel] = []
            self.counts[channel] = 0
            if text:
                self.sink(RuntimeEvent(MESSAGE_DELTA, {
                    "channel": channel, "text": text, "tokens": count}))
        self.last = time.monotonic()


# ── the server ─────────────────────────────────────────────────────────────


class _ConnectionHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:  # pragma: no cover - threading glue
        self.server.host.handle_connection(self.request)  # type: ignore[attr-defined]


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


class SessionHostServer:
    """The detached session process: one Repl, one socket, one active client."""

    def __init__(self, spec: HostSpec, paths: HostPaths, *,
                 client_factory: Optional[Callable[..., Any]] = None,
                 store: Optional[SessionStore] = None,
                 console_obj: Any = None,
                 err_console_obj: Any = None,
                 clock: Callable[[], float] = time.monotonic):
        self.spec = spec
        self.paths = paths
        self.session_id = spec.session_id or spec.session.get("id") or ""
        self.session = Session.from_dict(spec.session)
        self.clock = clock
        self.config: dict[str, Any] = load_config(
            Path(spec.config_path) if spec.config_path else None)
        set_tool_colors(self.config.get("tool_colors"))
        self._client_factory = client_factory or (
            lambda base, insecure, timeout: OllamaClient(
                base, insecure=insecure, timeout=timeout))
        self.store = store or SessionStore()
        self.console = console_obj or console
        self.err_console = err_console_obj or err_console
        self.output = OutputRouter(self)
        raw_buffer = self.config.get("background_event_buffer")
        raw_ttl = self.config.get("background_event_ttl")
        raw_approval = self.config.get("background_approval_timeout")
        try:
            event_buffer = max(1, int(raw_buffer if raw_buffer is not None
                                      else 20000))
        except (TypeError, ValueError):
            event_buffer = 20000
        try:
            event_ttl = max(0.0, float(raw_ttl if raw_ttl is not None else 300))
        except (TypeError, ValueError):
            event_ttl = 300.0
        try:
            approval_timeout = max(0.0, float(
                raw_approval if raw_approval is not None else 0))
        except (TypeError, ValueError):
            approval_timeout = 0.0
        self.manager = TurnManager(
            max_active=1, buffer_events=event_buffer,
            completed_ttl=event_ttl, subscriber_queue=2000)
        self.approval_timeout = approval_timeout
        self.repl: Optional[Repl] = None
        self.repl_thread: Optional[threading.Thread] = None
        self.server: Optional[_UnixServer] = None
        self.active: Optional[LineConnection] = None
        self.connections: set[LineConnection] = set()
        # Output produced between taking a new client over and sending its
        # hello must not arrive before the hello: the client consumes the
        # first line synchronously.
        self.accepting_output = False
        self.lock = threading.RLock()
        self._attach_lock = threading.RLock()
        self.closing = False
        self.closed = threading.Event()
        self._finished = False
        self.state = "starting"
        self.fresh = True
        self.pending_prompt: Optional[dict[str, Any]] = None
        self.prompt_future: Optional[Future] = None
        self.pending_question: Optional[dict[str, Any]] = None
        self.question_future: Optional[Future] = None
        self.pending_error: Optional[str] = None
        self.active_turn_id: Optional[str] = None
        self.terminal_sent = threading.Event()
        self.unseen_turn: Optional[dict[str, Any]] = None
        self.output_buffer: list[str] = []
        self.output_size = 0
        self._saved_console: Optional[tuple[Any, Any, Any, Any]] = None
        self._stop_reason = "close"
        self.started_at = now_iso()

    # ── console ────────────────────────────────────────────────────────────

    def _install_console(self) -> None:
        try:
            self._saved_console = (
                self.console.file, self.console.size,
                self.err_console.file, self.err_console.size)
        except Exception:  # pragma: no cover - exotic console doubles
            self._saved_console = None
        try:
            self.console.file = self.output
            self.err_console.file = self.output
            self.console.size = (100, 30)
            self.err_console.size = (100, 30)
        except Exception:
            logger.debug("could not route the console", exc_info=True)

    def _restore_console(self) -> None:
        if self._saved_console is None:
            return
        try:
            con_file, con_size, err_file, err_size = self._saved_console
            self.console.file = con_file
            self.console.size = (con_size.width, con_size.height)
            self.err_console.file = err_file
            self.err_console.size = (err_size.width, err_size.height)
        except Exception:
            logger.debug("could not restore the console", exc_info=True)
        self._saved_console = None

    def write_output(self, text: str) -> None:
        if not text:
            return
        with self.lock:
            self.output_buffer.append(text)
            self.output_size += len(text)
            while self.output_size > OUTPUT_BUFFER_BYTES \
                    and len(self.output_buffer) > 1:
                removed = self.output_buffer.pop(0)
                self.output_size -= len(removed)
            if self.output_size > OUTPUT_BUFFER_BYTES \
                    and self.output_buffer:
                overflow = self.output_size - OUTPUT_BUFFER_BYTES
                tail = self.output_buffer[-1][overflow:]
                self.output_buffer = [tail]
                self.output_size = len(tail)
            active = self.active
            if active is not None and not active.closed \
                    and self.accepting_output:
                active.send({"op": "output", "text": text})

    def _take_output(self) -> str:
        with self.lock:
            text = "".join(self.output_buffer)
            self.output_buffer = []
            self.output_size = 0
            return text

    def _attach_output(self, conn: LineConnection) -> None:
        """Hand a new client the output since the last prompt, then go live."""
        with self.lock:
            text = "".join(self.output_buffer)
            self.output_buffer = []
            self.output_size = 0
            self.accepting_output = True
        if text and not conn.closed:
            conn.send({"op": "output", "text": text})

    def _set_size(self, width: Any, height: Any) -> None:
        try:
            width = max(20, min(500, int(width)))
            height = max(5, min(300, int(height)))
        except (TypeError, ValueError):
            return
        try:
            self.console.size = (width, height)
            self.err_console.size = (width, height)
        except Exception:
            pass

    # ── metadata ───────────────────────────────────────────────────────────

    def _meta(self) -> dict[str, Any]:
        session = self.repl.session if self.repl is not None else self.session
        return {
            "protocol": HOST_PROTOCOL, "version": VERSION,
            "pid": os.getpid(), "socket": str(self.paths.socket),
            "session_id": session.id, "title": session.display_title(),
            "model": session.model, "remote": session.remote,
            "started_at": self.started_at, "state": self.state,
            "attached_pid": self.active.pid if self.active is not None else None,
            "turn_id": self.active_turn_id,
        }

    def _write_meta(self) -> None:
        if self._finished:
            return
        _write_meta(self.paths, self._meta())

    # ── settings ───────────────────────────────────────────────────────────

    def settings(self) -> dict[str, Any]:
        r = self.repl
        if r is None:
            return {
                "model": self.session.model,
                "session_id": self.session.id,
                "title": self.session.display_title(),
                "host": self.spec.base, "remote": self.session.remote,
                "render": str(self.spec.args.get("render") or "markdown"),
                "math": True, "thinking": "collapsed", "tps": True,
                "pager": False, "nowordwrap": False,
                "max_render_chars": 12000, "show_stats": True,
                "preloading": False,
            }
        try:
            return {
                "model": r.session.model,
                "session_id": r.session.id,
                "title": r.session.display_title(),
                "host": r.client.base, "remote": r.session.remote,
                "render": r.render_mode, "math": bool(r.math),
                "thinking": r.thinking_mode, "tps": bool(r.tps_footer),
                "pager": bool(r.pager),
                "nowordwrap": bool(getattr(r.args, "nowordwrap", False)),
                "max_render_chars": int(r.max_render_chars),
                "show_stats": bool(r.show_stats),
                "preloading": bool(r._preload_thread
                                   and r._preload_thread.is_alive()),
            }
        except Exception:
            logger.debug("settings snapshot failed", exc_info=True)
            return {"model": r.session.model, "session_id": r.session.id,
                    "title": r.session.display_title()}

    def send_settings(self) -> None:
        with self.lock:
            active = self.active if self.accepting_output else None
        if active is not None and not active.closed:
            active.send({"op": "settings", "settings": self.settings()})

    # ── UI requests ────────────────────────────────────────────────────────

    def _wait(self, future: Future, *, interrupt: str) -> dict[str, Any]:
        while True:
            if self.closing:
                if interrupt == "detach":
                    raise HostClosed("the session host is closing")
                if interrupt == "eof":
                    raise EOFError
                raise KeyboardInterrupt
            try:
                return future.result(timeout=0.2)
            except FutureTimeout:
                continue

    def read_line(self, prompt: str) -> str:
        if self.closing:
            raise EOFError
        if self.repl is not None and self.repl.session.id != self.session_id:
            self.rekey(self.repl.session.id)
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        future: Future = Future()
        with self.lock:
            self.pending_prompt = {
                "id": request_id, "prompt": prompt,
                "settings": self.settings()}
            self.prompt_future = future
            self.state = "prompt"
            active = self.active if self.accepting_output else None
            # The attached client already saw this command's output live;
            # a prompt starts a new output epoch. With no client, keep the
            # buffer so the next attach can replay it.
            if active is not None and not active.closed:
                self.output_buffer = []
                self.output_size = 0
        if active is not None and not active.closed:
            active.send({"op": "prompt", **self.pending_prompt})
        try:
            answer = self._wait(future, interrupt="eof")
        finally:
            with self.lock:
                self.pending_prompt = None
                self.prompt_future = None
                if not self.closing:
                    self.state = "busy"
        if answer.get("eof"):
            raise EOFError
        return str(answer.get("text") or "")

    def request_question(self, kind: str, payload: dict[str, Any], *,
                         interrupt: str = "cancel") -> dict[str, Any]:
        if self.closing:
            if interrupt == "detach":
                raise HostClosed("the session host is closing")
            raise KeyboardInterrupt
        request_id = f"q_{uuid.uuid4().hex[:12]}"
        data = {"id": request_id, "kind": kind, **payload,
                "interrupt": interrupt}
        future: Future = Future()
        with self.lock:
            self.pending_question = data
            self.question_future = future
            self.state = "question"
            active = self.active if self.accepting_output else None
        if active is not None and not active.closed:
            active.send({"op": "question", **data})
        started = self.clock()
        try:
            while True:
                if self.closing:
                    if interrupt == "detach":
                        raise HostClosed("the session host is closing")
                    raise KeyboardInterrupt
                try:
                    answer = future.result(timeout=0.2)
                    break
                except FutureTimeout:
                    if self.approval_timeout > 0 and kind == "approval" \
                            and self.clock() - started >= self.approval_timeout:
                        answer = {"cancelled": True, "timeout": True}
                        break
                    continue
        finally:
            with self.lock:
                self.pending_question = None
                self.question_future = None
                if not self.closing:
                    self.state = "busy"
        if answer.get("cancelled"):
            if answer.get("stop"):
                raise KeyboardInterrupt
            if answer.get("timeout") and kind == "approval":
                return {"action": "deny", "rule": None}
            if interrupt == "detach":
                # A detach is not a cancel: the host keeps the turn parked.
                raise HostClosed("client detached")
            raise KeyboardInterrupt
        return answer

    def mark_waiting_approval(self, waiting: bool) -> None:
        turn_id = self.active_turn_id
        if turn_id:
            try:
                self.manager.mark_waiting_approval(turn_id, waiting=waiting)
            except Exception:
                pass

    # ── turns ──────────────────────────────────────────────────────────────

    def _turn_context(self) -> dict[str, Any]:
        try:
            report = self.repl.context_report() if self.repl else {}
        except Exception:
            return {}
        return {"used": report.get("used"), "total": report.get("window"),
                "source": report.get("source")}

    def run_turn(self, repl: Repl) -> Any:
        from .renderer import GenerationResult

        turn_id = f"turn_{uuid.uuid4().hex[:12]}"
        box: dict[str, Any] = {}
        token = CancelToken()
        user_event = getattr(repl, "_pending_user_event", None)
        repl._pending_user_event = None
        saved_sink = repl.event_sink
        saved_token = repl.cancel_token
        saved_tool_token = repl.tool_ctx.cancel

        def work(sink: Callable[[RuntimeEvent], None], cancel: CancelToken
                 ) -> TurnResult:
            coalescer = _Coalescer(sink)
            repl.event_sink = coalescer
            repl.cancel_token = cancel
            repl.tool_ctx.cancel = cancel
            try:
                if user_event:
                    coalescer(RuntimeEvent(MESSAGE_USER, dict(user_event)))
                result = repl._generate_local()
                box["result"] = result
                status = ("failed" if result.error else
                          "cancelled" if result.cancelled else "completed")
                return TurnResult(
                    id=turn_id, session_id=repl.session.id, status=status,
                    final_message={
                        "role": "assistant", "content": result.content,
                        "thinking": result.thinking or None,
                        "tool_calls": result.tool_calls or None},
                    usage=dict(result.stats or {}), rounds=repl._tool_round,
                    error=result.error)
            finally:
                coalescer.flush()
                repl.event_sink = saved_sink
                repl.cancel_token = saved_token
                repl.tool_ctx.cancel = saved_tool_token

        self.active_turn_id = turn_id
        self.terminal_sent = threading.Event()
        try:
            self.manager.submit(
                session_id=repl.session.id, work=work, turn_id=turn_id,
                cancel=token, context=self._turn_context)
        except ManagerClosed:
            self.active_turn_id = None
            return GenerationResult(cancelled=True)
        except SessionBusy:
            self.active_turn_id = None
            return GenerationResult(error="the session is already running a turn")
        with self.lock:
            active = self.active if self.accepting_output else None
        if active is not None and not active.closed:
            self.start_turn_writer(active, turn_id)
        self.manager.wait(turn_id)
        with self.lock:
            attached = self.active is not None
        if attached:
            self.terminal_sent.wait(timeout=3.0)
        self.active_turn_id = None
        result = box.get("result")
        if result is None:
            return GenerationResult(error="the turn ended without a result")
        if not attached:
            status = ("failed" if result.error else
                      "cancelled" if result.cancelled else "completed")
            self.unseen_turn = {
                "turn_id": turn_id, "status": status,
                "model": repl.session.model, "content": result.content,
                "thinking": result.thinking or None,
                "stats": dict(result.stats or {}),
                "error": result.error, "rounds": repl._tool_round,
                "tool_calls": len(result.tool_calls or []),
            }
        return result

    def start_turn_writer(self, conn: LineConnection, turn_id: str) -> None:
        try:
            snapshot = self.manager.get(turn_id)
        except Exception:
            snapshot = None
        attach_seq = snapshot.latest_seq if snapshot is not None else 0

        def write() -> None:
            subscription = None
            replay_done = False
            try:
                subscription = self.manager.subscribe(turn_id, after=0)
                while not self.closing and not conn.closed:
                    event = subscription.get(timeout=0.2)
                    if event is None:
                        if subscription.closed:
                            break
                        continue
                    if not conn.send({"op": "turn.event",
                                      "event": event.to_dict()}):
                        break
                    if event.type in TERMINAL_EVENTS:
                        if not replay_done:
                            conn.send({"op": "replay_done",
                                       "seq": event.seq})
                        self.terminal_sent.set()
                        break
                    if not replay_done and event.seq >= attach_seq:
                        conn.send({"op": "replay_done", "seq": event.seq})
                        replay_done = True
                if not replay_done and not self.closing:
                    conn.send({"op": "replay_done", "seq": attach_seq})
            except Exception:
                logger.debug("turn writer stopped", exc_info=True)
            finally:
                if subscription is not None:
                    subscription.close()

        threading.Thread(target=write, daemon=True,
                         name=f"host-turn-{turn_id}").start()

    # ── attach ─────────────────────────────────────────────────────────────

    def _hello(self, conn: LineConnection, fresh: bool) -> dict[str, Any]:
        snapshot = None
        if self.active_turn_id:
            try:
                snapshot = self.manager.get(self.active_turn_id)
            except Exception:
                snapshot = None
        active_turn = None
        if snapshot is not None and snapshot.status not in (
                "completed", "failed", "cancelled"):
            active_turn = {
                "id": snapshot.id, "status": snapshot.status,
                "created_at": snapshot.created_at, "seq": snapshot.latest_seq}
        return {
            "op": "hello", "protocol": HOST_PROTOCOL, "version": VERSION,
            "features": list(HOST_FEATURES),
            "fresh": fresh, "pid": os.getpid(), "session": self._meta(),
            "state": self.state, "active_turn": active_turn,
            "unseen_turn": self.unseen_turn,
            "settings": self.settings(),
        }

    def _take_over(self, new: LineConnection) -> None:
        with self.lock:
            old = self.active
            self.active = new
        if old is not None and old is not new:
            try:
                old.send({"op": "taken_over",
                          "pid": new.pid, "socket": str(self.paths.socket)})
                _wake_pid(old.pid)
            finally:
                old.close()
        self._write_meta()

    def handle_connection(self, sock: socket.socket) -> None:
        conn: Optional[LineConnection] = None
        try:
            conn = LineConnection(sock)
            if conn.uid is not None and conn.uid != os.getuid():
                logger.warning("refused a session-host connection from uid %s",
                               conn.uid)
                return
            with self.lock:
                self.connections.add(conn)
            # First message decides the connection's fate: ping/close are
            # control connections, attach becomes the one active client.
            try:
                sock.settimeout(30.0)
                first = conn.recv()
            finally:
                try:
                    sock.settimeout(None)
                except OSError:
                    pass
            if first is None:
                return
            op = str(first.get("op") or "")
            if op == "ping":
                conn.send({"op": "pong", "pid": os.getpid(),
                           "version": VERSION})
                return
            if op == "close":
                conn.send({"op": "shutdown", "reason": "close-request"})
                self.request_shutdown("close")
                return
            if op != "attach":
                conn.send({"op": "error", "message": "expected attach"})
                return
            protocol = first.get("protocol")
            if protocol != HOST_PROTOCOL:
                conn.send({"op": "error",
                           "message": f"protocol {protocol!r} is not "
                                      f"supported (host speaks {HOST_PROTOCOL})"})
                return
            # Serialize attach/setup: two terminals racing the first attach
            # must not build two Repls or deliver output before a hello.
            with self._attach_lock:
                self._complete_attach(conn, first)
            while not self.closing:
                message = conn.recv()
                if message is None:
                    break
                self._handle_client_message(conn, message)
        except (OSError, ValueError):
            logger.debug("host connection ended", exc_info=True)
        finally:
            if conn is not None:
                conn.close()
                with self.lock:
                    self.connections.discard(conn)
                    if self.active is conn:
                        self.active = None
                self._write_meta()

    def _complete_attach(self, conn: LineConnection,
                         first: dict[str, Any]) -> None:
        self._set_size(first.get("width"), first.get("height"))
        self.accepting_output = False
        self._take_over(conn)
        fresh = self.repl is None
        if fresh:
            # Hello first: the client consumes the first line synchronously,
            # and the Repl build may print.
            conn.send(self._hello(conn, fresh=True))
            self._attach_output(conn)
            try:
                self._build_repl()
            except Exception as exc:
                logger.debug("repl build failed", exc_info=True)
                conn.send({"op": "error",
                           "message": f"the session host could not start: {exc}"})
                self.request_shutdown("startup-error")
                return
            self._start_repl()
            return
        conn.send(self._hello(conn, fresh=False))
        self._attach_output(conn)
        with self.lock:
            pending_error = self.pending_error
            self.pending_error = None
        if pending_error:
            conn.send({"op": "error", "message": pending_error})
        overrides = first.get("overrides") or {}
        if overrides:
            applied, reason = self.apply_overrides(overrides)
            if not applied and reason:
                conn.send({"op": "warning", "message": reason})
            elif self.repl is not None:
                conn.send({"op": "settings", "settings": self.settings()})
        with self.lock:
            pending_question = self.pending_question
            pending_prompt = self.pending_prompt
            active_turn = self.active_turn_id
        if active_turn:
            self.start_turn_writer(conn, active_turn)
        if pending_question is not None:
            conn.send({"op": "question", **pending_question})
        elif pending_prompt is not None:
            pending_prompt = dict(pending_prompt)
            pending_prompt["settings"] = self.settings()
            conn.send({"op": "prompt", **pending_prompt})

    def _handle_client_message(self, conn: LineConnection,
                               message: dict[str, Any]) -> None:
        op = str(message.get("op") or "")
        if op == "line":
            with self.lock:
                pending = self.pending_prompt
                future = self.prompt_future
            if pending is not None and future is not None \
                    and message.get("id") == pending.get("id") \
                    and not future.done():
                future.set_result({"text": str(message.get("text") or ""),
                                   "eof": bool(message.get("eof"))})
            return
        if op == "answer":
            with self.lock:
                pending = self.pending_question
                future = self.question_future
            if pending is not None and future is not None \
                    and message.get("id") == pending.get("id") \
                    and not future.done():
                future.set_result(dict(message))
            return
        if op == "complete":
            request_id = message.get("id")
            try:
                from .repl import completion_candidates
                candidates = ([str(item) for item in
                               completion_candidates(self.repl, str(message.get("text") or ""))]
                              if self.repl is not None else [])
            except Exception:
                logger.debug("completion failed", exc_info=True)
                candidates = []
            conn.send({"op": "complete.result", "id": request_id,
                       "candidates": candidates})
            return
        if op == "resize":
            self._set_size(message.get("width"), message.get("height"))
            return
        if op == "seen":
            if self.unseen_turn is not None \
                    and self.unseen_turn.get("turn_id") == message.get("turn_id"):
                self.unseen_turn = None
            return
        if op == "cancel":
            self.cancel_active_turn()
            return
        if op == "thinking":
            mode = str(message.get("mode") or "")
            if mode in THINKING_MODES and self.repl is not None:
                self.repl.thinking_mode = mode
                self.send_settings()
            return
        if op == "close":
            self.request_shutdown("close")
            return
        if op == "detach":
            conn.close()
            return
        if op == "ping":
            conn.send({"op": "pong", "pid": os.getpid(), "version": VERSION})
            return
        if op == "chat":
            # Friendly answer for a stale/unknown protocol from a newer client.
            conn.send({"op": "error", "message": f"unknown op {op!r}"})

    def cancel_active_turn(self) -> bool:
        """Ctrl+Q: cancel the running turn, including a parked approval."""
        turn_id = self.active_turn_id
        if not turn_id:
            return False
        try:
            self.manager.cancel(turn_id)
        except Exception:
            logger.debug("turn cancel failed", exc_info=True)
        with self.lock:
            pending = self.pending_question
            future = self.question_future
            active = self.active
        if pending is not None and future is not None and not future.done():
            if active is not None and not active.closed:
                active.send({"op": "question_cancelled",
                             "id": pending.get("id")})
            future.set_result({"cancelled": True, "stop": True})
        return True

    # ── Repl lifecycle ─────────────────────────────────────────────────────

    def _build_repl(self) -> None:
        args_dict = dict(self.spec.args)
        args = argparse.Namespace(**args_dict)
        config = self.config
        timeout = (args_dict.get("timeout") or config.get("timeout")
                   or DEFAULT_TIMEOUT)
        client = self._client_factory(self.spec.base,
                                      insecure=bool(self.spec.insecure),
                                      timeout=timeout)
        repl = Repl(args, config, client, self.store, self.session)
        repl.host_mode = True
        repl.headless = True
        repl.pending_images = list(self.spec.pending_images)
        repl.ui = HostUi(self)
        repl.prompt_provider = repl.ui.read_line
        repl.turn_runner = self.run_turn
        repl.switch_guard = self.switch_guard
        repl.settings_changed = self.send_settings
        self.repl = repl
        self.session = repl.session
        # A brand-new session has no file until its first turn. Save it now
        # so `-c ID` and `--close-session ID` can resolve the live host even
        # when the window closes before the first message.
        if not repl.no_save:
            try:
                self.store.load(self.session.id)
            except FileNotFoundError:
                try:
                    self.store.save(self.session)
                except OSError:
                    logger.debug("initial session save failed", exc_info=True)
            except (ValueError, TypeError):
                pass

    def _start_repl(self) -> None:
        if self.repl is None or (self.repl_thread is not None
                                 and self.repl_thread.is_alive()):
            return
        self.fresh = False
        self._write_meta()

        def run() -> None:
            try:
                self.repl.run()  # type: ignore[union-attr]
            except EOFError:
                pass
            except HostClosed:
                logger.debug("host closed the repl")
                if self.repl is not None and not self.repl.no_save:
                    try:
                        self.store.save(self.repl.session)
                    except Exception:
                        pass
            except BaseException:
                logger.debug("session repl crashed", exc_info=True)
                try:
                    if self.repl is not None and not self.repl.no_save:
                        self.store.save(self.repl.session)
                except Exception:
                    pass
                self._send_error("the session host failed")
            finally:
                self.request_shutdown("repl-exit")

        self.repl_thread = threading.Thread(target=run, daemon=True,
                                            name="host-repl")
        self.repl_thread.start()

    def _send_error(self, message: str) -> None:
        with self.lock:
            self.pending_error = message
            active = self.active if self.accepting_output else None
        if active is not None and not active.closed:
            active.send({"op": "error", "message": message})
            self.pending_error = None

    # ── overrides / rekey ──────────────────────────────────────────────────

    def apply_overrides(self, overrides: dict[str, Any]) -> tuple[bool, str]:
        r = self.repl
        if r is None:
            return True, ""
        if self.state != "prompt" or self.active_turn_id:
            return False, ("launch overrides were ignored: the session is "
                           "running; reattach without them or apply them later")
        changed = False
        session = r.session
        model = overrides.get("model")
        if model and model != session.model:
            try:
                r.unload_model()
            except Exception:
                pass
            session.model = str(model)
            changed = True
        if "system" in overrides:
            session.system = overrides["system"]
            changed = True
        if "think" in overrides:
            session.think = overrides["think"]
            changed = True
        if "format" in overrides:
            session.format = overrides["format"]
            changed = True
        if "keepalive" in overrides:
            session.keepalive = overrides["keepalive"]
            changed = True
        if "gpu_layers" in overrides:
            new_layers = overrides["gpu_layers"]
            if new_layers != session.gpu_layers:
                try:
                    r.unload_model()
                except Exception:
                    pass
                session.gpu_layers = new_layers
                changed = True
        if "parameters" in overrides:
            session.parameters = dict(overrides["parameters"] or {})
            changed = True
        if "tags" in overrides:
            session.tags = list(overrides["tags"] or [])
            changed = True
        if "title" in overrides:
            session.title = str(overrides["title"] or "")
            changed = True
        if changed:
            session.touch()
            if not r.no_save:
                try:
                    r.store.save(session)
                except OSError:
                    logger.debug("override save failed", exc_info=True)
        return True, ""

    def switch_guard(self, target: Session) -> bool:
        if self.repl is None or target.id == self.repl.session.id:
            return True
        if host_is_running(target.id):
            console.print(f"[err]session {escape_text(target.id)} is already "
                          f"served by a background host[/err] — close it with "
                          f"`ollamarun --close-session {escape_text(target.id)}`")
            return False
        return True

    def rekey(self, new_id: str) -> bool:
        if not new_id or new_id == self.session_id:
            return True
        new_paths = host_paths(new_id, self.paths.root)
        if host_is_running(new_id, self.paths.root):
            console.print(f"[err]session {escape_text(new_id)} already has a "
                          f"background host[/err]")
            return False
        try:
            new_paths.socket.parent.mkdir(parents=True, exist_ok=True)
            try:
                new_paths.socket.unlink()
            except OSError:
                pass
            os.replace(self.paths.socket, new_paths.socket)
            try:
                os.replace(self.paths.meta, new_paths.meta)
            except OSError:
                pass
        except OSError:
            logger.debug("socket rekey failed", exc_info=True)
            return False
        self.paths = new_paths
        self.session_id = new_id
        self.spec.session_id = new_id
        if self.repl is not None and not self.repl.no_save:
            try:
                self.repl.store.save(self.repl.session)
            except OSError:
                pass
        self._write_meta()
        return True

    # ── server lifecycle ───────────────────────────────────────────────────

    def _bind(self) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.paths.root, 0o700)
        except OSError:
            pass
        try:
            self.paths.socket.unlink()
        except OSError:
            pass
        old_umask = os.umask(0o077)
        try:
            self.server = _UnixServer(str(self.paths.socket),
                                      _ConnectionHandler)
            self.server.host = self  # type: ignore[attr-defined]
        finally:
            os.umask(old_umask)
        try:
            os.chmod(self.paths.socket, 0o600)
        except OSError:
            pass
        self._write_meta()

    def _install_signals(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        def on_term(signum, frame) -> None:
            self.request_shutdown("signal")
        signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        except (AttributeError, ValueError):
            pass

    def serve(self) -> None:
        self._install_console()
        self._bind()
        self._install_signals()
        try:
            assert self.server is not None
            self.server.serve_forever(poll_interval=0.2)
        finally:
            self._finish()

    def request_shutdown(self, reason: str = "close") -> None:
        with self.lock:
            if self.closing:
                return
            self.closing = True
            self.state = "closing"
            self._stop_reason = reason
        self.manager.begin_shutdown()
        if self.active_turn_id:
            try:
                self.manager.cancel(self.active_turn_id)
            except Exception:
                pass
        with self.lock:
            current = self.server
        if current is not None:
            threading.Thread(target=current.shutdown, daemon=True,
                             name="host-shutdown").start()
        self._write_meta()

    def _finish(self) -> None:
        with self.lock:
            already = self.closed.is_set()
        if already:
            return
        self._finished = True
        self.closing = True
        if self.server is not None:
            try:
                self.server.shutdown()
            except Exception:
                pass
        try:
            self.manager.begin_shutdown()
            self.manager.shutdown(cancel=True, timeout=5.0)
        except Exception:
            logger.debug("turn manager shutdown failed", exc_info=True)
        thread = self.repl_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=20.0)
        with self.lock:
            active = self.active
            connections = list(self.connections)
            self.active = None
        if active is not None and not active.closed:
            try:
                active.send({"op": "shutdown", "reason": self._stop_reason})
                _wake_pid(active.pid)
            except Exception:
                pass
        for conn in connections:
            conn.close()
        with self.lock:
            self.connections.clear()
        if self.server is not None:
            try:
                self.server.server_close()
            except Exception:
                pass
        self._restore_console()
        for path in (self.paths.socket, self.paths.meta, self.paths.spec):
            try:
                path.unlink()
            except OSError:
                pass
        self.closed.set()


def escape_text(value: str) -> str:
    # Kept local so this module does not import rich's markup helpers just
    # for two error lines.
    from rich.markup import escape
    return escape(str(value))


def daemon_main(argv: Optional[list[str]] = None) -> int:
    """Hidden entry point: rebuild and serve one session from its spec."""
    parser = argparse.ArgumentParser(prog="ollamarun __session-host")
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    try:
        spec = HostSpec.read(Path(args.spec))
    except (OSError, ValueError) as exc:
        print(f"session host: cannot read spec: {exc}", file=sys.stderr)
        return 2
    if spec.protocol != HOST_PROTOCOL:
        print(f"session host: protocol {spec.protocol} is not supported",
              file=sys.stderr)
        return 2
    setup_logging(spec.debug)
    paths = host_paths(spec.session_id,
                       Path(spec.host_root) if spec.host_root else None)
    try:
        Path(args.spec).unlink()
    except OSError:
        pass
    if spec.cwd:
        try:
            os.chdir(spec.cwd)
        except OSError:
            pass
    server = SessionHostServer(spec, paths)
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.request_shutdown("exit")
        except Exception:
            pass
        server._finish()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual debugging aid
    raise SystemExit(daemon_main())
