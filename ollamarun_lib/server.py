"""Generic HTTP plumbing for the ollamarun API server.

`Router` + handler functions, a small JSON request/response model with
sanitized error envelopes, request body caps, and a Server-Sent-Events
writer -- the transport layer `openai_api.py` maps the OpenAI-compatible
surface onto. Nothing in this module knows about OpenAI, Ollama or the
agent runtime.

Handlers are plain callables `(request, params) -> Response | StreamResponse`.
The server turns `ApiError` into a structured JSON error envelope and every
other exception into a sanitized 500: responses carry no tracebacks and no
internal paths, while the full exception goes to the `--debug` log.

The listener is a `ThreadingHTTPServer` (one thread per connection, daemon
threads so a stalled SSE stream can never block shutdown); request bodies
are capped before they are read, sockets have a read timeout, and
`run_forever()` maps SIGINT/SIGTERM onto a clean `shutdown()`.
"""
from __future__ import annotations

import json
import logging
import signal
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional, Union
from urllib.parse import parse_qs, unquote, urlsplit

from .constants import (VERSION, DEFAULT_SERVE_BODY_LIMIT,
                        DEFAULT_SERVE_PING_INTERVAL,
                        DEFAULT_SERVE_REQUEST_TIMEOUT)

#: Cap on one request body, checked before the body is read.
DEFAULT_BODY_LIMIT = DEFAULT_SERVE_BODY_LIMIT
#: How long an SSE stream may stay silent before a `: ping` comment goes out.
DEFAULT_PING_INTERVAL = DEFAULT_SERVE_PING_INTERVAL
#: Socket timeout for one request (a stalled client is dropped, not queued).
DEFAULT_REQUEST_TIMEOUT = DEFAULT_SERVE_REQUEST_TIMEOUT

logger = logging.getLogger("ollamarun.server")


class ApiError(Exception):
    """A structured error a handler raises; the server renders the envelope."""

    def __init__(self, status: int, message: str, *,
                 type_: str = "invalid_request_error",
                 code: Optional[str] = None, param: Optional[str] = None,
                 extra: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.status = int(status)
        self.message = message
        self.type = type_
        self.code = code
        self.param = param
        #: Extra keys merged into the error object (e.g. `active_turn_id` on
        #: a 409 `session_busy`), keeping the OpenAI-style envelope shape.
        self.extra = dict(extra or {})

    def to_dict(self) -> dict[str, Any]:
        error = {"message": self.message, "type": self.type,
                 "param": self.param, "code": self.code}
        error.update(self.extra)
        return {"error": error}


@dataclass
class Request:
    """One parsed HTTP request, as a handler sees it."""

    method: str
    path: str
    query: dict[str, list[str]] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    json: Any = None
    remote: str = ""
    request_id: str = ""

    def header(self, name: str, default: str = "") -> str:
        return self.headers.get(name.lower(), default)


@dataclass
class Response:
    """A complete JSON/text response with a Content-Length."""

    status: int = 200
    body: bytes = b""
    media_type: str = "application/json"
    headers: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def json(cls, payload: Any, *, status: int = 200,
             headers: Optional[list[tuple[str, str]]] = None) -> "Response":
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return cls(status=status, body=body, media_type="application/json",
                   headers=list(headers or []))

    @classmethod
    def text(cls, text: str, *, status: int = 200,
             media_type: str = "text/plain; charset=utf-8") -> "Response":
        return cls(status=status, body=text.encode("utf-8"),
                   media_type=media_type)


@dataclass
class StreamResponse:
    """An SSE body: `produce(writer)` writes events until it returns.

    `on_disconnect` is called at most once when a write fails (the client
    went away), so a streaming handler can cancel upstream work; the
    heartbeat thread calls it too when it is the first to notice.
    """

    produce: Callable[["SseWriter"], None]
    on_disconnect: Optional[Callable[[], None]] = None


Handler = Callable[[Request, dict[str, str]], Union[Response, StreamResponse]]


class SseWriter:
    """Server-Sent-Events over chunked transfer encoding.

    Every event is one HTTP chunk, flushed immediately. Writes are
    lock-guarded because the server's heartbeat thread shares the stream; a
    write failure marks the stream broken exactly once and notifies the
    disconnect callback. After that every write is a no-op.
    """

    def __init__(self, wfile: Any, *,
                 on_disconnect: Optional[Callable[[], None]] = None):
        self._wfile = wfile
        self._lock = threading.Lock()
        self._on_disconnect = on_disconnect
        self._notified = False
        self.broken = False
        self.closed = False
        self.last_write = time.monotonic()

    def _write(self, data: bytes) -> None:
        notify = False
        with self._lock:
            if self.closed or self.broken:
                return
            try:
                self._wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                self._wfile.flush()
                self.last_write = time.monotonic()
            except (OSError, ValueError):
                self.broken = True
                notify = not self._notified
                self._notified = True
        if notify and self._on_disconnect is not None:
            try:
                self._on_disconnect()
            except Exception:  # a going-away client must not break the server
                logger.debug("on_disconnect callback failed", exc_info=True)

    def event(self, data: str, *, event: Optional[str] = None,
              id: Optional[str] = None) -> None:
        """Send one `data:` event (optionally with an `event:` name / `id:`).

        The `id:` is what an SSE client echoes back as `Last-Event-ID`, so
        the native turn stream sets it to the envelope's `<turn>:<seq>`.
        """
        prefix = ""
        if id is not None:
            prefix += f"id: {id}\n"
        if event:
            prefix += f"event: {event}\n"
        self._write(f"{prefix}data: {data}\n\n".encode("utf-8"))

    def ping(self) -> None:
        """Send an SSE comment; keeps proxies/clients from timing out."""
        self._write(b": ping\n\n")

    def close(self) -> None:
        """Finish the chunked body (safe to call twice, never raises)."""
        with self._lock:
            if self.closed or self.broken:
                self.closed = True
                return
            self.closed = True
            try:
                self._wfile.write(b"0\r\n\r\n")
                self._wfile.flush()
            except (OSError, ValueError):
                pass


class Router:
    """Method + path-pattern dispatch; `{name}` captures one decoded segment.

    Splitting before unquoting means an encoded slash (`%2F`) stays inside
    one segment, so model ids like `owner/name` match `{model}` in one piece.
    """

    def __init__(self) -> None:
        self._routes: list[tuple[str, tuple[str, ...], Handler]] = []

    def add(self, method: str, pattern: str, handler: Handler) -> None:
        self._routes.append((method.upper(), _split_path(pattern), handler))

    def resolve(self, method: str, path: str) -> tuple[Handler, dict[str, str]]:
        segments = _split_path(path)
        path_seen = False
        for route_method, pattern, handler in self._routes:
            params = _match(pattern, segments)
            if params is None:
                continue
            path_seen = True
            if route_method == method.upper():
                return handler, params
        if path_seen:
            raise ApiError(405, f"method {method} not allowed for {path}",
                           code="method_not_allowed")
        raise ApiError(404, f"unknown endpoint: {method} {path}",
                       code="not_found")


def _split_path(path: str) -> tuple[str, ...]:
    return tuple(unquote(part) for part in path.strip("/").split("/")
                 if part != "")


def _match(pattern: tuple[str, ...], segments: tuple[str, ...]
           ) -> Optional[dict[str, str]]:
    if len(pattern) != len(segments):
        return None
    params: dict[str, str] = {}
    for expected, actual in zip(pattern, segments):
        if expected.startswith("{") and expected.endswith("}"):
            params[expected[1:-1]] = actual
        elif expected != actual:
            return None
    return params


class _ThreadedServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True
    router: Router
    body_limit: int
    ping_interval: float
    request_timeout: float


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"ollamarun/{VERSION}"
    sys_version = ""
    timeout = DEFAULT_REQUEST_TIMEOUT

    def version_string(self) -> str:
        return self.server_version  # no trailing space from the empty sys_version

    def setup(self) -> None:
        # Apply the server's configured per-operation socket timeout before
        # the stdlib setup reads `self.timeout` (the class default would
        # otherwise ignore `HttpServer(request_timeout=…)`). It is a
        # per-operation timeout, not a total deadline: an SSE stream that
        # keeps writing (heartbeats included) is never cut off, a stalled
        # client's write fails and fires `on_disconnect`.
        configured = getattr(self.server, "request_timeout", None)
        if configured and float(configured) > 0:
            self.timeout = float(configured)
        super().setup()

    # ── verbs ───────────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch()

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch()

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch()

    def log_message(self, format: str, *args: Any) -> None:
        # The default writes every request to stderr; keep it in the debug log.
        logger.debug("http %s - %s", self.address_string(), format % args)

    # ── one request ─────────────────────────────────────────────────────────

    def _dispatch(self) -> None:
        request: Optional[Request] = None
        try:
            request = self._build_request()
            handler, params = self.server.router.resolve(request.method,
                                                         request.path)
            result = handler(request, params)
            if isinstance(result, StreamResponse):
                self._send_stream(request, result)
            elif isinstance(result, Response):
                self._send(result, request)
            else:
                raise TypeError(f"handler returned {type(result).__name__}")
        except socket.timeout:
            self._send_json_error(ApiError(408, "request timed out",
                                           type_="api_error",
                                           code="timeout"), request)
        except ApiError as exc:
            self._send_json_error(exc, request)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            _log_exception(
                f"unhandled error serving {self.command} {self.path}")
            self._send_json_error(
                ApiError(500, "internal server error", type_="api_error",
                         code="internal_error"), request)

    def _build_request(self) -> Request:
        parsed = urlsplit(self.path)
        body = self._read_body()
        request = Request(
            method=self.command,
            path=parsed.path or "/",
            query=parse_qs(parsed.query, keep_blank_values=True),
            headers={name.lower(): value for name, value in self.headers.items()},
            body=body,
            remote=self.client_address[0] if self.client_address else "",
            request_id="req_" + uuid.uuid4().hex[:16],
        )
        if body:
            try:
                request.json = json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise ApiError(400, "request body is not valid JSON",
                               code="invalid_json")
        return request

    def _read_body(self) -> bytes:
        transfer = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in transfer:
            raise ApiError(411, "chunked request bodies are not supported",
                           code="length_required")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        try:
            length = int(raw_length)
        except ValueError:
            raise ApiError(400, "invalid Content-Length header",
                           code="invalid_header")
        if length < 0:
            raise ApiError(400, "invalid Content-Length header",
                           code="invalid_header")
        limit = self.server.body_limit
        if length > limit:
            raise ApiError(413, f"request body exceeds the {limit} byte limit",
                           code="body_too_large")
        if length == 0:
            return b""
        data = self.rfile.read(length)
        if len(data) < length:
            raise ApiError(400, "request body was truncated",
                           code="invalid_body")
        return data

    def _send(self, response: Response, request: Optional[Request] = None) -> None:
        self.send_response(response.status)
        self.send_header("Content-Type", response.media_type)
        self.send_header("Content-Length", str(len(response.body)))
        self.send_header("X-Request-Id", request.request_id if request else "-")
        for name, value in response.headers:
            self.send_header(name, value)
        self.end_headers()
        if response.body:
            self.wfile.write(response.body)

    def _send_json_error(self, error: ApiError,
                         request: Optional[Request] = None) -> None:
        # The request body may not have been read (oversize, truncated, ...):
        # close the connection instead of parsing leftover bytes as a request.
        self.close_connection = True
        self._send(Response.json(error.to_dict(), status=error.status), request)

    def _send_stream(self, request: Request, stream: StreamResponse) -> None:
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Request-Id", request.request_id)
        self.end_headers()
        writer = SseWriter(self.wfile, on_disconnect=stream.on_disconnect)
        interval = float(self.server.ping_interval)
        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(1.0):
                if writer.closed or writer.broken:
                    return
                if time.monotonic() - writer.last_write >= interval:
                    writer.ping()

        beat: Optional[threading.Thread] = None
        if interval > 0:
            beat = threading.Thread(target=heartbeat, name="sse-ping",
                                    daemon=True)
            beat.start()
        try:
            stream.produce(writer)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            _log_exception(
                f"stream failed serving {request.method} {request.path}")
            writer.event(json.dumps({"error": {
                "message": "internal server error", "type": "api_error",
                "param": None, "code": "internal_error"}}))
        finally:
            stop.set()
            if beat is not None:
                beat.join(timeout=2.0)
            writer.close()


class HttpServer:
    """A threaded HTTP server bound to a router; `serve_forever()` blocks."""

    def __init__(self, router: Router, *, bind: str = "127.0.0.1",
                 port: int = 11440, body_limit: int = DEFAULT_BODY_LIMIT,
                 ping_interval: float = DEFAULT_PING_INTERVAL,
                 request_timeout: float = DEFAULT_REQUEST_TIMEOUT):
        self.router = router
        self.body_limit = int(body_limit)
        self.ping_interval = float(ping_interval)
        self.request_timeout = float(request_timeout)
        self._httpd = _ThreadedServer((bind, int(port)), _Handler)
        self._httpd.router = router
        self._httpd.body_limit = self.body_limit
        self._httpd.ping_interval = self.ping_interval
        self._httpd.request_timeout = self.request_timeout

    @property
    def address(self) -> tuple[str, int]:
        """The bound (host, port); port 0 in the constructor picks a free one."""
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    def serve_forever(self) -> None:
        self._httpd.serve_forever(poll_interval=0.2)

    def shutdown(self) -> None:
        self._httpd.shutdown()

    def close(self) -> None:
        try:
            self._httpd.server_close()
        except OSError:
            pass


def run_forever(server: HttpServer) -> None:
    """`serve_forever()` with SIGINT/SIGTERM mapped to a clean shutdown.

    The accept loop runs on a worker thread so a signal handler can only set
    a flag -- calling `shutdown()` from the thread that is *inside*
    `serve_forever()` would deadlock. Ctrl-C therefore lands here, stops the
    listener, and lets the process exit after the join.
    """
    stop = threading.Event()
    previous: dict[int, Any] = {}

    def request_stop(signum: int, frame: Any) -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, request_stop)
        except (ValueError, OSError):  # not the main thread / unsupported
            pass
    worker = threading.Thread(target=server.serve_forever, daemon=True,
                              name="ollamarun-listen")
    worker.start()
    try:
        while not stop.wait(0.2):
            if not worker.is_alive():
                break
    except KeyboardInterrupt:  # pragma: no cover - the handler covers this
        pass
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        server.shutdown()
        worker.join(timeout=10.0)


def _log_exception(message: str) -> None:
    """Tracebacks go to the debug log only; responses stay sanitized."""
    if logging.getLogger().handlers:
        logger.exception(message)
