"""Terminal client for a background session host (`session_host.py`).

The client is deliberately thin: it owns the prompt line, the terminal
rendering of turn events, command output and the interactive questions, but
never the session itself. Losing this process (window closed, Ctrl+C twice)
does not stop the host; `ollamarun -c` reattaches, replays the active turn and
shows anything that finished while no terminal was here.

The wire is newline-delimited JSON over a Unix socket. All messages are
handled by `SessionClient.run`; a reader thread only frames and dispatches,
so a prompt can be interrupted by `taken_over`/`shutdown` without waiting for
input.
"""
from __future__ import annotations

import os
import queue
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Optional

from rich.markup import escape
from rich.panel import Panel

from .constants import (
    HAS_PROMPT_TOOLKIT,
    HISTORY_PATH,
    VERSION,
    Completer,
    Completion,
    FileHistory,
    KeyBindings,
    PromptSession,
    console,
    patch_stdout,
    tool_color,
)
from .interactive import open_in_editor
from .renderer import (
    Renderer,
    format_stats,
    model_message_rule,
    render_content,
    thinking_panel,
    tool_call_panel,
    tool_result_panel,
    user_message_box,
)
from .repl import completion_fragment
from .session_host import (
    HOST_PROTOCOL,
    HostSpec,
    LineConnection,
    SessionHostError,
    SessionHostUnavailable,
    ensure_host,
    host_paths,
)
from .tools import ToolRule
from .ui import ConsoleUi


class InterruptGuard:
    """Two presses inside the window detach; one arms the warning."""

    def __init__(self, window: float = 10.0,
                 clock=time.monotonic):
        self.window = float(window)
        self.clock = clock
        self.armed_until: Optional[float] = None

    def press(self) -> str:
        now = self.clock()
        if self.armed_until is not None and now <= self.armed_until:
            self.armed_until = None
            return "detach"
        self.armed_until = now + self.window
        return "arm"

    def reset(self) -> None:
        self.armed_until = None


def interrupt_lines(generating: bool) -> list[str]:
    """The Ctrl+C detach warning plus the explicit-close/stop hints.

    The Ctrl+Q hint only appears while a response is actually streaming: a
    command such as /pull has no renderer polling the keyboard, so offering
    to stop "generation" there would be misleading.
    """
    lines = ["[warn]still running — press Ctrl+C again within 10s to detach; "
             "the session keeps going in the background[/warn]"]
    extra = "Ctrl+D terminates the session"
    if generating:
        extra += " · Ctrl+Q stops the current generation"
    lines.append(f"[dim]{extra}[/dim]")
    return lines


class RemoteCompleter(Completer):  # type: ignore[misc]
    """Completion lives in the host (commands, ids, models, @files)."""

    def __init__(self, client: "SessionClient"):
        self.client = client

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        fragment = completion_fragment(text)
        for candidate in self.client.complete(text):
            yield Completion(candidate, start_position=-len(fragment))


class SessionClient:
    """One terminal attached to one background session host."""

    def __init__(self, spec: HostSpec, conn: LineConnection, *,
                 overrides: Optional[dict[str, Any]] = None,
                 popen: Optional[Any] = None,
                 output_stream: Optional[Any] = None,
                 input_stream: Optional[Any] = None,
                 console_obj: Any = None):
        self.spec = spec
        self.conn = conn
        self.overrides = dict(overrides or {})
        self.popen = popen
        self._output = output_stream
        self._input = input_stream
        self.console = console_obj or console
        self.ui = ConsoleUi()
        self.queue: queue.Queue = queue.Queue(maxsize=20000)
        self.settings: dict[str, Any] = {}
        self.session: dict[str, Any] = {}
        self.session_id = spec.session_id or str(spec.session.get("id") or "")
        self.title = ""
        self.model = ""
        self.host_pid: Optional[int] = None
        self.fresh = False
        self.attached = False
        self.done = False
        self.exit_code = 0
        self.taken_over = False
        self.shutdown_requested = False
        self.closing = False
        self.busy = False
        self.turn_active = False
        self.replaying = False
        self.active_turn: Optional[dict[str, Any]] = None
        self.unseen_turn: Optional[dict[str, Any]] = None
        self.deferred_questions: list[dict[str, Any]] = []
        self.deferred_prompt: Optional[dict[str, Any]] = None
        self.prompt_active = threading.Event()
        self.question_active = threading.Event()
        self.output_pending: list[str] = []
        self.output_lock = threading.Lock()
        self.reader_thread: Optional[threading.Thread] = None
        self.renderer: Optional[Renderer] = None
        self.render_after_tools = False
        self._prompt_session: Optional[Any] = None
        self._prompt_session_tried = False
        self._completion: dict[str, Future] = {}
        self._guard = InterruptGuard()
        self._cancel_sent = False
        self._cancel_warned = False
        self._thinking_warned = False
        self._cancelled_questions: set[str] = set()
        self.features: set[str] = set()
        self._shutdown_event = threading.Event()
        self._proc_sigint = False
        self._loop_thread: Optional[threading.Thread] = None
        self._last_settings_sent = 0.0

    # ── attach ─────────────────────────────────────────────────────────────

    def attach(self) -> None:
        try:
            width, height = self.console.size
        except Exception:
            width, height = 100, 30
        attach = {
            "op": "attach", "protocol": HOST_PROTOCOL, "version": VERSION,
            "pid": os.getpid(), "width": width, "height": height,
            "overrides": self.overrides,
        }
        if not self.conn.send(attach):
            raise SessionHostUnavailable("could not send the attach request")
        try:
            self.conn.sock.settimeout(60.0)
            hello = self.conn.recv()
        except OSError as exc:
            raise SessionHostUnavailable(f"the session host did not answer: {exc}") from exc
        finally:
            try:
                self.conn.sock.settimeout(None)
            except OSError:
                pass
        if hello is None:
            raise SessionHostUnavailable("the session host closed the connection")
        op = str(hello.get("op") or "")
        if op in {"startup_error", "error"}:
            raise SessionHostError(str(hello.get("message") or
                                       "the session host failed to start"))
        if op != "hello":
            raise SessionHostUnavailable(f"unexpected host reply: {op!r}")
        protocol = hello.get("protocol")
        if protocol != HOST_PROTOCOL:
            raise SessionHostError(
                f"the running session host speaks protocol {protocol!r}, "
                f"this client speaks {HOST_PROTOCOL}")
        self.fresh = bool(hello.get("fresh"))
        self.features = set(hello.get("features") or ())
        self.host_pid = hello.get("pid")
        self.session = dict(hello.get("session") or {})
        self.settings = dict(hello.get("settings") or {})
        self.unseen_turn = hello.get("unseen_turn")
        self.active_turn = hello.get("active_turn")
        self.session_id = str(self.session.get("session_id")
                              or self.session_id)
        self.title = str(self.session.get("title") or "")
        self.model = str(self.session.get("model")
                         or self.settings.get("model") or "")
        self.busy = str(hello.get("state") or "") in {"busy", "question"}
        self.replaying = self.active_turn is not None
        self._guard.reset()
        self._cancel_sent = False
        self._cancelled_questions.clear()
        self.attached = True
        if not self.fresh:
            self._banner()
        if self.unseen_turn:
            self._render_unseen()
        self.reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="host-client-reader")
        self.reader_thread.start()

    # ── terminal output ────────────────────────────────────────────────────

    def _raw_stream(self) -> Any:
        return self._output if self._output is not None else sys.stdout

    def _write_raw(self, text: str) -> None:
        if not text:
            return
        stream = self._raw_stream()
        try:
            stream.write(text)
            stream.flush()
        except (OSError, ValueError):
            pass

    def _flush_output(self) -> None:
        with self.output_lock:
            pending = self.output_pending
            self.output_pending = []
        for text in pending:
            self._write_raw(text)

    def _queue_output(self, text: str) -> None:
        if self.renderer is not None or self.turn_active:
            with self.output_lock:
                self.output_pending.append(text)
        else:
            self._write_raw(text)

    def _banner(self) -> None:
        session = self.session
        model = escape(str(session.get("model") or self.model or "?"))
        sid = escape(self.session_id or "?")
        title = escape(str(session.get("title") or self.title or ""))
        host = escape(str(self.settings.get("host") or ""))
        remote = session.get("remote")
        lines = [f"[key]model[/key]   {model}",
                 f"[key]session[/key] {sid}" + (f" · {title}" if title else ""),
                 f"[key]host[/key]    {host}"
                 + (f" · {escape(str(remote))}" if remote else "")]
        if self.host_pid:
            lines.append(f"[key]background[/key] pid {self.host_pid}")
        if self.active_turn:
            status = str(self.active_turn.get("status") or "running")
            lines.append(f"[warn]{escape(status)} · replaying from the start[/warn]")
        self.console.print(Panel("\n".join(lines), title="ollamarun · attached",
                                 border_style="cyan", title_align="left"))

    def _render_unseen(self) -> None:
        turn = self.unseen_turn
        if not turn:
            return
        s = self.settings
        self.console.print()
        self.console.print("[answer]while you were away[/answer]", style="answer")
        self.console.print(model_message_rule(
            str(turn.get("model") or self.model or "")))
        thinking = turn.get("thinking")
        if thinking and s.get("thinking") != "hidden":
            self.console.print(thinking_panel(str(thinking)))
        content = str(turn.get("content") or "")
        if content:
            width = max(20, (self.console.width or 80) - 4)
            self.console.print(render_content(
                content, math=bool(s.get("math", True)), width=width))
        if turn.get("status") == "failed":
            self.console.print(f"[err]{escape(str(turn.get('error') or 'the turn failed'))}[/err]")
        elif turn.get("status") == "cancelled":
            self.console.print("[warn]the turn was cancelled[/warn]")
        stats = turn.get("stats") or {}
        if stats and s.get("show_stats", True):
            self.console.print(f"[dim]⏱  {format_stats(stats)}[/dim]")
        try:
            self.conn.send({"op": "seen",
                            "turn_id": turn.get("turn_id")})
        except Exception:
            pass
        self.unseen_turn = None

    # ── reader thread ──────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        while not self.done:
            message = self.conn.recv()
            if message is None:
                self._shutdown_event.set()
                try:
                    self.queue.put_nowait(("eof", None))
                except queue.Full:
                    pass
                return
            op = str(message.get("op") or "")
            if op == "output":
                text = str(message.get("text") or "")
                if self.prompt_active.is_set() or self.question_active.is_set():
                    self._write_raw(text)
                else:
                    try:
                        self.queue.put(("output", text))
                    except queue.Full:
                        pass
            elif op == "complete.result":
                future = self._completion.pop(str(message.get("id") or ""), None)
                if future is not None and not future.done():
                    future.set_result(list(message.get("candidates") or []))
            elif op in {"taken_over", "shutdown"}:
                if op == "taken_over":
                    self.taken_over = True
                    pid = message.get("pid")
                    self.console.print(
                        "[warn]session moved to another terminal"
                        + (f" (pid {pid})" if pid else "") + "[/warn]")
                else:
                    self.shutdown_requested = True
                    self.console.print("[dim]session closed[/dim]")
                self._shutdown_event.set()
                try:
                    self.queue.put((op, message))
                except queue.Full:
                    pass
                if self.prompt_active.is_set() or self.question_active.is_set():
                    self._wake()
            elif op == "question_cancelled":
                question_id = str(message.get("id") or "")
                if question_id:
                    self._cancelled_questions.add(question_id)
                if self.question_active.is_set():
                    self._wake()
            elif op == "settings":
                # Apply immediately instead of queueing it: the main loop is
                # parked inside prompt_toolkit while the user types, so a
                # queued update would not reach the rprompt until the next
                # line -- exactly when the waiting line is most useful.
                self.settings = {**self.settings,
                                 **(message.get("settings") or {})}
                self._sync_identity()
            elif op in {"prompt", "question", "turn.event", "replay_done",
                        "warning", "error"}:
                try:
                    self.queue.put((op, message))
                except queue.Full:
                    pass
            elif op == "error":
                try:
                    self.queue.put(("error", message))
                except queue.Full:
                    pass
            # hello is consumed synchronously in attach; pong is for discovery.

    def _wake(self) -> None:
        """Interrupt the main thread's prompt/question without killing it.

        Only a real terminal client running its loop on the process's main
        thread can be woken by SIGINT; an embedded/test client running on a
        worker thread is woken by its own input instead.
        """
        if (self._proc_sigint or self._input is not None
                or os.environ.get("OLLAMARUN_HOST_NO_SIGNAL")):
            return
        if self._loop_thread is not threading.main_thread():
            return
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except OSError:
            pass

    # ── main loop ──────────────────────────────────────────────────────────

    def run(self) -> int:
        if not self.attached:
            self.attach()
        self._loop_thread = threading.current_thread()
        while not self.done:
            try:
                kind, message = self.queue.get(timeout=0.1)
            except queue.Empty:
                if self.renderer is not None:
                    self.renderer.poll()
                self._check_renderer_keys()
                self._flush_output()
                continue
            except KeyboardInterrupt:
                self._on_interrupt()
                continue
            try:
                self._dispatch(kind, message or {})
                self._check_renderer_keys()
            except KeyboardInterrupt:
                self._on_interrupt()
            except (BrokenPipeError, ConnectionError, OSError) as exc:
                if self.renderer is not None:
                    self.renderer.abort()
                    self.renderer = None
                self.console.print(f"[err]session host connection lost: "
                                   f"{escape(str(exc))}[/err]")
                self.exit_code = 1
                self.done = True
            except Exception as exc:
                if self.renderer is not None:
                    self.renderer.abort()
                    self.renderer = None
                self.console.print(f"[err]session client failed: "
                                   f"{escape(str(exc))}[/err]")
                self.exit_code = 1
                self.done = True
        return self.exit_code

    def _dispatch(self, kind: str, message: dict[str, Any]) -> None:
        if kind == "output":
            text = (message if isinstance(message, str)
                    else str(message.get("text") or ""))
            self._queue_output(text)
            return
        if kind == "prompt":
            if self.replaying:
                self.deferred_prompt = message
            else:
                self._prompt_loop(message)
            return
        if kind == "question":
            if self.replaying:
                self.deferred_questions.append(message)
            else:
                self._question_loop(message)
            return
        if kind == "turn.event":
            self._handle_event(message.get("event") or {})
            return
        if kind == "replay_done":
            self.replaying = False
            self._flush_output()
            self._show_deferred()
            return
        if kind == "settings":
            self.settings.update(message.get("settings") or {})
            self._sync_identity()
            return
        if kind == "warning":
            self.console.print(f"[warn]{escape(str(message.get('message') or ''))}[/warn]")
            return
        if kind == "error":
            self.console.print(f"[err]{escape(str(message.get('message') or ''))}[/err]")
            self.exit_code = 1
            self.done = True
            return
        if kind == "taken_over":
            self.taken_over = True
            self.exit_code = 0
            self.done = True
            return
        if kind == "shutdown":
            self.shutdown_requested = True
            self.exit_code = 0
            self.done = True
            return
        if kind == "eof":
            if not (self.taken_over or self.shutdown_requested):
                self.console.print(
                    f"[err]the session host disconnected[/err] — reattach with "
                    f"`ollamarun -c {escape(self.session_id)}`")
                self.exit_code = 1
            self.done = True

    def _on_interrupt(self) -> None:
        if self.taken_over or self.shutdown_requested:
            self.done = True
            return
        if not (self.turn_active or self.busy):
            self.console.print("[dim]use /bye (or Ctrl+D) to close the "
                               "session[/dim]")
            return
        result = self._guard.press()
        if result == "detach":
            self._detach()
        else:
            for line in interrupt_lines(self.renderer is not None
                                        and "cancel" in self.features):
                self.console.print(line)

    def _detach(self) -> None:
        if self.renderer is not None:
            self.renderer.abort()
            self.renderer = None
        self.conn.close()
        self.console.print(
            f"[warn]detached[/warn] · the session keeps running · reattach with "
            f"`ollamarun -c {escape(self.session_id)}`")
        self.exit_code = 0
        self.done = True

    def cancel_turn(self) -> bool:
        """Ctrl+Q: ask the host to cancel the active turn, stay attached."""
        if self._cancel_sent:
            return False
        if not (self.turn_active or self.renderer is not None):
            return False
        if "cancel" not in self.features:
            if not self._cancel_warned:
                self._cancel_warned = True
                self.console.print(
                    "[warn]this session host predates Ctrl+Q; close and resume "
                    "the session to upgrade[/warn]")
            return False
        self._cancel_sent = True
        return self.conn.send({"op": "cancel"})

    def _thinking_changed(self, mode: str) -> None:
        """Ctrl+T in the client: keep the mode here and tell the host.

        The host owns the Repl, so without this its `thinking_mode` stays at
        the launch value and every later round/turn would come back
        collapsed. The local settings copy is updated too, so the next
        `round.started` in this client uses it before the host's echo lands.
        """
        self.settings["thinking"] = mode
        if "thinking" in self.features:
            self.conn.send({"op": "thinking", "mode": mode})
        elif not self._thinking_warned:
            self._thinking_warned = True
            self.console.print(
                "[warn]this session host predates the persistent thinking "
                "toggle; close and resume the session to upgrade[/warn]")

    def _close_session(self) -> None:
        """Ctrl+D while streaming: explicit close, not a detach."""
        if self.closing:
            return
        self.closing = True
        if self.renderer is not None:
            self.renderer.abort()
            self.renderer = None
        self.conn.send({"op": "close"})
        self._shutdown_event.wait(timeout=20.0)
        self.exit_code = 0
        self.done = True

    def _check_renderer_keys(self) -> bool:
        """Consume Ctrl+Q/Ctrl+D seen by the live renderer."""
        renderer = self.renderer
        if renderer is None:
            return False
        if renderer.close_requested:
            renderer.close_requested = False
            self._close_session()
            return True
        if renderer.stop_requested:
            renderer.stop_requested = False
            self.cancel_turn()
            return True
        return False

    def _sync_identity(self) -> None:
        self.session_id = str(self.settings.get("session_id")
                              or self.session_id)
        self.model = str(self.settings.get("model") or self.model)
        self.title = str(self.settings.get("title") or self.title)

    # ── prompt ─────────────────────────────────────────────────────────────

    def _build_prompt_session(self) -> Optional[Any]:
        if self._prompt_session_tried:
            return self._prompt_session
        self._prompt_session_tried = True
        if not HAS_PROMPT_TOOLKIT or self._input is not None:
            return None
        try:
            if not sys.stdin.isatty():
                return None
        except (AttributeError, ValueError, OSError):
            return None
        history = None
        if not os.environ.get("OLLAMA_NOHISTORY"):
            try:
                HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
                history = FileHistory(str(HISTORY_PATH))
            except OSError:
                history = None
        bindings = KeyBindings()

        @bindings.add("c-g")
        def _open_editor(event) -> None:  # pragma: no cover - interactive
            buffer = event.current_buffer
            edited = open_in_editor(buffer.text)
            if edited is not None:
                buffer.text = edited
                buffer.cursor_position = len(edited)

        @bindings.add("c-j")
        def _newline(event) -> None:  # pragma: no cover - interactive
            event.current_buffer.insert_text("\n")

        @bindings.add("c-q")
        def _ignore_stop(event) -> None:  # pragma: no cover - interactive
            # Ctrl+Q only stops a generation, not the idle prompt; consume it
            # so prompt_toolkit does not treat it as quoted-insert.
            return

        try:
            self._prompt_session = PromptSession(
                history=history, completer=RemoteCompleter(self),
                key_bindings=bindings, complete_while_typing=True,
                rprompt=self._rprompt, refresh_interval=0.1)
        except Exception:
            self._prompt_session = None
        return self._prompt_session

    def _rprompt(self) -> Optional[str]:
        # The host samples what the wait is actually doing (loading, waiting
        # for room, reloading at a new window, prefilling) and ships the line
        # in `settings`; `stage_active` says whether a wait is running, so an
        # idle note like "reloads ... on send" does not animate a spinner.
        text = self.settings.get("stage")
        frames = "⠋⠙⠹⠸⠴⠦⠧⠇"
        frame = frames[int(time.monotonic() * 10) % len(frames)]
        if text:
            if self.settings.get("stage_active"):
                return f"{frame} {text} "
            return f"{text} "
        if not self.settings.get("preloading"):
            return None
        return f"{frame} waiting for {self.model}… "

    def _read_prompt(self, prompt: str) -> str:
        session = self._build_prompt_session()
        if session is not None:
            with patch_stdout(raw=True):
                return session.prompt(prompt)
        if self._input is not None:
            self._write_raw(prompt)
            line = self._input.readline()
            if line == "":
                raise EOFError
            return line.rstrip("\n")
        return input(prompt)

    def _show_deferred(self) -> None:
        while self.deferred_questions and not self.done:
            self._question_loop(self.deferred_questions.pop(0))
        if self.deferred_prompt is not None and not self.done:
            prompt = self.deferred_prompt
            self.deferred_prompt = None
            self._prompt_loop(prompt)

    def _prompt_loop(self, message: dict[str, Any]) -> None:
        request_id = str(message.get("id") or "")
        prompt = str(message.get("prompt") or "› ")
        settings = message.get("settings") or {}
        if settings:
            self.settings.update(settings)
            self._sync_identity()
        self._current_prompt = request_id
        self._cancel_sent = False
        self.busy = False
        self.turn_active = False
        self._guard.reset()
        self._flush_output()
        self.prompt_active.set()
        try:
            while not self.done:
                try:
                    line = self._read_prompt(prompt)
                except EOFError:
                    self.conn.send({"op": "close"})
                    self.closing = True
                    self._shutdown_event.wait(timeout=20.0)
                    self.exit_code = 0
                    self.done = True
                    return
                except KeyboardInterrupt:
                    if self.taken_over or self.shutdown_requested:
                        self.done = True
                        return
                    self.console.print("[dim]use /bye (or Ctrl+D) to close "
                                       "the session[/dim]")
                    continue
                if self.taken_over or self.shutdown_requested:
                    return
                if not self.conn.send({"op": "line", "id": request_id,
                                       "text": line}):
                    if self.taken_over or self.shutdown_requested:
                        self.done = True
                        return
                    self.console.print("[err]the session host disconnected[/err]")
                    self.exit_code = 1
                    self.done = True
                    return
                self.busy = True
                return
        finally:
            self.prompt_active.clear()

    # ── questions ──────────────────────────────────────────────────────────

    def _question_loop(self, message: dict[str, Any]) -> None:
        request_id = str(message.get("id") or "")
        self.busy = False
        self.question_active.set()
        try:
            while not self.done:
                try:
                    answer = self._ask_question(message)
                except KeyboardInterrupt:
                    if self.taken_over or self.shutdown_requested:
                        self.done = True
                        return
                    if request_id in self._cancelled_questions:
                        self._cancelled_questions.discard(request_id)
                        self.busy = True
                        return
                    if message.get("interrupt") == "detach":
                        if self._guard.press() == "detach":
                            self._detach()
                            return
                        self.console.print(
                            "[warn]still waiting for your decision — press "
                            "Ctrl+C again within 10s to detach[/warn]")
                        continue
                    self.conn.send({"op": "answer", "id": request_id,
                                    "cancelled": True})
                    self.busy = True
                    return
                if not self.conn.send({"op": "answer", "id": request_id,
                                       **answer}):
                    self.console.print("[err]the session host disconnected[/err]")
                    self.exit_code = 1
                    self.done = True
                    return
                self.busy = True
                return
        finally:
            self.question_active.clear()

    def _ask_question(self, message: dict[str, Any]) -> dict[str, Any]:
        kind = str(message.get("kind") or "")
        if kind == "confirm":
            value = self.ui.confirm(str(message.get("question") or ""),
                                    default=bool(message.get("default")))
            return {"value": bool(value)}
        if kind == "ask":
            choices = message.get("choices")
            value = self.ui.ask(str(message.get("question") or ""),
                                choices=list(choices) if choices else None,
                                default=message.get("default"))
            return {"value": value}
        if kind == "approval":
            name = str(message.get("name") or "tool")
            reason = str(message.get("reason") or "")
            if reason:
                self.console.print(
                    f"[{tool_color(name)}]{escape(name)}[/{tool_color(name)}]"
                    f"[warn] needs confirmation: {escape(reason)}[/warn]")
            rules: list[ToolRule] = []
            for entry in message.get("suggestions") or []:
                rule = ToolRule.from_dict(entry)
                if rule is not None:
                    rules.append(rule)
            action, rule = self.ui.approval(
                rules, name=str(message.get("name") or ""),
                detail=str(message.get("detail") or ""),
                reason=str(message.get("reason") or ""))
            return {"action": action,
                    "rule": rule.to_dict() if rule is not None else None}
        if kind == "editor":
            value = self.ui.editor(str(message.get("initial") or ""))
            if value is None:
                return {"cancelled": True}
            return {"text": value}
        if kind == "copy":
            ok, detail = self.ui.copy(str(message.get("text") or ""))
            return {"ok": bool(ok), "message": detail}
        return {"cancelled": True}

    # ── completion ─────────────────────────────────────────────────────────

    def complete(self, text: str, timeout: float = 2.0) -> list[str]:
        request_id = f"c_{uuid.uuid4().hex[:12]}"
        future: Future = Future()
        self._completion[request_id] = future
        if not self.conn.send({"op": "complete", "id": request_id,
                               "text": text}):
            self._completion.pop(request_id, None)
            return []
        try:
            return list(future.result(timeout=timeout))
        except (FutureTimeout, Exception):
            return []
        finally:
            self._completion.pop(request_id, None)

    # ── turn events ────────────────────────────────────────────────────────

    def _handle_event(self, event: dict[str, Any]) -> None:
        etype = str(event.get("type") or "")
        data = dict(event.get("data") or {})
        if etype == "message.user":
            self._render_user(data)
        elif etype == "turn.started":
            self.turn_active = True
            self.busy = True
            self._cancel_sent = False
            self._guard.reset()
        elif etype == "round.started":
            self._start_round(data)
        elif etype == "message.delta":
            if self.renderer is None:
                self._start_round({})
            if self.renderer is not None:
                self.renderer.feed(str(data.get("channel") or "content"),
                                   str(data.get("text") or ""),
                                   int(data.get("tokens") or 1))
        elif etype == "message.completed":
            self._end_round(data)
        elif etype == "tool.call":
            self._flush_output()
            name = str(data.get("name") or "tool")
            args = data.get("args")
            if args is not None:
                self.console.print(tool_call_panel(name, args))
        elif etype == "tool.result":
            self._flush_output()
            name = str(data.get("name") or "tool")
            detail = data.get("detail")
            if detail is not None:
                self.console.print(tool_result_panel(
                    name, detail=str(detail), text=str(data.get("text") or ""),
                    ok=bool(data.get("ok")),
                    duration=data.get("duration") or None,
                    note=str(data.get("note") or "")))
            else:
                self.console.print(tool_result_panel(
                    name, text=str(data.get("text") or ""),
                    ok=bool(data.get("ok"))))
        elif etype == "turn.warning":
            self.console.print(f"[warn]{escape(str(data.get('message') or ''))}[/warn]")
        elif etype == "turn.error":
            self.console.print(f"[err]{escape(str(data.get('message') or ''))}[/err]")
        elif etype in {"turn.cancelled", "turn.completed", "turn.error"}:
            if etype == "turn.error":
                self.console.print(
                    f"[err]{escape(str(data.get('message') or ''))}[/err]")
            if self.renderer is not None:
                self._end_round({"cancelled": etype == "turn.cancelled",
                                 "content": data.get("content"),
                                 "thinking": data.get("thinking"),
                                 "stats": data.get("stats")})
            self.turn_active = False
            # The daemon may still be speaking/compacting before it prompts;
            # keep the turn guard armed until the prompt arrives.
            self.busy = True
            self._guard.reset()
            self._flush_output()
            if self.replaying:
                self.replaying = False
                self.deferred_questions = []
                self._show_deferred()
        elif etype == "stream.gap":
            self.console.print(
                "[warn]the replay buffer dropped some early output; "
                "continuing from the live stream[/warn]")
            self._flush_output()

    def _render_user(self, data: dict[str, Any]) -> None:
        text = str(data.get("text") or "")
        images = data.get("images") or []
        for warning in data.get("warnings") or []:
            self.console.print(f"[warn]{escape(str(warning))}[/warn]")
        if text or images:
            self.console.print(user_message_box(text, self.console))
        for image in images:
            name = image if isinstance(image, str) else (
                image.get("name") if isinstance(image, dict) else "")
            if name:
                self.console.print(f"[dim]🖼  {escape(str(name))}[/dim]")

    def _start_round(self, data: dict[str, Any]) -> None:
        s = self.settings
        mode = str(s.get("render") or "markdown")
        if mode == "auto":
            mode = "markdown" if self.console.is_terminal else "plain"
        if not self.console.is_terminal and mode == "markdown":
            mode = "plain"
        model = str(data.get("model") or s.get("model") or self.model or "")
        renderer = Renderer(
            self.console, mode=mode, math=bool(s.get("math", True)),
            nowordwrap=bool(s.get("nowordwrap", False)),
            thinking_mode=str(s.get("thinking") or "collapsed"),
            tps_footer=bool(s.get("tps", True)),
            context_used=data.get("context_used"),
            context_total=data.get("context_total"),
            show_stats=bool(s.get("show_stats", True)),
            max_render_chars=int(s.get("max_render_chars") or 12000),
            pager=bool(s.get("pager", False)), model=model,
            after_tools=bool(data.get("after_tools")),
            on_thinking_toggle=self._thinking_changed,
            # The host samples the wait (loading/waiting/reloading/prefill)
            # and ships the line in `settings`; the spinner reads it live.
            stage_provider=lambda: self.settings.get("stage"))
        self.renderer = renderer
        self.render_after_tools = bool(data.get("after_tools"))
        if renderer.mode != "raw" and not renderer.render_frames:
            self.console.print(f"[ai]{escape(model)} ›[/ai]")
        renderer.begin()

    def _end_round(self, data: dict[str, Any]) -> None:
        self._check_renderer_keys()
        renderer = self.renderer
        if renderer is None:
            return
        tool_calls = int(data.get("tool_calls") or 0)
        cancelled = bool(data.get("cancelled"))
        error = data.get("error")
        final = bool(self.render_after_tools and tool_calls == 0
                     and not cancelled and not error)
        renderer.end(final=final, stats=data.get("stats"),
                     error=str(error) if error else None,
                     cancelled=cancelled,
                     content=data.get("content"),
                     thinking=data.get("thinking"))
        self.renderer = None
        self._flush_output()

    # ── lifecycle ──────────────────────────────────────────────────────────

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.abort()
            self.renderer = None
        self.conn.close()

    def handle_resize(self) -> None:
        try:
            width, height = self.console.size
        except Exception:
            return
        self.conn.send({"op": "resize", "width": width, "height": height})


def run_client(spec: HostSpec, *, overrides: Optional[dict[str, Any]] = None,
               timeout: float = 35.0,
               output_stream: Optional[Any] = None,
               input_stream: Optional[Any] = None,
               console_obj: Any = None) -> int:
    """Ensure a host, attach, and run the terminal until detach/close."""
    paths = host_paths(spec.session_id,
                       Path(spec.host_root) if spec.host_root else None)
    try:
        popen, _spawned = ensure_host(spec, paths, timeout=timeout)
    except SessionHostUnavailable:
        raise
    try:
        conn = LineConnection.connect(paths.socket, timeout=15.0)
    except OSError as exc:
        raise SessionHostUnavailable(
            f"cannot connect to the session host: {exc}") from exc
    client = SessionClient(spec, conn, overrides=overrides, popen=popen,
                           output_stream=output_stream,
                           input_stream=input_stream,
                           console_obj=console_obj)
    try:
        client.attach()
    except SessionHostError:
        conn.close()
        if popen is not None:
            try:
                popen.terminate()
            except OSError:
                pass
        raise
    try:
        previous = None
        if threading.current_thread() is threading.main_thread():
            try:
                previous = signal.getsignal(signal.SIGWINCH)
                signal.signal(signal.SIGWINCH,
                              lambda *_: client.handle_resize())
            except (ValueError, OSError):
                previous = None
        try:
            return client.run()
        finally:
            if previous is not None:
                try:
                    signal.signal(signal.SIGWINCH, previous)
                except (ValueError, OSError):
                    pass
    finally:
        client.close()
