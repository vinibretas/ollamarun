"""Textual front end: transcript pane, rule, status bar and an always-on
input box.

The REPL is unchanged underneath: this module provides a `Ui` implementation
(`TuiUi`) for the prompts, a `StreamSink` that draws the same live frames
`rich.Live` would, and an `App` that owns the layout. Everything the rest of
the codebase prints through the shared `console` singleton lands in the
transcript pane: `TranscriptWriter` is a file-like sink that Rich writes ANSI
into, and the app turns each written block into a transcript widget.

Enable it with `--ui tui` (config `"ui": "tui"`); the inline prompt_toolkit
REPL stays the default and is the fallback when Textual is not installed.
`OLLAMARUN_TUI_FULLSCREEN=1` runs the same app on the alternate screen instead
of inline (the transcript then disappears on exit instead of staying in the
terminal), and `OLLAMA_TUI_DISABLE` keeps `--ui tui` on the inline REPL.
"""
from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Optional, Sequence

from rich.markup import escape
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from .constants import HAS_PROMPT_TOOLKIT, HISTORY_PATH, console
from .helpers import human_duration, shortcut
from .renderer import GenerationResult, Renderer, StreamSink
from .repl import Repl, completion_candidates, completion_fragment
from .ui import (Ui, approval_entries, approval_lines, format_status,
                 spinner_frame)

# Commands that only look at state or change display knobs: they run straight
# away even while the model is working. Everything else queues (it may rewrite
# the conversation the running turn is built on).
IMMEDIATE_COMMANDS = {
    "/help", "/?", "/thinking", "/tps", "/render", "/math", "/pager", "/stats",
    "/context", "/queue", "/ps", "/show", "/tools", "/tts", "/params", "/tag",
}
QUIT_COMMANDS = {"/bye", "/exit"}
STOP_WINDOW = 2.0           # seconds between the two Esc presses
QUIT_WINDOW = 3.0           # seconds between the two Ctrl+Q presses
DONE_LINGER = 4.0           # how long "done in 3.2s · 41 tok/s" stays up
LOG = logging.getLogger("ollamarun.tui")


class TranscriptWriter:
    """A `rich.Console` target that forwards written blocks to the app.

    Rich only needs `write`/`flush`; claiming to be a tty keeps the shared
    console rendering styles, tables and boxes exactly as it does inline.
    """

    def __init__(self, app: "OllamarunTui"):
        self.app = app
        self.lock = threading.Lock()
        self.buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self.lock:
            self.buffer += text
            while "\n" in self.buffer:
                block, self.buffer = self.buffer.split("\n", 1)
                self.app.post_transcript(block)
        return len(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return True


class AnswerView(Static):
    """The answer currently streaming (markdown + 2D math), updated in place."""


class PromptInput(Input):
    """Input box with history (Up/Down) and Tab completion.

    `history` is the same `~/etc/data/ollamarun/history` prompt_toolkit keeps,
    and completion mirrors `SlashCompleter` through `completion_candidates`.
    """

    BINDINGS = [
        Binding("up", "history_prev", "previous", show=False),
        Binding("down", "history_next", "next", show=False),
        Binding("tab", "complete", "complete", show=False),
    ]

    def __init__(self, repl: Repl, **kwargs: Any):
        super().__init__(placeholder="message or /command — Enter queues while busy",
                         **kwargs)
        self.repl = repl
        self.history: list[str] = []
        self.history_index: Optional[int] = None
        self._completion: Optional[tuple[str, list[str], int]] = None
        self._load_history()

    def _history_file(self):
        if os.environ.get("OLLAMA_NOHISTORY") or not HAS_PROMPT_TOOLKIT:
            return None
        try:
            from prompt_toolkit.history import FileHistory

            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            return FileHistory(str(HISTORY_PATH))
        except Exception:
            return None

    def _load_history(self) -> None:
        handle = self._history_file()
        if handle is None:
            return
        try:
            self.history = list(handle.load_history_strings())
        except Exception:
            self.history = []

    def remember(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.history_index = None
        self._completion = None
        if self.history and self.history[0] == text:
            return
        self.history.insert(0, text)
        handle = self._history_file()
        if handle is not None:
            try:
                handle.append_string(text)
            except Exception:
                pass

    def _set_input(self, value: str) -> None:
        self.value = value
        self.cursor_position = len(value)

    # ── keys ────────────────────────────────────────────────────────────────

    def action_history_prev(self) -> None:
        if not self.history:
            return
        index = 0 if self.history_index is None else min(self.history_index + 1,
                                                         len(self.history) - 1)
        self.history_index = index
        self._set_input(self.history[index])

    def action_history_next(self) -> None:
        if self.history_index is None:
            return
        index = self.history_index - 1
        if index < 0:
            self.history_index = None
            self._set_input("")
            return
        self.history_index = index
        self._set_input(self.history[index])

    def action_complete(self) -> None:
        """Complete the current token; repeated Tab presses cycle the matches."""
        if self._completion is not None:
            stem, candidates, index = self._completion
            if self.value == stem + candidates[index]:
                index = (index + 1) % len(candidates)
                self._completion = (stem, candidates, index)
                self._set_input(stem + candidates[index])
                self._completion_hint(candidates)
                return
        before = self.value[: self.cursor_position]
        fragment = completion_fragment(before)
        stem = before[: len(before) - len(fragment)]
        candidates = [item for item in completion_candidates(self.repl, before)
                      if item != fragment]
        if not candidates:
            self._completion = None
            return
        self._completion = (stem, candidates, 0)
        self._set_input(stem + candidates[0])
        self._completion_hint(candidates)

    def _completion_hint(self, candidates: list[str]) -> None:
        if len(candidates) > 1:
            self.app.show_hint(f"{len(candidates)} matches — Tab cycles")


class ConfirmScreen(ModalScreen[bool]):
    """y/n question (y runs the action, Esc means the default)."""

    BINDINGS = [
        Binding("y", "yes", "yes", show=False),
        Binding("n", "no", "no", show=False),
        Binding("escape", "default", "cancel", show=False),
        Binding("enter", "default", "cancel", show=False),
    ]

    def __init__(self, question: str, default: bool = False):
        super().__init__()
        self.question = question
        self.default = default

    def compose(self) -> ComposeResult:
        default = "y" if self.default else "n"
        yield Static(f"{self.question}\n\n[y] yes · [n] no · [dim]default {default}[/dim]",
                     id="modal-question")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)

    def action_default(self) -> None:
        self.dismiss(self.default)


class AskScreen(ModalScreen[Optional[str]]):
    """Free-text (or fixed-choice) question returning what was typed."""

    BINDINGS = [Binding("escape", "cancel", "cancel", show=False)]

    def __init__(self, question: str, choices: Optional[Sequence[str]] = None,
                 default: Optional[str] = None):
        super().__init__()
        self.question = question
        self.choices = list(choices or [])
        self.default = default

    def compose(self) -> ComposeResult:
        hint = ""
        if self.choices:
            hint = "\n[dim]choices: " + " · ".join(self.choices) + "[/dim]"
        elif self.default:
            hint = f"\n[dim]default {self.default}[/dim]"
        yield Static(f"{self.question}{hint}", id="modal-question")
        yield Input(value=self.default or "", id="modal-input")

    def on_mount(self) -> None:
        self.query_one("#modal-input", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(self.default)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "modal-input":
            return
        answer = event.value.strip()
        if self.choices and answer not in self.choices:
            self.query_one("#modal-question", Static).update(
                f"{self.question}\n[err]pick one of: {' · '.join(self.choices)}[/err]")
            return
        self.dismiss(answer or self.default)


class ApprovalScreen(ModalScreen[Optional[int]]):
    """The tool-approval question: the same numbered choices as the console
    prompt, picked by typing the number/letter or with up/down + enter."""

    BINDINGS = [
        Binding("enter", "choose_cursor", "choose", show=False),
        Binding("up", "cursor(-1)", "up", show=False),
        Binding("k", "cursor(-1)", "up", show=False),
        Binding("down", "cursor(1)", "down", show=False),
        Binding("j", "cursor(1)", "down", show=False),
        Binding("y", "choose_letter('y')", "run once", show=False),
        Binding("a", "choose_letter('a')", "always", show=False),
        Binding("A", "choose_letter('A')", "allow all", show=False),
        Binding("n", "choose_letter('n')", "deny", show=False),
        Binding("escape", "deny", "deny", show=False),
    ]

    def __init__(self, tool_name: str, detail: str, reason: str,
                 tags: list[str], labels: list[str]):
        super().__init__()
        # `name` is a read-only Textual widget property -- keep the tool's name
        # under a different attribute.
        self.tool_name = tool_name
        self.detail = detail
        self.reason = reason
        self.tags = tags
        self.labels = labels
        self.cursor = 0

    def compose(self) -> ComposeResult:
        yield Static("", id="modal-question")

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        lines = [f"[warn]{self.tool_name} needs confirmation:[/warn] {self.reason}"]
        if self.detail:
            lines.append("")
            lines.extend(self.detail.splitlines()[:8])
        lines.append("")
        for index, (tag, label) in enumerate(zip(self.tags, self.labels)):
            pointer = "❯" if index == self.cursor else " "
            style = "bold" if index == self.cursor else "dim"
            lines.append(f"[{style}]{pointer} [key]{tag:<4}[/key] "
                         f"{escape(label)}[/{style}]")
        lines.append("")
        lines.append("[dim]type the number or letter · enter picks the highlighted "
                     "line[/dim]")
        self.query_one("#modal-question", Static).update("\n".join(lines))

    # ── keys ────────────────────────────────────────────────────────────────

    def on_key(self, event: events.Key) -> None:
        if event.character and event.character.isdigit():
            index = int(event.character) - 1
            if 0 <= index < len(self.labels):
                self.cursor = index
                self._refresh()
                event.stop()

    def action_cursor(self, delta: int) -> None:
        self.cursor = (self.cursor + delta) % max(1, len(self.labels))
        self._refresh()

    def action_choose_cursor(self) -> None:
        self.dismiss(self.cursor)

    def action_choose_letter(self, letter: str) -> None:
        for index, tag in enumerate(self.tags):
            if tag.endswith("/" + letter):
                self.dismiss(index)
                return

    def action_deny(self) -> None:
        self.dismiss(len(self.labels) - 1 if self.labels else None)


class TuiStreamSink(StreamSink):
    """Feeds the streaming frames to the app as fast as it can consume them."""

    def __init__(self, app: "OllamarunTui"):
        self.app = app
        self.lock = threading.Lock()
        self.frame: Any = None
        self.posted = False
        self.active = False

    def start(self, renderer: Renderer) -> None:
        with self.lock:
            self.active = True
            self.frame = None
        self.app.post_stream("start")

    def update(self, renderable: Any) -> None:
        with self.lock:
            if not self.active:
                return
            self.frame = renderable
            if self.posted:
                return  # a frame is already on its way to the UI thread
            self.posted = True
        self.app.post_stream("frame")

    def finish(self, result: GenerationResult) -> None:
        with self.lock:
            self.active = False
            self.frame = None
        self.app.post_stream("end")

    def take_frame(self) -> Any:
        """Hand the newest frame to the UI thread (None when none is pending)."""
        with self.lock:
            frame = self.frame
            self.frame = None
            if frame is None:
                self.posted = False
            return frame


class TuiUi(Ui):
    """`Ui` implementation backed by the app's modal screens."""

    def __init__(self, app: "OllamarunTui"):
        self.app = app

    def notify(self, event: str, **data: Any) -> None:
        self.app.notify_event(event, **data)

    def ask(self, question: str, *, choices: Optional[Sequence[str]] = None,
            default: Optional[str] = None) -> Optional[str]:
        return self.app.run_modal(AskScreen(question, choices, default))

    def confirm(self, question: str, *, default: bool = False) -> bool:
        return bool(self.app.run_modal(ConfirmScreen(question, default)))

    def approval(self, suggestions: list[Any], *,
                 name: str = "", detail: str = "", reason: str = ""
                 ) -> tuple[str, Optional[Any]]:
        """Ask in a modal offering `suggestions` (see `ui.approval_entries`)."""
        entries = approval_entries(suggestions)
        lines = approval_lines(entries)
        index = self.app.run_modal(ApprovalScreen(
            name or "tool call", detail, reason,
            [tag for tag, _label in lines], [label for _tag, label in lines]))
        if not isinstance(index, int) or not 0 <= index < len(entries):
            index = len(entries) - 1  # deny
        action, rule, _label = entries[index]
        return action, rule

    def editor(self, initial: str = "") -> Optional[str]:
        return self.app.run_editor(initial)


class OllamarunTui(App[None]):
    """The chat front end: transcript, rule, status bar, input."""

    CSS = """
    /* An explicit height keeps inline mode from shrinking the app to its
       fixed-height children (the transcript is the 1fr pane). */
    Screen { layout: vertical; height: 100%; }
    #transcript { height: 1fr; padding: 0 1; }
    #status { height: 2; background: $panel; color: $text; padding: 0 1;
              border-top: solid $accent; }
    #prompt { height: 3; }
    ModalScreen { align: center middle; background: $background 60%; }
    #modal-question { width: 80%; max-width: 100; padding: 1 2; border: round $accent;
                      background: $surface; }
    #modal-input { width: 80%; max-width: 100; margin-top: 1; }
    AnswerView { padding: 0 0 1 0; }
    Static { height: auto; }
    """

    BINDINGS = [
        Binding("escape", "stop_or_clear", "stop / clear", priority=True, show=False),
        Binding("ctrl+b", "send_now", "send now", priority=True, show=False),
        Binding("ctrl+t", "toggle_thinking", "thinking", priority=True, show=False),
        Binding("ctrl+g", "editor", "editor", priority=True, show=False),
        Binding("ctrl+q", "quit_app", "quit", priority=True, show=False),
        Binding("ctrl+c", "interrupt", "cancel / quit", priority=True, show=False),
        Binding("ctrl+l", "clear_view", "clear view", priority=True, show=False),
        Binding("pgup", "scroll_up_page", "scroll up", priority=False, show=False),
        Binding("pgdown", "scroll_down_page", "scroll down", priority=False, show=False),
    ]

    def __init__(self, repl: Repl):
        super().__init__()
        self.repl = repl
        self.writer = TranscriptWriter(self)
        self.sink = TuiStreamSink(self)
        self._busy = False
        self._stop_armed_at = 0.0
        self._quit_armed_at = 0.0
        self._hint = ""
        self._hint_until = 0.0
        self._loaded_state = False
        self._loaded_models = False
        self._ui_thread = threading.get_ident()
        self._turn_began = time.monotonic()
        self._answer_view: Optional[AnswerView] = None
        self._done_timer: Optional[Any] = None
        self._console_file: Any = None
        self._console_width: Optional[int] = None

    # ── layout ──────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="transcript")
        yield Static("", id="status")
        yield PromptInput(self.repl, id="prompt")

    def on_mount(self) -> None:
        self.repl.tui_mode = True
        self.repl.stream_sink = self.sink
        # Textual redirects stdout while it runs, so remember "stdout" as None
        # (rich's own default) rather than the redirected object: lines printed
        # after the app exits (session saved, model unloaded) must land on the
        # real terminal.
        current = console.file
        self._console_file = None if current in (sys.stdout, sys.stderr) else current
        self._console_width = getattr(console, "width", None)
        console.file = self.writer
        self._sync_console_width()
        self.set_interval(0.1, self._refresh_status)
        self.set_interval(0.25, self._poll_background)
        self.repl._banner()
        self.repl._context_warning()
        self.query_one("#prompt", PromptInput).focus()
        self._start_background_work()

    def _start_background_work(self) -> None:
        repl = self.repl
        if repl.session.model:
            repl.set_state("loading", f"waiting for {repl.session.model}")
        repl._preload_model()

        def _warm() -> None:
            try:
                repl.refresh_models()
                self.call_from_thread(setattr, self, "_loaded_models", True)
                repl.context_length(refresh=True)
                self.call_from_thread(setattr, self, "_loaded_state", True)
            except Exception:
                LOG.debug("background warm-up interrupted", exc_info=True)

        threading.Thread(target=_warm, daemon=True).start()

    def _poll_background(self) -> None:
        """Pick up the preload/model-list threads finishing without them
        having to talk to the app."""
        repl = self.repl
        if repl.state == "loading":
            thread = repl._preload_thread
            if thread is None or not thread.is_alive():
                repl.set_state("ready")
        if self._loaded_state and self._hint.startswith("waiting"):
            self.clear_hint()

    def _sync_console_width(self) -> None:
        """Keep Rich's wrapping in step with the transcript pane."""
        try:
            console.width = max(20, self.size.width - 2)
        except Exception:
            pass

    def on_resize(self, event: events.Resize) -> None:
        self._sync_console_width()

    def on_unmount(self) -> None:
        # Stop any turn that is still streaming before the terminal goes back.
        self.repl.cancel_token.cancel()
        try:
            self.repl.client.abort_stream()
        except Exception:
            LOG.debug("abort at shutdown failed", exc_info=True)
        if self._done_timer is not None:
            self._done_timer.stop()
        console.file = self._console_file  # None means rich's default (stdout)
        if self._console_width:
            try:
                console.width = self._console_width
            except Exception:
                LOG.debug("console width restore failed", exc_info=True)
        self.repl.tui_mode = False
        self.repl.stream_sink = None

    # ── transcript ──────────────────────────────────────────────────────────

    def post_transcript(self, block: str) -> None:
        """Append one printed block (called from any thread)."""
        if self._on_ui_thread():
            self._mount_transcript(block)
            return
        try:
            self.call_from_thread(self._mount_transcript, block)
        except Exception:
            LOG.debug("transcript write dropped", exc_info=True)

    def _mount_transcript(self, block: str) -> None:
        try:
            container = self.query_one("#transcript", VerticalScroll)
        except Exception:
            return
        widget = Static(Text.from_ansi(block) if block else Text(""), markup=False)
        live = self._answer_view
        if live is not None and live.is_mounted:
            # Keep printed output (user bubbles, tool panels, warnings) in the
            # order it happened: just above the answer that is still streaming.
            container.mount(widget, before=live)
        else:
            container.mount(widget)
        self._follow()

    def _follow(self) -> None:
        try:
            self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    def post_stream(self, event: str) -> None:
        if self._on_ui_thread():
            self._apply_stream(event)
            return
        try:
            self.call_from_thread(self._apply_stream, event)
        except Exception:
            LOG.debug("stream frame dropped", exc_info=True)

    def _apply_stream(self, event: str) -> None:
        try:
            container = self.query_one("#transcript", VerticalScroll)
        except Exception:
            return
        if event != "frame":
            # A new response starts, or the finished one stops streaming: in
            # both cases printed output goes back to the end of the transcript.
            self._answer_view = None
            with self.sink.lock:
                self.sink.frame = None
                self.sink.posted = False
            return
        while True:
            frame = self.sink.take_frame()
            if frame is None:
                return
            if self._answer_view is None or not self._answer_view.is_mounted:
                self._answer_view = AnswerView(frame)
                container.mount(self._answer_view)
            else:
                self._answer_view.update(frame)
            self._follow()

    # ── input ───────────────────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt" or isinstance(self.screen, ModalScreen):
            return
        repl = self.repl
        value = event.value.strip()
        if not value and not repl.pending_images:
            return
        event.input.value = ""
        prompt = self.query_one("#prompt", PromptInput)
        if value:
            prompt.remember(value)
        if value.startswith("/"):
            name = value.split()[0]
            if name in QUIT_COMMANDS:
                self.action_quit_app()
                return
            if self._busy and name not in IMMEDIATE_COMMANDS:
                repl.queue_command(value)
                self.show_hint(f"queued {name} · runs at the end", seconds=6)
                return
            self._dispatch_command(value, immediate=self._busy)
            return
        if self._busy:
            repl.queue_message(value)
            self.show_hint("queued · Ctrl+B sends", seconds=6)
            return
        self.start_turn(value)

    def _dispatch_command(self, line: str, *, immediate: bool = False) -> None:
        if immediate:
            self._work_immediate(line)
        else:
            self._work_command(line)

    @work(thread=True, exclusive=True, group="turn")
    def _work_command(self, line: str) -> None:
        self.call_from_thread(self._enter_busy)
        try:
            self.call_from_thread(self.repl.set_state, "running", line)
            keep_going = bool(self.repl._dispatch(line))
        except Exception as exc:  # a broken command must not kill the app
            self.call_from_thread(self._report_error, exc)
            keep_going = True
        self.call_from_thread(self._leave_busy)
        if not keep_going:
            self.call_from_thread(self.exit, 0)

    @work(thread=True, group="side")
    def _work_immediate(self, line: str) -> None:
        try:
            self.repl._dispatch(line)
        except Exception as exc:
            self.call_from_thread(self._report_error, exc)
        self.call_from_thread(self.repl.sync_live_renderer)

    # ── turns ───────────────────────────────────────────────────────────────

    def start_turn(self, text: Optional[str]) -> None:
        self._work_turn(text)

    @work(thread=True, exclusive=True, group="turn")
    def _work_turn(self, text: Optional[str]) -> None:
        repl = self.repl
        self.call_from_thread(self._enter_busy)
        try:
            repl.cancel_token.reset()
            result = repl.send(text) if text is not None else repl._generate()
        except Exception as exc:
            self.call_from_thread(self._report_error, exc)
            result = GenerationResult(error=str(exc))
        self.call_from_thread(self._leave_busy)
        self.call_from_thread(self._finish_turn, result)

    def _enter_busy(self) -> None:
        self._busy = True
        self._hint = ""
        self._turn_began = time.monotonic()
        self.repl.cancel_token.reset()
        self.repl.set_state("generating")

    def _leave_busy(self) -> None:
        self._busy = False
        self.sink.posted = False

    def _finish_turn(self, result: GenerationResult) -> None:
        self._busy = False
        repl = self.repl
        elapsed = time.monotonic() - getattr(self, "_turn_began", time.monotonic())
        rate = repl.current_tps()
        if result.error:
            repl.set_state("error", shortcut(result.error, 40))
        elif result.cancelled:
            repl.set_state("done", "stopped — partial answer kept")
        else:
            bits = [f"done in {human_duration(elapsed)}" if elapsed else "done"]
            if rate:
                bits.append(f"{rate:.1f} tok/s")
            repl.set_state("done", " · ".join(bits))
        # /api/show is a network hop (possibly to another machine): refresh the
        # context length off the UI thread so the status bar never stalls.
        threading.Thread(target=repl.context_length, kwargs={"refresh": True},
                         daemon=True).start()
        self._done_timer = self.set_timer(DONE_LINGER, self._clear_done)
        self._process_queue()
    def _clear_done(self) -> None:
        try:
            if self.repl.state == "done":
                self.repl.set_state("ready")
        except Exception:
            LOG.debug("done-state clear skipped", exc_info=True)

    def _process_queue(self) -> None:
        """Run whatever was typed while the last turn worked, in order."""
        if self._busy:
            return
        item = self.repl.queue.pop_next()
        if item is None:
            return
        if item.kind == "command":
            self._work_command(item.text)
        else:
            self.start_turn(item.text)

    # ── keys ────────────────────────────────────────────────────────────────

    def action_stop_or_clear(self) -> None:
        prompt = self.query_one("#prompt", PromptInput)
        if not self._busy:
            self._stop_armed_at = 0.0
            if prompt.value:
                prompt.value = ""
            return
        now = time.monotonic()
        if self._stop_armed_at and now - self._stop_armed_at <= STOP_WINDOW:
            self._stop_armed_at = 0.0
            self._cancel_turn("stopped")
            return
        self._stop_armed_at = now
        self.show_hint("Esc again to stop", seconds=STOP_WINDOW)

    def action_send_now(self) -> None:
        prompt = self.query_one("#prompt", PromptInput)
        text = prompt.value.strip()
        if not text:
            return
        prompt.remember(text)
        prompt.value = ""
        if self._busy:
            self.repl.queue_message(text)
            self._cancel_turn("sending your message now")
            self.show_hint("sending now, tool keeps running")
        else:
            self.start_turn(text)

    def action_toggle_thinking(self) -> None:
        mode = self.repl.toggle_thinking()
        self.show_hint(f"thinking {mode}")

    def action_editor(self) -> None:
        prompt = self.query_one("#prompt", PromptInput)
        edited = self.run_editor(prompt.value)
        if edited is not None:
            prompt.value = edited
            prompt.cursor_position = len(edited)

    def action_interrupt(self) -> None:
        prompt = self.query_one("#prompt", PromptInput)
        if self._busy:
            self._cancel_turn("cancelled")
            return
        if prompt.value:
            prompt.value = ""
            return
        self.action_quit_app()

    def action_quit_app(self) -> None:
        repl = self.repl
        pending = len(repl.queue)
        if (self._busy or pending) and not self._recent_quit():
            detail = []
            if self._busy:
                detail.append("the model is still working")
            if pending:
                detail.append(f"{pending} queued item(s) would be dropped")
            self.show_hint("Ctrl+Q again to quit " + ", ".join(detail),
                           seconds=QUIT_WINDOW)
            return
        self.exit(0)

    def action_clear_view(self) -> None:
        container = self.query_one("#transcript", VerticalScroll)
        for child in list(container.children):
            child.remove()
        self._answer_view = None

    def action_scroll_up_page(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_page_up(animate=False)

    def action_scroll_down_page(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_page_down(animate=False)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _on_ui_thread(self) -> bool:
        return threading.get_ident() == self._ui_thread

    def _recent_quit(self) -> bool:
        now = time.monotonic()
        armed = bool(self._quit_armed_at) and now - self._quit_armed_at <= QUIT_WINDOW
        self._quit_armed_at = 0.0 if armed else now
        return armed

    def _cancel_turn(self, why: str) -> None:
        self.repl.cancel_token.cancel()
        self.repl.client.abort_stream()
        self.repl.set_state("stopping", why)

    def show_hint(self, text: str, *, seconds: float = 4.0) -> None:
        self._hint = text
        self._hint_until = time.monotonic() + seconds

    def clear_hint(self) -> None:
        self._hint = ""
        self._hint_until = 0.0

    def notify_event(self, event: str, **data: Any) -> None:
        """`Ui.notify` from the REPL (already on whichever thread called it)."""
        if event == "queued_delivered":
            count = data.get("count", 0)
            if count:
                self.show_hint("message handed to the model")

    def _report_error(self, exc: BaseException) -> None:
        self.repl.set_state("error", shortcut(str(exc), 40))
        console.print(f"[err]{type(exc).__name__}: {exc}[/err]")

    # ── modality ────────────────────────────────────────────────────────────

    def run_modal(self, screen: ModalScreen[Any]) -> Any:
        """Push `screen` and block until it is dismissed.

        Never call this from the UI thread: the modal needs that thread to
        run, so callers are the REPL workers (`#tools`/commands).
        """
        if self._on_ui_thread():
            # Nothing on the UI thread asks questions today; refuse loudly
            # rather than deadlocking.
            raise RuntimeError("modal prompts must run off the UI thread")
        box: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self.call_from_thread(self.push_screen, screen, box.put)
        return box.get()

    def run_editor(self, initial: str = "") -> Optional[str]:
        """`$EDITOR` handoff: hand the terminal over, then take it back."""
        if not self._on_ui_thread():
            return self.call_from_thread(self.run_editor, initial)
        from .interactive import open_in_editor

        with self.suspend():
            return open_in_editor(initial)

    # ── status bar ──────────────────────────────────────────────────────────

    def _refresh_status(self) -> None:
        repl = self.repl
        snapshot = repl.status_snapshot()
        hint = self._hint if time.monotonic() < self._hint_until else ""
        if not hint and self._busy and not self._stop_armed_at:
            hint = "Esc Esc stop · Ctrl+B send"
        frame = spinner_frame(snapshot.state, time.monotonic())
        line = format_status(snapshot, frame=frame, width=max(20, self.size.width - 2),
                             hint=hint)
        try:
            self.query_one("#status", Static).update(line)
        except Exception:
            LOG.debug("status refresh skipped", exc_info=True)  # app is closing


def tui_available() -> tuple[bool, str]:
    """(usable, reason) for the Textual front end on this machine."""
    try:
        import textual
    except ImportError:
        return False, "the textual package is not installed (pip install --user textual)"
    version = getattr(textual, "__version__", "?")
    try:
        major = int(str(version).split(".")[0])
    except ValueError:
        major = 0
    if major < 1:
        return False, f"textual {version} is too old for the TUI (need 1.0+)"
    if not console.is_terminal:
        return False, "not a terminal"
    if os.environ.get("OLLAMA_TUI_DISABLE"):
        return False, "OLLAMA_TUI_DISABLE is set"
    return True, ""


def run_tui(repl: Repl) -> int:
    """Run the Textual front end for `repl` and return the exit code."""
    app = OllamarunTui(repl)
    repl.ui = TuiUi(app)
    kwargs: dict[str, Any] = {}
    if os.environ.get("OLLAMARUN_TUI_FULLSCREEN"):
        kwargs = {}
    else:
        kwargs = {"inline": True, "inline_no_clear": True}
    try:
        app.run(**kwargs)
    finally:
        if repl.unload_on_exit:
            repl.unload_model()
        if not repl.no_save:
            repl.store.save(repl.session)
            console.print(f"[dim]session {repl.session.id} saved · {repl.store.dir}[/dim]")
        else:
            console.print("[dim]--no-save: session not persisted[/dim]")
    return 0
