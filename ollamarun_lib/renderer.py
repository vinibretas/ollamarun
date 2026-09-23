"""Streaming renderer: drives a `rich.Live` view while chat events stream in.

`Renderer` is the piece that talks to the terminal frame-by-frame -- markdown
re-render while tokens stream, the collapsible thinking panel (Ctrl+T pager
handoff) and `TpsEstimator` for the live `~tok/s` footer. `user_message_box`/
`model_message_panel` are the chat-bubble helpers shared with transcript
replay (`render_transcript`, used by `/show`/`--show`).
"""
from __future__ import annotations

import json
import os
import select
import sys
import termios
import time
import tty
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Optional

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.text import Text

from .constants import CTRL_T, THINKING_MODES
from .helpers import human_chars, human_duration, short_stamp
from .mathrender import render_content, render_thinking

if TYPE_CHECKING:
    from .session import Session


@dataclass
class GenerationResult:
    content: str = ""
    thinking: str = ""
    cancelled: bool = False
    stats: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class TpsEstimator:
    """Live tokens/sec from streamed chunks (~1 chunk per generated token).

    Uses a sliding window so the rate reacts to speed changes, falling back
    to the cumulative average when the window is too fresh.
    """

    def __init__(self, window: float = 5.0):
        self.window = window
        self.samples: deque[tuple[float, int]] = deque()
        self.tokens = 0
        self.started_at: Optional[float] = None

    def add(self, count: int = 1, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        if self.started_at is None:
            self.started_at = now
        self.tokens += count
        self.samples.append((now, self.tokens))
        horizon = now - self.window
        while len(self.samples) > 2 and self.samples[0][0] < horizon:
            self.samples.popleft()

    def rate(self, now: Optional[float] = None) -> Optional[float]:
        if self.started_at is None or self.tokens == 0:
            return None
        now = time.monotonic() if now is None else now
        horizon = now - self.window
        while len(self.samples) > 2 and self.samples[0][0] < horizon:
            self.samples.popleft()
        base_time, base_tokens = self.samples[0]
        if len(self.samples) > 2 and now - base_time >= 1.0 and self.tokens > base_tokens:
            return (self.tokens - base_tokens) / (now - base_time)
        elapsed = now - self.started_at
        return self.tokens / elapsed if elapsed > 0 else None


def stats_from_final(chunk: dict[str, Any]) -> dict[str, Any]:
    def seconds(key: str) -> Optional[float]:
        value = chunk.get(key)
        return value / 1e9 if isinstance(value, (int, float)) else None

    eval_count = chunk.get("eval_count")
    eval_duration = seconds("eval_duration")
    prompt_count = chunk.get("prompt_eval_count")
    prompt_duration = seconds("prompt_eval_duration")
    stats: dict[str, Any] = {
        "total_duration": seconds("total_duration"),
        "load_duration": seconds("load_duration"),
        "prompt_tokens": prompt_count,
        "completion_tokens": eval_count,
        "prompt_duration": prompt_duration,
        "eval_duration": eval_duration,
        "done_reason": chunk.get("done_reason"),
    }
    if eval_count and eval_duration:
        stats["eval_tps"] = eval_count / eval_duration
    if prompt_count and prompt_duration:
        stats["prompt_tps"] = prompt_count / prompt_duration
    return stats


def format_stats(stats: dict[str, Any]) -> str:
    bits = []
    if stats.get("load_duration") is not None:
        bits.append(f"load {human_duration(stats['load_duration'])}")
    if stats.get("prompt_tokens") is not None:
        rate = stats.get("prompt_tps")
        suffix = f" ({rate:.0f} tok/s)" if rate else ""
        bits.append(f"prompt {stats['prompt_tokens']} tok{suffix}")
    if stats.get("completion_tokens") is not None:
        rate = stats.get("eval_tps")
        suffix = f" ({rate:.1f} tok/s)" if rate else ""
        bits.append(f"eval {stats['completion_tokens']} tok{suffix}")
    if stats.get("total_duration") is not None:
        bits.append(f"total {human_duration(stats['total_duration'])}")
    return " · ".join(bits)


def thinking_panel(text: str, title: str = "thinking",
                   preview_lines: Optional[int] = None) -> Panel:
    """Dim thinking block rendered like the answer; preview_lines keeps the tail."""
    return Panel(render_thinking(text, preview_lines=preview_lines), title=title,
                 border_style="dim", title_align="left", padding=(0, 1))


def user_message_box(text: str, console: Console) -> Align:
    """Small right-aligned box for a user turn."""
    lines = text.splitlines() or [""]
    content_width = max(len(line) for line in lines)
    width = max(12, min(content_width + 4, (console.width or 80) - 4, 72))
    panel = Panel(Text(text), border_style="you", padding=(0, 1), width=width,
                 title="you", title_align="right")
    return Align.right(panel)


def model_message_panel(renderable: Any, model: str) -> Panel:
    """Full-width box for a model turn."""
    return Panel(renderable, border_style="ai", padding=(0, 1),
                title=f"{model} ›" if model else "assistant", title_align="left")


class Renderer:
    """Streams NDJSON chat events to the terminal and returns the transcript."""

    def __init__(self, console: Console, *, mode: str = "markdown", math: bool = True,
                 nowordwrap: bool = False, thinking_mode: str = "collapsed",
                 tps_footer: bool = True, show_stats: bool = False, verbose: bool = False,
                 max_render_chars: int = 12000, pager: bool = False, model: str = ""):
        self.console = console
        self.mode = mode
        self.model = model
        self.math = math
        self.nowordwrap = nowordwrap
        self.thinking_mode = thinking_mode if thinking_mode in THINKING_MODES else "collapsed"
        self.tps_footer = tps_footer
        self.estimator = TpsEstimator()
        self.show_stats = show_stats
        self.verbose = verbose
        self.max_render_chars = max_render_chars
        self.pager = pager
        self.use_pager = bool(pager) and console.is_terminal and mode != "raw"
        self.live_render = mode == "markdown" and console.is_terminal and not self.use_pager
        self.interactive = False
        try:
            self.interactive = (self.live_render and sys.stdin.isatty())
        except (ValueError, OSError):
            self.interactive = False
        self._pager_request = False

    def thinking_visible(self) -> bool:
        return self.thinking_mode != "hidden"

    def _toggle_thinking(self) -> None:
        self.thinking_mode = "collapsed" if self.thinking_mode == "expanded" else "expanded"

    def _poll_keys(self, fd: int) -> bool:
        """Reads pending stdin keys; True when Ctrl+T changed the view.

        Ctrl+T on a collapsed view expands it and sets `_pager_request` so the
        stream loop opens the full thinking in the pager; on an expanded view
        it collapses back.
        """
        changed = False
        try:
            while select.select([fd], [], [], 0)[0]:
                data = os.read(fd, 64)
                if not data:
                    break
                if CTRL_T in data:
                    if self.thinking_mode == "expanded":
                        self.thinking_mode = "collapsed"
                    else:
                        self.thinking_mode = "expanded"
                        self._pager_request = True
                    changed = True
        except (OSError, ValueError):
            pass
        return changed

    def _tps_text(self) -> Optional[Text]:
        if not self.tps_footer:
            return None
        rate = self.estimator.rate()
        if not rate or not self.estimator.tokens:
            return None
        return Text(f"~{rate:.1f} tok/s · {self.estimator.tokens} tok", style="dim")

    def _thinking_preview_lines(self) -> int:
        height = getattr(self.console, "height", None) or 24
        return max(5, height - 10)

    def render_final(self, content: str) -> None:
        if self.mode == "raw" or not content:
            return
        if self.mode == "markdown":
            body = self._markdown(content)
            if self.live_render:
                self.console.print(model_message_panel(body, self.model),
                                   soft_wrap=self.nowordwrap)
            else:
                self.console.print(body, soft_wrap=self.nowordwrap)
        else:
            self.console.print(escape(content), soft_wrap=self.nowordwrap)

    def _markdown(self, content: str) -> Group:
        width = max(20, (self.console.width or 80) - 4)
        return render_content(content, math=self.math, width=width)

    def _show_thinking_pager(self, text: str) -> None:
        """Full thinking in the system pager (scroll/search); stream pauses."""
        try:
            with self.console.pager(styles=True):
                self.console.print(Rule("[dim]thinking · leave the pager to resume[/dim]"))
                self.console.print(render_thinking(text))
        except Exception as exc:  # pager failures must not kill the stream
            self.console.print(f"[warn]pager failed: {escape(str(exc))}[/warn]")

    def stream(self, events: Iterator[dict[str, Any]]) -> GenerationResult:
        result = GenerationResult()
        use_live = self.live_render
        live: Optional[Live] = None
        last_refresh = 0.0
        printed_thinking_header = False
        fd: Optional[int] = None
        saved_term = None
        if self.interactive:
            try:
                fd = sys.stdin.fileno()
                saved_term = termios.tcgetattr(fd)
                tty.setcbreak(fd)
            except (OSError, ValueError, termios.error):
                fd = None
                saved_term = None

        def refresh(interval: float, force: bool = False) -> None:
            nonlocal last_refresh
            if live is None:
                return
            now = time.monotonic()
            if force or now - last_refresh >= interval:
                live.update(self._live_view(result), refresh=True)
                last_refresh = now

        def handle_keys() -> None:
            nonlocal live
            if live is None or fd is None:
                return
            if not self._poll_keys(fd):
                return
            if self._pager_request:
                self._pager_request = False
                if result.thinking:
                    live.stop()
                    self._show_thinking_pager(result.thinking)
                    live.start()
            refresh(0.0, force=True)

        try:
            if use_live:
                live = Live(self._live_view(result), console=self.console,
                            refresh_per_second=10, vertical_overflow="visible")
                live.start()
            try:
                for event in events:
                    if event.get("error"):
                        result.error = str(event["error"])
                        break
                    message = event.get("message") or {}
                    thinking = message.get("thinking")
                    content = message.get("content") or ""
                    if thinking:
                        result.thinking += thinking
                        self.estimator.add()
                        if use_live:
                            refresh(0.10)
                        elif self.thinking_visible() and self.mode != "raw" \
                                and not self.use_pager:
                            if not printed_thinking_header:
                                self.console.print("[dim italic]thinking…[/dim italic]")
                                printed_thinking_header = True
                            self.console.print(f"[dim italic]{escape(thinking)}[/dim italic]", end="")
                    if content:
                        result.content += content
                        self.estimator.add()
                        if use_live:
                            heavy = len(result.content) > self.max_render_chars
                            refresh(1.0 if heavy else 0.10)
                        elif self.mode != "raw" and not self.use_pager:
                            self.console.print(escape(content), end="",
                                               soft_wrap=self.nowordwrap, markup=False,
                                               highlight=False)
                    if message.get("tool_calls"):
                        result.tool_calls.extend(message["tool_calls"])
                    if event.get("done"):
                        result.stats = stats_from_final(event)
                    handle_keys()
            except KeyboardInterrupt:
                result.cancelled = True
        finally:
            if live is not None:
                live.stop()
            if fd is not None and saved_term is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, saved_term)
                except (OSError, ValueError, termios.error):
                    pass
        if self.mode == "raw":
            sys.stdout.write(result.content)
            if result.content and not result.content.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
        elif use_live:
            if result.thinking and self.thinking_visible() \
                    and self.thinking_mode == "expanded":
                self.console.print(thinking_panel(result.thinking))
            self.render_final(result.content)
        elif self.use_pager:
            self._pager_render(result)
        else:
            if result.content and not result.cancelled:
                self.console.print()
        if result.cancelled and self.mode != "raw":
            self.console.print("[warn]cancelled[/warn]")
        if result.tool_calls and self.mode != "raw" and not self.use_pager:
            self._render_tool_calls(result.tool_calls)
        if (self.show_stats or self.verbose) and result.stats and self.mode != "raw" \
                and not self.use_pager:
            self.console.print(f"[dim]⏱  {format_stats(result.stats)}[/dim]")
        if result.error:
            self.console.print(f"[err]ollama error:[/err] {escape(result.error)}",
                               stderr=self.mode == "raw")
        return result

    def _pager_render(self, result: GenerationResult) -> None:
        with self.console.pager(styles=True):
            if self.thinking_visible() and result.thinking:
                self.console.print(thinking_panel(result.thinking))
            self.render_final(result.content)
            if result.tool_calls:
                self._render_tool_calls(result.tool_calls)
            if (self.show_stats or self.verbose) and result.stats:
                self.console.print(f"[dim]⏱  {format_stats(result.stats)}[/dim]")

    def _live_view(self, result: GenerationResult) -> Any:
        parts: list[Any] = []
        has_thinking = bool(result.thinking)
        preview = self._thinking_preview_lines()
        if result.content:
            if has_thinking and self.thinking_visible():
                if self.thinking_mode == "expanded":
                    parts.append(thinking_panel(result.thinking,
                                                "thinking · Ctrl+T to re-read",
                                                preview))
                else:
                    parts.append(Text(f"thinking · {human_chars(len(result.thinking))} · "
                                      "Ctrl+T to expand", style="dim italic"))
            parts.append(self._markdown(result.content))
        elif has_thinking and self.thinking_mode == "expanded":
            parts.append(thinking_panel(result.thinking, "thinking · Ctrl+T to re-read",
                                        preview))
            parts.append(Spinner("dots", text=Text(" thinking…", style="dim")))
        elif has_thinking and self.thinking_mode == "collapsed":
            parts.append(Spinner("dots", text=Text(
                f" thinking · {human_chars(len(result.thinking))} · Ctrl+T to expand",
                style="dim italic")))
        elif has_thinking:
            parts.append(Spinner("dots", text=Text(" thinking…", style="dim")))
        else:
            parts.append(Spinner("dots", text=Text(" waiting for model…", style="dim")))
        tps = self._tps_text()
        if tps is not None:
            parts.append(tps)
        group = Group(*parts)
        if self.live_render:
            return model_message_panel(group, self.model)
        return group

    def _render_tool_calls(self, calls: list[dict[str, Any]]) -> None:
        for call in calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = function.get("name", "tool")
            args = function.get("arguments", {})
            if not isinstance(args, str):
                args = json.dumps(args, indent=2)
            self.console.print(Panel(escape(args), title=f"tool · {name}",
                                     border_style="magenta", title_align="left"))


def render_transcript(session: "Session", console: Console, *, show_stats: bool = True,
                      pager: bool = False) -> None:
    """Render a saved conversation as it looked live (markdown for assistant)."""
    def body() -> None:
        console.print(Rule(f"[key]{escape(session.display_title())}[/key]"))
        meta = (f"[dim]id {session.id} · model {escape(session.model)} · "
                f"created {short_stamp(session.created_at)} · "
                f"updated {short_stamp(session.updated_at)} · "
                f"{len(session.messages)} messages[/dim]")
        console.print(meta)
        if session.system:
            console.print(Panel(escape(session.system), title="system",
                                border_style="dim", title_align="left"))
        for message in session.messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "user":
                for image in message.get("images") or []:
                    console.print(f"[dim]🖼  {escape(image.get('name') or image.get('path', '?'))}[/dim]")
                if content:
                    console.print(user_message_box(content, console))
            elif role == "assistant":
                if message.get("thinking"):
                    console.print(thinking_panel(message["thinking"]))
                if message.get("cancelled"):
                    console.print("[warn]cancelled partial response[/warn]")
                console.print(model_message_panel(
                    render_content(content, math=True, width=max(20, console.width - 4)),
                    session.model))
                for call in message.get("tool_calls") or []:
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    console.print(Panel(escape(str(function.get("arguments", ""))),
                                        title=f"tool · {function.get('name', 'tool')}",
                                        border_style="magenta", title_align="left"))
            else:
                console.print(Panel(escape(content), title=role, border_style="dim",
                                    title_align="left"))
        if show_stats and session.stats:
            stats = session.stats
            console.print(f"[dim]{stats.get('turns', 0)} turns · "
                          f"{stats.get('prompt_tokens', 0)} prompt tok · "
                          f"{stats.get('completion_tokens', 0)} completion tok · "
                          f"{stats.get('duration_s', 0):.0f}s total[/dim]")

    if pager and console.is_terminal:
        with console.pager(styles=True):
            body()
    else:
        body()
