"""Streaming renderer: drives a `rich.Live` view while chat events stream in.

`Renderer` is the piece that talks to the terminal frame-by-frame -- markdown
re-render while tokens stream, the collapsible thinking panel (Ctrl+T expand/
collapse) and `TpsEstimator` for the live `~tok/s` footer. `user_message_box`/
`model_message_rule` are the chat-bubble helpers shared with transcript
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
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.text import Text

from .constants import CTRL_T, THINKING_MODES, err_console, tool_color
from .events import MESSAGE_DELTA, RuntimeEvent
from .helpers import (CancelToken, diff_preview, format_context, human_chars,
                      human_duration, short_stamp)
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

    Uses a sliding window so the rate reacts to speed changes; `average()`
    keeps the cumulative figure the footer shows next to the windowed one,
    and `rate()` stays the display fallback (the window, else the average).
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
        self._prune(now)

    def _prune(self, now: float) -> None:
        horizon = now - self.window
        while len(self.samples) > 2 and self.samples[0][0] < horizon:
            self.samples.popleft()

    def windowed(self, now: Optional[float] = None) -> Optional[float]:
        """The sliding-window rate, or None while the window is too fresh."""
        if self.started_at is None or self.tokens == 0:
            return None
        now = time.monotonic() if now is None else now
        self._prune(now)
        base_time, base_tokens = self.samples[0]
        if len(self.samples) > 2 and now - base_time >= 1.0 and self.tokens > base_tokens:
            return (self.tokens - base_tokens) / (now - base_time)
        return None

    def average(self, now: Optional[float] = None) -> Optional[float]:
        """The cumulative average since the first streamed chunk."""
        if self.started_at is None or self.tokens == 0:
            return None
        now = time.monotonic() if now is None else now
        elapsed = now - self.started_at
        return self.tokens / elapsed if elapsed > 0 else None

    def rate(self, now: Optional[float] = None) -> Optional[float]:
        return self.windowed(now) or self.average(now)


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


def consume_chunk(result: "GenerationResult", event: dict[str, Any],
                  estimator: Optional["TpsEstimator"] = None) -> tuple[str, str]:
    """Apply one streamed chat chunk to `result`; returns (thinking, content).

    Shared by the terminal `Renderer.stream()` and the headless renderer, so
    both accumulate exactly the same `GenerationResult` from the same NDJSON.
    """
    message = event.get("message") or {}
    thinking = message.get("thinking") or ""
    content = message.get("content") or ""
    if thinking:
        result.thinking += thinking
        if estimator is not None:
            estimator.add()
    if content:
        result.content += content
        if estimator is not None:
            estimator.add()
    if message.get("tool_calls"):
        result.tool_calls.extend(message["tool_calls"])
    if event.get("done"):
        result.stats = stats_from_final(event)
    return thinking, content


def thinking_panel(text: str, title: str = "thinking",
                   preview_lines: Optional[int] = None) -> Panel:
    """Dim thinking block rendered like the answer; preview_lines keeps the tail."""
    return Panel(render_thinking(text, preview_lines=preview_lines), title=title,
                 border_style="dim", title_align="left", padding=(0, 1))


def tool_call_panel(name: str, args: Any) -> Panel:
    """The arguments panel for one tool call, bordered in the tool's colour.

    `edit_file` calls render an old→new diff instead of raw JSON so the
    change is readable before it is approved; anything that does not parse
    into the expected shape falls back to the JSON dump.
    """
    if name == "edit_file":
        preview = _edit_diff_body(args)
        if preview is not None:
            color = tool_color(name)
            return Panel(preview, border_style=color, title_align="left",
                         title=Text(f"tool · {name}", style=color))
    if not isinstance(args, str):
        args = json.dumps(args, indent=2)
    color = tool_color(name)
    return Panel(escape(args), border_style=color, title_align="left",
                 title=Text(f"tool · {name}", style=color))


def _edit_diff_body(args: Any) -> Optional[Text]:
    """Colored diff body for an edit_file call; None = use the JSON dump."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    old, new = args.get("old_string"), args.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        return None
    body = Text()
    path = args.get("path")
    if path:
        body.append(f"{path}\n", style="bold")
    diff = diff_preview(old, new, limit=20)
    for line in (diff.splitlines() or ["(no change)"]):
        if line.startswith("+"):
            style = "green"
        elif line.startswith("-"):
            style = "red"
        else:
            style = "dim"
        body.append(line + "\n", style=style)
    return body


def tool_result_panel(name: str, detail: str = "", text: str = "", *,
                      ok: bool = True, duration: Optional[float] = None,
                      note: str = "", head: int = 12, max_chars: int = 1400) -> Panel:
    """Compact view of what a tool returned; the model gets the full text.

    The whole box wears the tool's colour, exactly like its call panel; the
    ok/failed state lives in the title instead of the border (green ok, red
    failed), so the tool identity stays unbroken across a run. `note` is an
    optional extra title bit (the speak tool's `optiplex → g3`), also drawn
    in the tool's colour.
    """
    lines = (text or "").splitlines()
    shown = lines[:head]
    if len(lines) > head:
        shown.append(f"… {len(lines) - head} more lines (full output sent to the model)")
    body = "\n".join(shown) or "(no output)"
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "…"
    rendered = Text()
    if detail:
        rendered.append("$ " + detail + "\n", style="dim")
    rendered.append(body)
    color = tool_color(name)
    title = Text.assemble("tool · ", (str(name), color), " · ",
                          ("ok" if ok else "failed", "green" if ok else "red"))
    if note:
        title.append(f" · {note}", style=color)
    if duration is not None:
        title.append(f" · {duration:.2f}s")
    return Panel(rendered, title=title, border_style=color, title_align="left")


def user_message_box(text: str, console: Console) -> Align:
    """Small right-aligned box for a user turn."""
    lines = text.splitlines() or [""]
    content_width = max(len(line) for line in lines)
    width = max(12, min(content_width + 4, (console.width or 80) - 4, 72))
    panel = Panel(Text(text), border_style="you", padding=(0, 1), width=width,
                 title="you", title_align="right")
    return Align.right(panel)


def model_message_rule(model: str) -> Rule:
    """Divider (no box) marking the start of a model turn."""
    title = f"[ai]{escape(model)} ›[/ai]" if model else "[ai]assistant[/ai]"
    return Rule(title, style="ai", align="left")


def final_response_rule() -> Rule:
    """Bright centered divider between a tool loop and the committed answer.

    A round is only known to be the final answer once it has finished without
    requesting more tools, so this rule is not part of the streaming view:
    it appears in the frame frozen at that moment (and in the pager, which
    renders a round after it completed), separating the processing above
    from the answer below.
    """
    return Rule("[answer]response[/answer]", style="answer", align="center")


class StreamSink:
    """Where a streaming response is displayed.

    The default is `rich.Live` (the inline REPL); the Textual front end passes
    its own sink and renders the same frames into a widget instead. `update`
    is called at most ~10 times a second (once a second for very long
    responses) with exactly the renderable the live view would have shown.
    """

    def start(self, renderer: "Renderer") -> None:
        """A new streamed response starts."""

    def update(self, renderable: Any) -> None:
        """A fresh frame (may be skipped by the renderer when throttled)."""

    def finish(self, result: "GenerationResult") -> None:
        """The response ended; the last frame is the final one."""


class Renderer:
    """Streams NDJSON chat events to the terminal and returns the transcript."""

    def __init__(self, console: Console, *, mode: str = "markdown", math: bool = True,
                 nowordwrap: bool = False, thinking_mode: str = "collapsed",
                 tps_footer: bool = True, context_used: Optional[int] = None,
                 context_total: Optional[int] = None,
                 show_stats: bool = False,
                 max_render_chars: int = 12000, pager: bool = False, model: str = "",
                 after_tools: bool = False):
        self.console = console
        self.mode = mode
        self.model = model
        self.after_tools = after_tools
        self._committed_final = False
        self.math = math
        self.nowordwrap = nowordwrap
        self.thinking_mode = thinking_mode if thinking_mode in THINKING_MODES else "collapsed"
        self.tps_footer = tps_footer
        # Tokens the conversation already carried when this round started
        # (prompt + completion of the last turn) and the model's window; the
        # footer adds the round's streamed chunks on top for a live ctx count.
        self.context_used = context_used
        self.context_total = context_total
        self.estimator = TpsEstimator()
        self.show_stats = show_stats
        self.max_render_chars = max_render_chars
        self.pager = pager
        self.use_pager = bool(pager) and console.is_terminal and mode != "raw"
        self.live_render = mode == "markdown" and console.is_terminal and not self.use_pager
        self.sink_render = False
        self.interactive = False
        try:
            self.interactive = (self.live_render and sys.stdin.isatty())
        except (ValueError, OSError):
            self.interactive = False

    def with_sink(self, sink: Optional[StreamSink]) -> "Renderer":
        """Render through `sink` (a widget) instead of a `rich.Live` view."""
        if sink is not None:
            self.live_render = False
            self.sink_render = True
            self.interactive = False
        return self

    @property
    def render_frames(self) -> bool:
        """True when something other than the console shows the live frames
        (a `rich.Live` region or a front-end widget): they get the model rule
        and the response divider inside the frame instead of a printed line."""
        return self.live_render or self.sink_render

    def thinking_visible(self) -> bool:
        return self.thinking_mode != "hidden"

    def _toggle_thinking(self) -> None:
        self.thinking_mode = "collapsed" if self.thinking_mode == "expanded" else "expanded"

    def _poll_keys(self, fd: int) -> bool:
        """Reads pending stdin keys; True when Ctrl+T changed the view.

        Ctrl+T expands the thinking panel inline (the boxed view the pager
        used to show) or collapses it again -- no external pager.
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
                    changed = True
        except (OSError, ValueError):
            pass
        return changed

    def _tps_text(self, now: Optional[float] = None) -> Optional[Text]:
        """The speed footer: windowed rate, average, tokens and live ctx.

        `ctx` counts what the next request will carry (the last round's
        prompt + completion) plus this round's streamed chunks, so the
        percentage moves with every token; it appears from the second
        round on, when there is a base to count from.
        """
        if not self.tps_footer:
            return None
        estimator = self.estimator
        rate = estimator.rate(now)
        if not rate or not estimator.tokens:
            return None
        bits = [f"~{rate:.1f} tok/s"]
        average = estimator.average(now)
        if average:
            bits.append(f"(avg {average:.1f})")
        bits.append(f"{estimator.tokens} tok")
        if self.context_used is not None:
            bits.append(format_context(self.context_used + estimator.tokens,
                                       self.context_total))
        return Text(" · ".join(bits), style="dim")

    def _thinking_preview_lines(self) -> int:
        height = getattr(self.console, "height", None) or 24
        return max(5, height - 10)

    def final_rule_wanted(self, result: GenerationResult) -> bool:
        """True when this round is the committed answer of a tool loop.

        Detection happens the only way it can: a round that finishes with
        tool calls is interim (the model is still working), so a round that
        runs after tools and finishes without them is the answer the model
        committed to. Cancelled or errored rounds never get the divider.
        """
        return bool(self.after_tools and not result.tool_calls
                    and not result.cancelled and not result.error)

    def render_final(self, content: str) -> None:
        if self.mode == "raw" or not content:
            return
        if self.mode == "markdown":
            body = self._markdown(content)
            if self.render_frames:
                self.console.print(model_message_rule(self.model))
            self.console.print(body, soft_wrap=self.nowordwrap)
        else:
            self.console.print(escape(content), soft_wrap=self.nowordwrap)

    def _markdown(self, content: str) -> Group:
        width = max(20, (self.console.width or 80) - 4)
        return render_content(content, math=self.math, width=width)

    def live_frame(self, result: GenerationResult) -> Any:
        """The renderable the live view shows for `result`: public alias used
        by the TUI, which updates a widget instead of a `rich.Live`."""
        return self._live_view(result)

    def stream(self, events: Iterator[dict[str, Any]],
               cancel: Optional[CancelToken] = None,
               sink: Optional[StreamSink] = None) -> GenerationResult:
        result = GenerationResult()
        use_live = self.live_render
        if sink is not None and self.mode != "raw":
            sink.start(self)
        else:
            sink = None
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
            if live is None and sink is None:
                return
            now = time.monotonic()
            if force or now - last_refresh >= interval:
                frame = self._live_view(result)
                if sink is not None:
                    sink.update(frame)
                else:
                    live.update(frame, refresh=True)
                last_refresh = now

        def handle_keys() -> None:
            if live is None or fd is None:
                return
            if self._poll_keys(fd):
                refresh(0.0, force=True)

        try:
            if use_live:
                live = Live(self._live_view(result), console=self.console,
                            refresh_per_second=10, vertical_overflow="visible")
                live.start()
            try:
                for event in events:
                    if cancel is not None and cancel.cancelled:
                        result.cancelled = True
                        break
                    if event.get("error"):
                        result.error = str(event["error"])
                        break
                    thinking, content = consume_chunk(result, event, self.estimator)
                    if thinking:
                        if use_live or sink is not None:
                            refresh(0.10)
                        elif self.thinking_visible() and self.mode != "raw" \
                                and not self.use_pager:
                            if not printed_thinking_header:
                                self.console.print("[dim italic]thinking…[/dim italic]")
                                printed_thinking_header = True
                            self.console.print(f"[dim italic]{escape(thinking)}[/dim italic]", end="")
                    if content:
                        if use_live or sink is not None:
                            heavy = len(result.content) > self.max_render_chars
                            refresh(1.0 if heavy else 0.10)
                        elif self.mode != "raw" and not self.use_pager:
                            self.console.print(escape(content), end="",
                                               soft_wrap=self.nowordwrap, markup=False,
                                               highlight=False)
                    handle_keys()
            except KeyboardInterrupt:
                result.cancelled = True
        finally:
            if live is not None:
                # Force a final refresh so the frame frozen when the live
                # view stops reflects the complete content -- throttled
                # refreshes during streaming can leave the last live frame
                # stale otherwise, which used to be papered over by printing
                # the whole response a second time below. This is also the
                # moment a round's fate is known: `_committed_final` lets the
                # frozen frame (and only it) carry the response divider when
                # this was the answer that ended a tool loop.
                self._committed_final = self.final_rule_wanted(result)
                refresh(0.0, force=True)
                live.stop()
            if fd is not None and saved_term is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, saved_term)
                except (OSError, ValueError, termios.error):
                    pass
        if cancel is not None and cancel.cancelled:
            # The reader may have ended because the response was closed under
            # it (see OllamaClient.abort_stream) rather than by a chunk.
            result.cancelled = True
        if sink is not None:
            # Freeze the final frame (as the live view does) and hand the
            # response back to the front end -- nothing is re-printed. The
            # committed-final divider is decided now, so the frozen frame
            # carries it exactly like the live one.
            self._committed_final = self.final_rule_wanted(result)
            refresh(0.0, force=True)
            sink.finish(result)
        if self.mode == "raw":
            sys.stdout.write(result.content)
            if result.content and not result.content.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
        elif use_live:
            # The frozen live frame (see the forced refresh above) already
            # shows the complete rule + rendered content -- printing it
            # again here would duplicate the whole response.
            if result.thinking and self.thinking_visible() \
                    and self.thinking_mode == "expanded":
                self.console.print(thinking_panel(result.thinking))
        elif self.use_pager:
            self._pager_render(result)
        elif sink is None:
            if result.content and not result.cancelled:
                self.console.print()
        if result.cancelled and self.mode != "raw":
            self.console.print("[warn]cancelled[/warn]")
        if result.tool_calls and self.mode != "raw" and not self.use_pager:
            self._render_tool_calls(result.tool_calls)
        if self.show_stats and result.stats and self.mode != "raw" \
                and not self.use_pager:
            self.console.print(f"[dim]⏱  {format_stats(result.stats)}[/dim]")
        if result.error and self.mode == "raw" and sink is None:
            # Raw mode suppresses the runtime's `turn.error` console line
            # (`quiet`) and writes the payload to stdout, so the error is
            # reported here, on stderr. Every other mode is reported exactly
            # once by that event line -- the way an immediate failure is.
            err_console.print(f"[err]ollama error:[/err] {escape(result.error)}")
        return result

    def _pager_render(self, result: GenerationResult) -> None:
        with self.console.pager(styles=True):
            if self.final_rule_wanted(result):
                self.console.print(final_response_rule())
            if self.thinking_visible() and result.thinking:
                self.console.print(thinking_panel(result.thinking))
            self.render_final(result.content)
            if result.tool_calls:
                self._render_tool_calls(result.tool_calls)
            if self.show_stats and result.stats:
                self.console.print(f"[dim]⏱  {format_stats(result.stats)}[/dim]")

    def _live_view(self, result: GenerationResult) -> Any:
        parts: list[Any] = []
        has_thinking = bool(result.thinking)
        preview = self._thinking_preview_lines()
        if result.content:
            if has_thinking and self.thinking_visible():
                if self.thinking_mode == "expanded":
                    parts.append(thinking_panel(result.thinking,
                                                "thinking · Ctrl+T to collapse",
                                                preview))
                else:
                    parts.append(Text(f"thinking · {human_chars(len(result.thinking))} · "
                                      "Ctrl+T to expand", style="dim italic"))
            parts.append(self._markdown(result.content))
        elif has_thinking and self.thinking_mode == "expanded":
            parts.append(thinking_panel(result.thinking, "thinking · Ctrl+T to collapse",
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
        if self.render_frames:
            head: list[Any] = []
            if self._committed_final:
                head.append(final_response_rule())
            head.append(model_message_rule(self.model))
            return Group(*head, group)
        return group

    def _render_tool_calls(self, calls: list[dict[str, Any]]) -> None:
        for call in calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            self.console.print(tool_call_panel(function.get("name", "tool"),
                                               function.get("arguments", {})))


class HeadlessRenderer:
    """A terminal-free `Renderer` for hosts driven by events (the API server).

    Consumes the same chat chunks and returns the same `GenerationResult`, but
    prints nothing, starts no `rich.Live` and polls no keyboard. Every delta is
    handed to `on_event` as a `message.delta` runtime event, so a front end can
    stream content/thinking without owning the chunk parsing (shared with
    `Renderer.stream` through `consume_chunk`).
    """

    def __init__(self, *, on_event: Optional[Callable[[RuntimeEvent], None]] = None,
                 thinking_mode: str = "hidden", after_tools: bool = False):
        self.estimator = TpsEstimator()
        self.on_event = on_event
        self.after_tools = after_tools
        self.mode = "plain"
        self.render_frames = False
        self.thinking_mode = thinking_mode if thinking_mode in THINKING_MODES else "hidden"
        self.tps_footer = False
        self.context_used: Optional[int] = None
        self.context_total: Optional[int] = None
        self._committed_final = False

    def _emit(self, event_type: str, **data: Any) -> None:
        if self.on_event is not None:
            self.on_event(RuntimeEvent(event_type, data))

    def stream(self, events: Iterator[dict[str, Any]],
               cancel: Optional[CancelToken] = None,
               sink: Optional[StreamSink] = None) -> GenerationResult:
        result = GenerationResult()
        for event in events:
            if cancel is not None and cancel.cancelled:
                result.cancelled = True
                break
            if event.get("error"):
                result.error = str(event["error"])
                break
            thinking, content = consume_chunk(result, event, self.estimator)
            if thinking:
                self._emit(MESSAGE_DELTA, channel="thinking", text=thinking)
            if content:
                self._emit(MESSAGE_DELTA, channel="content", text=content)
        if cancel is not None and cancel.cancelled:
            result.cancelled = True
        return result


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
                console.print(model_message_rule(session.model))
                console.print(render_content(content, math=True, width=max(20, console.width - 4)))
                for call in message.get("tool_calls") or []:
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    console.print(tool_call_panel(function.get("name", "tool"),
                                                  function.get("arguments", "")))
            elif role == "tool":
                content = message.get("content") or ""
                if content:
                    console.print(tool_result_panel(message.get("tool_name") or "tool",
                                                    text=content,
                                                    note=message.get("tool_note") or ""))
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
