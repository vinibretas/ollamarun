"""UI seam: everything the REPL needs beyond plain text output.

`Ui` is the small interface `repl.py`/`interactive.py` talk to for interactive
prompts (confirmations, pickers, the tool-approval question, the `$EDITOR`
handoff) and for status notifications. `ConsoleUi` is the default, inline
implementation: the same `rich` prompts the REPL always used, plus no-op
notifications. The Textual front end (`tui.py`) implements the same interface
with modal dialogs and a live status bar, which is why nothing below the
`repl.py` layer needs to know which front end is running.

`StatusSnapshot`/`format_status()` are the status-bar model: the TUI renders
the formatted line, and the fields are plain data so they can be unit tested
without a terminal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from rich.markup import escape
from rich.prompt import Confirm, Prompt

from .constants import console
from .helpers import format_context, human_duration
from .tools import ToolRule, rule_label


@dataclass
class StatusSnapshot:
    """Everything the status bar shows, gathered from `Repl` on demand."""

    state: str = "ready"          # ready|generating|tool|approval|stopping|done|error
    detail: str = ""              # free text for the state (tool name, timing…)
    model: str = ""
    remote: Optional[str] = None
    thinking: str = "collapsed"
    tools: str = ""
    render: str = "markdown"
    context_used: Optional[int] = None
    context_total: Optional[int] = None
    queued: int = 0
    tps: Optional[float] = None
    tps_avg: Optional[float] = None
    elapsed: Optional[float] = None
    hint: str = ""

    def context_text(self) -> str:
        return format_context(self.context_used, self.context_total)


STATE_GLYPH = {
    "ready": "●",
    "running": "▸",
    "generating": "╎",
    "loading": "◌",
    "tool": "⚒",
    "approval": "?",
    "stopping": "×",
    "done": "✔",
    "error": "✖",
}
SPINNER_FRAMES = "⠋⠙⠹⠸⠴⠦⠧⠇"


def spinner_frame(state: str, tick: float) -> str:
    """The glyph for `state`: an animated frame while busy, a fixed icon when not."""
    if state in {"generating", "stopping", "loading", "running"}:
        return SPINNER_FRAMES[int(tick * 10) % len(SPINNER_FRAMES)]
    return STATE_GLYPH.get(state, "●")


def format_status(snapshot: StatusSnapshot, *, frame: str = "●", width: int = 80,
                  hint: str = "") -> str:
    """One line of rich markup for the status bar, fitted to `width`.

    Pieces carry a drop priority: when the terminal is too narrow, the mode
    summary goes first, then timings, then the hint -- the state, model and
    context usage always survive (truncated as a last resort).
    """
    state = snapshot.state
    busy = state in {"generating", "tool", "stopping", "loading", "running"}
    core = f"{frame} {state}"
    if snapshot.detail:
        core += f" [dim]{escape(snapshot.detail)}[/dim]"
    timing = []
    if busy and snapshot.elapsed:
        timing.append(human_duration(snapshot.elapsed))
    if snapshot.tps and state in {"generating", "done"}:
        speed = f"~{snapshot.tps:.1f} tok/s"
        if snapshot.tps_avg:
            speed += f" (avg {snapshot.tps_avg:.1f})"
        timing.append(speed)
    model = f"[key]{escape(snapshot.model or '?')}[/key]"
    if snapshot.remote:
        model += f" [dim]@{escape(snapshot.remote)}[/dim]"
    pieces = [  # (text, drop priority, side) -- lower priority drops first
        (core, 0, "left"),
        (model, 0, "left"),
        (snapshot.context_text(), 0, "right"),
        (hint or snapshot.hint, 0, "right"),
        (f"⏳ {snapshot.queued} queued" if snapshot.queued else "", 1, "right"),
        (f"render:{snapshot.render}", 2, "left"),
        (" ".join(timing), 3, "left"),
        (f"tools:{snapshot.tools}" if snapshot.tools else "", 4, "left"),
        (f"think:{snapshot.thinking}", 5, "left"),
    ]
    for limit in (5, 4, 3, 2, 1, 0):
        left = [text for text, priority, side in pieces
                if side == "left" and priority <= limit and text]
        right = [text for text, priority, side in pieces
                 if side == "right" and priority <= limit and text]
        left_text = " [dim]│[/dim] ".join(left)
        right_text = " [dim]│[/dim] ".join(escape(text) for text in right)
        room = width - len(strip_markup(left_text)) - len(strip_markup(right_text))
        if room >= 1:
            return left_text + " " * room + right_text
    # Even the essentials are wider than the terminal: keep the left side
    # (state + model) and the most actionable right pieces, then clip.
    essential_left = [text for text, priority, side in pieces
                      if side == "left" and priority == 0 and text]
    essential_right = list(reversed([text for text, priority, side in pieces
                                     if side == "right" and priority == 0 and text]))
    tail = " ".join(escape(text) for text in essential_right)
    return clip_text(" ".join([strip_markup(" [dim]│[/dim] ".join(essential_left)),
                               strip_markup(tail)]).strip(), width)


def clip_text(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def strip_markup(markup: str) -> str:
    """Length of `markup` as the terminal will show it (tags stripped)."""
    import re

    return re.sub(r"\[/?[a-zA-Z#][^\]]*\]", "", markup)


def approval_entries(suggestions: list[ToolRule]
                     ) -> list[tuple[str, Optional[ToolRule], str]]:
    """The numbered choices offered for one tool call.

    Index 0 runs it once, 1 remembers exactly this call, one entry per
    suggested generalized rule, then auto-allow-everything and deny. Shared by
    the console prompt and the TUI modal so both offer the same list.
    """
    entries: list[tuple[str, Optional[ToolRule], str]] = [
        ("run", None, "run once"),
        ("remember", None, "always allow exactly this call"),
    ]
    for suggestion in suggestions:
        entries.append(("rule", suggestion,
                        f"always allow {rule_label(suggestion)} (this session)"))
    entries.append(("auto", None, "auto-allow every later call  [CAREFUL]"))
    entries.append(("deny", None, "deny"))
    return entries


def approval_letters(count: int) -> dict[int, str]:
    """`1/y`, `2/a`, `A`, `n`: the letter that picks an entry, by index."""
    return {0: "y", 1: "a", count - 2: "A", count - 1: "n"}


def approval_lines(entries: list[tuple[str, Optional[ToolRule], str]]
                   ) -> list[tuple[str, str]]:
    """(tag, label) per entry, e.g. ("1/y", "run once"), for either front end."""
    letters = approval_letters(len(entries))
    return [(f"{index + 1}/{letters[index]}" if index in letters else str(index + 1),
             label)
            for index, (_action, _rule, label) in enumerate(entries)]


def _prompt_approval(suggestions: list[ToolRule]
                     ) -> tuple[str, Optional[ToolRule]]:
    """Numbered, typed approval prompt for one tool call.

    Prints every choice on its own line (`1/y` run once, `2/a` always allow
    the exact call, one line per suggested generalized rule, `A` auto-allow
    every later call, `n` deny) and reads a single typed number or letter.
    Returns (action, rule) with action "run" | "remember" | "rule" | "auto"
    | "deny"; "rule" carries the ToolRule to store.
    """
    entries = approval_entries(suggestions)
    lines = approval_lines(entries)
    for tag, label in lines:
        console.print(f"  [key]{tag:<4}[/key] {escape(label)}")
    letters = {tag.split("/")[1]: index
               for index, (tag, _label) in enumerate(lines) if "/" in tag}
    choices = [tag.split("/")[0] for tag, _label in lines] + list(letters)
    answer = Prompt.ask("[dim]choose[/dim]", choices=choices, default="1",
                        show_choices=False).strip()
    if answer in letters:
        index = letters[answer]
    else:
        index = int(answer) - 1
    action, rule, _label = entries[index]
    return action, rule


class Ui:
    """Interactive hooks the REPL calls; the console implementation is the default."""

    def notify(self, event: str, **data: Any) -> None:
        """Status/state change for the front end (`turn_start`, `done`, …)."""

    def ask(self, question: str, *, choices: Optional[Sequence[str]] = None,
            default: Optional[str] = None) -> Optional[str]:
        return Prompt.ask(question, choices=list(choices) if choices else None,
                          default=default)

    def confirm(self, question: str, *, default: bool = False) -> bool:
        return Confirm.ask(question, default=default)

    def editor(self, initial: str = "") -> Optional[str]:
        from .interactive import open_in_editor

        return open_in_editor(initial)

    def approval(self, suggestions: list[ToolRule], *,
                 name: str = "", detail: str = "", reason: str = ""
                 ) -> tuple[str, Optional[ToolRule]]:
        """Ask how to handle a tool call.

        `suggestions` are the generalized rules the tool call could be covered
        by (see `approval_entries`); `name`/`detail`/`reason` describe the call
        for front ends that ask in a modal (the console prompt already printed
        them). Returns (action, rule), with "rule" carrying the rule to store.
        """
        return _prompt_approval(suggestions)


class ConsoleUi(Ui):
    """Inline front end: rich prompts, nothing to notify."""
