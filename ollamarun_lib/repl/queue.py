"""Messages and commands typed while a turn was running."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class QueuedItem:
    """One thing typed while a turn was running: prompt text or a `/command`."""

    kind: str  # "message" | "command"
    text: str


class MessageQueue:
    """What the user typed while a turn was running, in typing order.

    Messages are handed to the model at the next sub-task boundary -- the loop
    re-sends the conversation before every `/api/chat` round, so a message
    queued during a tool call is in context for the round that reacts to the
    tool result, and one queued while an answer streams is delivered in a
    follow-up round right after it. Commands only run once the turn is over
    (`/clear`, `/undo`, `/load` … would otherwise rewrite the conversation the
    running turn is built on). Delivery stops at the first command so the
    order the user typed things in is never rearranged.
    """

    def __init__(self) -> None:
        self.items: list[QueuedItem] = []

    def add_message(self, text: str) -> "QueuedItem":
        item = QueuedItem("message", text)
        self.items.append(item)
        return item

    def add_command(self, line: str) -> "QueuedItem":
        item = QueuedItem("command", line)
        self.items.append(item)
        return item

    def has_messages(self) -> bool:
        """True when the head of the queue is a message (so a boundary may take it)."""
        return bool(self.items) and self.items[0].kind == "message"

    def take_messages(self) -> list[str]:
        taken: list[str] = []
        while self.items and self.items[0].kind == "message":
            taken.append(self.items.pop(0).text)
        return taken

    def pop_next(self) -> Optional["QueuedItem"]:
        return self.items.pop(0) if self.items else None

    def remove(self, index: int) -> Optional["QueuedItem"]:
        """Drop item `index` (1-based, as `/queue rm N` counts them)."""
        if 1 <= index <= len(self.items):
            return self.items.pop(index - 1)
        return None

    def clear(self) -> int:
        count = len(self.items)
        self.items = []
        return count

    def describe(self) -> list[str]:
        return [f"{index}. {'you' if item.kind == 'message' else 'cmd'} ▸ {item.text}"
                for index, item in enumerate(self.items, 1)]

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)
