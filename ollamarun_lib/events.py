"""Runtime events: the typed stream a turn emits for its front end.

This is the shared vocabulary between the agent runtime (`runtime.py`) and
whatever is consuming a turn: the CLI renderer, the Textual UI, or -- later --
the HTTP API. Phase 1 defines the types and a thin envelope; the API layer
adds turn ids, monotonic sequence numbers, replay and buffer semantics on top
of the same event types (see the API plan).

Event names are `namespace.verb` strings so a front end can switch on them
without guessing. `data` payloads are JSON-ready: no rich renderables, no
terminal formatting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TURN_STARTED = "turn.started"
MESSAGE_DELTA = "message.delta"
MESSAGE_COMPLETED = "message.completed"
TOOL_CALL = "tool.call"
TOOL_RESULT = "tool.result"
APPROVAL_REQUIRED = "approval.required"
APPROVAL_RESOLVED = "approval.resolved"
ARTIFACT_CREATED = "artifact.created"
TURN_WARNING = "turn.warning"
TURN_ERROR = "turn.error"
TURN_COMPLETED = "turn.completed"
TURN_CANCELLED = "turn.cancelled"

#: The states that end a turn; nothing is emitted after one of these.
TERMINAL_EVENTS = frozenset({TURN_COMPLETED, TURN_ERROR, TURN_CANCELLED})

#: The full Phase 1 vocabulary, for tests and documentation.
EVENT_TYPES = (
    TURN_STARTED, MESSAGE_DELTA, MESSAGE_COMPLETED, TOOL_CALL, TOOL_RESULT,
    APPROVAL_REQUIRED, APPROVAL_RESOLVED, ARTIFACT_CREATED, TURN_WARNING,
    TURN_ERROR, TURN_COMPLETED, TURN_CANCELLED,
)


@dataclass
class RuntimeEvent:
    """One semantic event of a turn, with a JSON-ready payload."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.data}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        keys = ", ".join(sorted(self.data))
        return f"RuntimeEvent({self.type!r}, {keys})"
