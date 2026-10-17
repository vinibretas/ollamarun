"""Runtime events: the typed stream a turn emits for its front end.

This is the shared vocabulary between the agent runtime (`runtime.py`) and
whatever is consuming a turn: the CLI renderer, the Textual UI, or the
HTTP API. This module defines the types and a thin envelope; the API layer
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
#: A turn-start context fit (`settings.num_ctx: "max-gpu"|"max-cpu"`): the
#: fitted num_ctx (or the failure) as the FitOutcome dict.
SESSION_FIT = "session.fit"
TURN_WARNING = "turn.warning"
TURN_ERROR = "turn.error"
TURN_COMPLETED = "turn.completed"
TURN_CANCELLED = "turn.cancelled"
#: Transport-level: the replay window predates the buffered history (or a
#: subscriber fell behind). Emitted only by the turn manager's replay path.
STREAM_GAP = "stream.gap"

#: The states that end a turn; nothing is emitted after one of these.
TERMINAL_EVENTS = frozenset({TURN_COMPLETED, TURN_ERROR, TURN_CANCELLED})

#: The runtime vocabulary plus the manager's `stream.gap`.
EVENT_TYPES = (
    TURN_STARTED, MESSAGE_DELTA, MESSAGE_COMPLETED, TOOL_CALL, TOOL_RESULT,
    APPROVAL_REQUIRED, APPROVAL_RESOLVED, ARTIFACT_CREATED, SESSION_FIT,
    TURN_WARNING, TURN_ERROR, TURN_COMPLETED, TURN_CANCELLED, STREAM_GAP,
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


@dataclass(frozen=True)
class TurnEvent:
    """One enveloped turn event: the runtime event plus its turn identity.

    `seq` is per turn, 1-based, assigned by the single writer (the turn's
    worker thread). `id` is `<turn_id>:<seq>` -- unique, ordered and
    dedupable, and exactly the `id:` the SSE layer sends -- so no random
    event ids are invented. `ts` is stamped by the turn manager (this
    module stays import-free, so it never calls into helpers).
    """

    turn_id: str
    session_id: str
    seq: int
    ts: str
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.turn_id}:{self.seq}"

    def to_dict(self) -> dict[str, Any]:
        """Deterministic key order: id, seq, type, turn_id, session_id, ts, data."""
        return {"id": self.id, "seq": self.seq, "type": self.type,
                "turn_id": self.turn_id, "session_id": self.session_id,
                "ts": self.ts, "data": self.data}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TurnEvent({self.id!r}, {self.type!r})"
