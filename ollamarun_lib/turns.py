"""Turn manager: turn records, workers, envelopes, replay and retention.

`TurnManager` is the one place that knows what a *turn* is at the API
level. `runtime.AgentRuntime` emits `RuntimeEvent`s and knows nothing
about transport; the manager wraps a host-supplied work closure, stamps
every event with a `TurnEvent` envelope (turn id, 1-based seq,
timestamp), buffers it for replay, fans it out to subscribers, and owns
the turn's lifecycle, cancellation and retention.

The work closure signature is `work(sink, cancel) -> TurnResult`:

    def work(sink, cancel):
        hooks = repl._turn_hooks()
        hooks.on_event = sink
        hooks.cancel = cancel
        ...
        result = AgentRuntime(hooks).run()
        return runtime.to_turn_result(result, session_id=repl.session.id)

so the manager never imports a front end. Events may only be emitted
from the worker thread; that single writer is what makes `seq` trivial.
A closure that raises, or that returns without a terminal event, still
gets one: the manager synthesizes the terminal matching the result's
status and flags it (`data["synthetic"]`).

Replay is what a reconnecting client needs: subscribe with
`after=<last seq it saw>` and the manager replays the buffered events
that follow, or -- when the requested window predates the buffer --
first emits a `stream.gap` so the client knows events are missing and
re-reads the turn/session state instead of assuming continuity. A gap
event carries `seq` = the resume point (the last unknown seq), so the
client's `after` stays meaningful; gap events are per-subscriber and are
never buffered. Subscribers have bounded queues: one that cannot keep up
is sent a `slow_subscriber` gap and closed -- it can never stall the
turn.

Typed errors map onto HTTP in Phase 4: `SessionBusy` -> 409,
`TooManyTurns` -> 429, `UnknownTurn` -> 404, `TurnExpired` -> 410.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional

from .events import (STREAM_GAP, TERMINAL_EVENTS, TURN_CANCELLED, TURN_COMPLETED,
                     TURN_ERROR, TURN_STARTED, RuntimeEvent, TurnEvent)
from .helpers import CancelToken, now_iso
from .runtime import TurnResult

EventSink = Callable[[RuntimeEvent], None]
Work = Callable[[EventSink, CancelToken], TurnResult]

#: Turn lifecycle states. `waiting_approval` is reserved for Phase 4.
QUEUED = "queued"
RUNNING = "running"
WAITING_APPROVAL = "waiting_approval"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATUSES = frozenset({COMPLETED, FAILED, CANCELLED})

_TERMINAL_OF_TYPE = {TURN_COMPLETED: COMPLETED,
                     TURN_ERROR: FAILED,
                     TURN_CANCELLED: CANCELLED}


class TurnManagerError(RuntimeError):
    """Base for the manager's typed errors (mapped to HTTP in Phase 4)."""


class SessionBusy(TurnManagerError):
    """The session already has an active turn; no concurrent mutation."""

    def __init__(self, session_id: str, active_turn_id: str):
        super().__init__(f"session {session_id!r} already has the active turn "
                         f"{active_turn_id!r}")
        self.session_id = session_id
        self.active_turn_id = active_turn_id


class TooManyTurns(TurnManagerError):
    """The global active-turn cap is full."""

    def __init__(self, limit: int):
        super().__init__(f"too many active turns (limit {limit})")
        self.limit = limit


class UnknownTurn(TurnManagerError):
    def __init__(self, turn_id: str):
        super().__init__(f"unknown turn {turn_id!r}")
        self.turn_id = turn_id


class TurnExpired(TurnManagerError):
    """The turn existed but its buffered events were swept."""

    def __init__(self, turn_id: str, status: str):
        super().__init__(f"turn {turn_id!r} expired (status {status})")
        self.turn_id = turn_id
        self.status = status


class ManagerClosed(TurnManagerError):
    """`shutdown()` was called; no new turns are accepted."""


@dataclass(frozen=True)
class TurnSnapshot:
    """A point-in-time view of one turn (the `GET /turns/{id}` shape)."""

    id: str
    session_id: str
    status: str
    created_at: str
    updated_at: str
    seq: int
    oldest_seq: int
    latest_seq: int
    result: Optional[TurnResult] = None
    error: Optional[str] = None
    context: Optional[dict[str, Any]] = None
    truncated: bool = False
    synthetic_terminal: bool = False


@dataclass
class _TurnRecord:
    id: str
    session_id: str
    created_at: str
    updated_at: str
    cancel: Optional[CancelToken]
    context: Optional[dict[str, Any]]
    work: Optional[Work] = None
    sink: Optional[EventSink] = None
    status: str = QUEUED
    events: Deque[TurnEvent] = field(default_factory=deque)
    oldest_seq: int = 1
    latest_seq: int = 0
    terminal_seq: Optional[int] = None
    terminal_seen: bool = False
    synthetic_terminal: bool = False
    truncated: bool = False
    result: Optional[TurnResult] = None
    error: Optional[str] = None
    subscribers: list["Subscription"] = field(default_factory=list)
    thread: Optional[threading.Thread] = None
    worker_done: bool = False
    finished_at: Optional[float] = None


@dataclass
class _Expired:
    status: str
    at: float


class Subscription:
    """One consumer's view of a turn (Phase 3 turns this into SSE).

    `get(timeout)` returns the next event, or None on timeout or after
    `close()`; consumers should loop while `not closed` with a finite
    timeout, then check `at_terminal` to decide whether the stream ended
    normally. For each gap (`stream.gap`) the client must treat every
    event up to `seq` as unknown and re-read the turn/session state.
    """

    def __init__(self, manager: "TurnManager", record: _TurnRecord, *,
                 after: Optional[int], queue_size: int):
        self.manager = manager
        self.turn_id = record.id
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._closed = False
        self.at_terminal = False
        self.gap_pending = False
        self._replay(record, after)

    # ── consumer side ───────────────────────────────────────────────────────

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def get(self, timeout: Optional[float] = None) -> Optional[TurnEvent]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._force(None)

    # ── manager side ────────────────────────────────────────────────────────

    def _replay(self, record: _TurnRecord, after: Optional[int]) -> None:
        latest = record.latest_seq
        if after is not None:
            wanted = max(1, int(after) + 1)
            if wanted < record.oldest_seq:
                self.gap_pending = True
                self._offer(self.manager._gap_event(
                    record, reason="truncated_buffer",
                    resume_seq=record.oldest_seq - 1,
                    requested_seq=int(after)))
            for event in list(record.events):
                if event.seq >= wanted:
                    self._offer(event)
        self.at_terminal = record.terminal_seen and latest == record.latest_seq

    def _offer(self, event: Optional[TurnEvent]) -> None:
        with self._lock:
            if self._closed:
                return
        if event is not None and event.type in TERMINAL_EVENTS:
            self.at_terminal = True
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._overflow()

    def _force(self, event: Optional[TurnEvent]) -> None:
        """Enqueue a notice even when the queue is full (drops the oldest)."""
        try:
            self._queue.put_nowait(event)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(event)
        except queue.Full:  # pragma: no cover - queue_size >= 1
            pass

    def _overflow(self) -> None:
        """A subscriber that cannot keep up gets one gap, then is closed."""
        with self.manager._lock:
            record = self.manager._records.get(self.turn_id)
            gap = None
            if record is not None:
                gap = self.manager._gap_event(
                    record, reason="slow_subscriber",
                    resume_seq=record.latest_seq,
                    requested_seq=max(0, record.latest_seq - 1))
        if gap is not None:
            self.gap_pending = True
            self._force(gap)
        with self._lock:
            self._closed = True
        self._force(None)  # unblock a waiting consumer


class TurnHandle:
    """The caller's handle on one submitted turn."""

    def __init__(self, manager: "TurnManager", record: _TurnRecord):
        self._manager = manager
        self._record = record
        self.id = record.id
        self.session_id = record.session_id

    def snapshot(self) -> Optional[TurnSnapshot]:
        return self._manager.get(self.id)

    def cancel(self) -> bool:
        return self._manager._request_cancel(self.id)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TurnHandle({self.id!r}, {self.session_id!r})"


class TurnManager:
    """Owns turns: registry, workers, buffers, subscriptions, retention."""

    def __init__(self, *, max_active: int = 4, buffer_events: int = 10_000,
                 completed_ttl: float = 86_400.0, expired_ttl: float = 3_600.0,
                 subscriber_queue: int = 1_000,
                 clock: Callable[[], float] = time.monotonic):
        if max_active < 1:
            raise ValueError("max_active must be >= 1")
        if buffer_events < 1:
            raise ValueError("buffer_events must be >= 1")
        self.max_active = int(max_active)
        self.buffer_events = int(buffer_events)
        self.completed_ttl = float(completed_ttl)
        # How long a swept turn still answers "expired" (410) instead of
        # "unknown" (404), so a reconnecting client gets a useful answer.
        self.expired_ttl = max(float(expired_ttl), float(completed_ttl))
        self.subscriber_queue = int(subscriber_queue)
        self._clock = clock
        self._lock = threading.RLock()
        self._records: dict[str, _TurnRecord] = {}
        self._active_by_session: dict[str, str] = {}
        self._active_count = 0
        self._expired: dict[str, _Expired] = {}
        self._closed = False

    # ── submission ──────────────────────────────────────────────────────────

    def submit(self, *, session_id: str, work: Work,
               turn_id: Optional[str] = None,
               cancel: Optional[CancelToken] = None,
               context: Optional[Callable[[], dict[str, Any]]] = None
               ) -> TurnHandle:
        """Start a turn on its own worker thread; returns its handle.

        `context` (optional) is evaluated once, before the turn starts, and
        attached to `turn.started`/`turn.completed` -- the manager performs
        no I/O of its own.
        """
        if not session_id:
            raise ValueError("session_id is required")
        context_data: Optional[dict[str, Any]] = None
        if context is not None:
            try:
                context_data = dict(context())
            except Exception:  # a broken provider must not block the turn
                context_data = None
        with self._lock:
            if self._closed:
                raise ManagerClosed("the turn manager is shut down")
            self._sweep_locked(self._clock())
            active = self._active_by_session.get(session_id)
            if active is not None:
                raise SessionBusy(session_id, active)
            if self._active_count >= self.max_active:
                raise TooManyTurns(self.max_active)
            turn_id = turn_id or f"turn_{uuid.uuid4().hex[:12]}"
            stamp = now_iso()
            record = _TurnRecord(id=turn_id, session_id=session_id,
                                 created_at=stamp, updated_at=stamp,
                                 cancel=cancel, context=context_data,
                                 work=work, events=deque(maxlen=self.buffer_events))
            record.sink = self._sink_for(record)
            record.thread = threading.Thread(target=self._run, args=(record,),
                                             name=f"turn-{turn_id}", daemon=True)
            self._records[turn_id] = record
            self._active_by_session[session_id] = turn_id
            self._active_count += 1
            thread = record.thread
        thread.start()
        return TurnHandle(self, record)

    def _run(self, record: _TurnRecord) -> None:
        result: Optional[TurnResult] = None
        try:
            assert record.work is not None and record.sink is not None
            result = record.work(record.sink, record.cancel)
        except BaseException as exc:  # a work-closure bug must not kill the manager
            message = f"{type(exc).__name__}: {exc}"
            result = TurnResult(id=record.id, session_id=record.session_id,
                                status=FAILED, error=message)
            self._publish(record.id, RuntimeEvent(TURN_ERROR, {
                "code": "internal_error", "message": message, "retryable": False,
            }), synthetic=True)
        with self._lock:
            if not record.terminal_seen:
                self._synthesize_terminal(record, result)
            record.result = result
            record.worker_done = True
            if record.finished_at is None:
                record.finished_at = self._clock()
            self._active_by_session.pop(record.session_id, None)
            self._active_count = max(0, self._active_count - 1)

    # ── the event path ──────────────────────────────────────────────────────

    def _publish(self, turn_id: str, event: RuntimeEvent,
                 *, synthetic: bool = False) -> None:
        """The sink handed to a work closure: envelope, buffer, fan out."""
        with self._lock:
            record = self._records.get(turn_id)
            if record is None or record.terminal_seen:
                return  # nothing is emitted after a terminal
            self._publish_locked(record, event, synthetic=synthetic)

    def _publish_locked(self, record: _TurnRecord, event: RuntimeEvent,
                        *, synthetic: bool = False) -> None:
        if record.status in (QUEUED, WAITING_APPROVAL):
            record.status = RUNNING
        seq = record.latest_seq + 1
        record.latest_seq = seq
        data = dict(event.data)
        if record.context is not None and event.type in (TURN_STARTED, TURN_COMPLETED):
            data.setdefault("context", record.context)
        if synthetic:
            data.setdefault("synthetic", True)
        envelope = TurnEvent(turn_id=record.id, session_id=record.session_id,
                             seq=seq, ts=now_iso(), type=event.type, data=data)
        overflow = len(record.events) == self.buffer_events
        record.events.append(envelope)
        if overflow and record.events:
            record.oldest_seq = record.events[0].seq
            record.truncated = True
        record.updated_at = envelope.ts
        if event.type in TERMINAL_EVENTS:
            record.terminal_seen = True
            record.terminal_seq = seq
            record.synthetic_terminal = synthetic
            record.status = _TERMINAL_OF_TYPE.get(event.type, record.status)
            record.error = data.get("message") if event.type == TURN_ERROR else None
            subscribers = list(record.subscribers)
            record.subscribers.clear()
        else:
            subscribers = list(record.subscribers)
        for subscription in subscribers:
            subscription._offer(envelope)

    def _synthesize_terminal(self, record: _TurnRecord,
                             result: Optional[TurnResult]) -> None:
        """A closure that returned without a terminal still gets one."""
        status = result.status if result is not None else FAILED
        if status == CANCELLED:
            event = RuntimeEvent(TURN_CANCELLED, {"reason": "user"})
        elif status == COMPLETED:
            event = RuntimeEvent(TURN_COMPLETED, {"content": None})
        else:
            event = RuntimeEvent(TURN_ERROR, {
                "code": "internal_error",
                "message": (result.error if result is not None and result.error
                            else "the turn ended without a terminal event"),
                "retryable": False})
        self._publish_locked(record, event, synthetic=True)

    def _sink_for(self, record: _TurnRecord) -> EventSink:
        def sink(event: RuntimeEvent) -> None:
            self._publish(record.id, event)
        return sink

    def _gap_event(self, record: _TurnRecord, *, reason: str,
                   resume_seq: int, requested_seq: int) -> TurnEvent:
        return TurnEvent(
            turn_id=record.id, session_id=record.session_id, seq=resume_seq,
            ts=now_iso(), type=STREAM_GAP,
            data={"reason": reason, "requested_seq": requested_seq,
                  "oldest_seq": record.oldest_seq,
                  "latest_seq": record.latest_seq})

    # ── snapshots, subscriptions, retention ─────────────────────────────────

    def get(self, turn_id: str) -> Optional[TurnSnapshot]:
        with self._lock:
            self._sweep_locked(self._clock())
            record = self._records.get(turn_id)
            if record is None:
                return None
            return self._snapshot(record)

    def _snapshot(self, record: _TurnRecord) -> TurnSnapshot:
        return TurnSnapshot(
            id=record.id, session_id=record.session_id, status=record.status,
            created_at=record.created_at, updated_at=record.updated_at,
            seq=record.latest_seq, oldest_seq=record.oldest_seq,
            latest_seq=record.latest_seq, result=record.result,
            error=record.error, context=record.context,
            truncated=record.truncated,
            synthetic_terminal=record.synthetic_terminal)

    def active_for(self, session_id: str) -> Optional[str]:
        with self._lock:
            return self._active_by_session.get(session_id)

    def wait(self, turn_id: str, timeout: Optional[float] = None
             ) -> Optional[TurnSnapshot]:
        """Block until the turn's worker finishes (or the timeout); snapshot."""
        with self._lock:
            record = self._records.get(turn_id)
            thread = record.thread if record is not None else None
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        return self.get(turn_id)

    def subscribe(self, turn_id: str, *, after: Optional[int] = None,
                  queue_size: Optional[int] = None) -> Subscription:
        with self._lock:
            record = self._records.get(turn_id)
            if record is None:
                expired = self._expired.get(turn_id)
                if expired is not None:
                    raise TurnExpired(turn_id, expired.status)
                raise UnknownTurn(turn_id)
            subscription = Subscription(
                self, record, after=after,
                queue_size=queue_size if queue_size is not None
                else self.subscriber_queue)
            if not record.terminal_seen:
                record.subscribers.append(subscription)
            return subscription

    def sweep(self, now: Optional[float] = None) -> int:
        with self._lock:
            return self._sweep_locked(self._clock() if now is None else now)

    def _sweep_locked(self, now: float) -> int:
        dropped = 0
        for turn_id, record in list(self._records.items()):
            if not record.worker_done or record.finished_at is None:
                continue
            if now - record.finished_at < self.completed_ttl:
                continue
            self._records.pop(turn_id, None)
            self._expired[turn_id] = _Expired(status=record.status, at=now)
            dropped += 1
        horizon = now - self.expired_ttl
        for turn_id, expired in list(self._expired.items()):
            if expired.at < horizon:
                self._expired.pop(turn_id, None)
        return dropped

    # ── cancellation and shutdown ───────────────────────────────────────────

    def _request_cancel(self, turn_id: str) -> bool:
        with self._lock:
            record = self._records.get(turn_id)
            if record is None or record.terminal_seen:
                return False
            token = record.cancel
        if token is not None:
            token.cancel()
        return True

    def cancel(self, turn_id: str) -> bool:
        return self._request_cancel(turn_id)

    def shutdown(self, *, cancel: bool = True, timeout: float = 10.0) -> None:
        """Cancel active turns, wait for workers, refuse new submissions."""
        with self._lock:
            self._closed = True
            records = list(self._records.values())
        if cancel:
            for record in records:
                if not record.terminal_seen and record.cancel is not None:
                    record.cancel.cancel()
        deadline = time.monotonic() + max(0.0, timeout)
        for record in records:
            thread = record.thread
            if thread is None or not thread.is_alive():
                continue
            thread.join(max(0.0, deadline - time.monotonic()))
