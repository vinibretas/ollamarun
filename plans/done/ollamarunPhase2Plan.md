# ollamarun Phase 2 — Internal API and event model

Phase 1/1b delivered the `AgentRuntime` extraction, the event vocabulary
and the safety rails. Phase 2 builds the **in-process turn manager and
event envelopes** on top of that emission API — still no HTTP server.
Phase 3 maps this onto `/v1`; Phase 4 exposes the agent endpoints. This
plan follows the API layer plan (`ollamarunAPILayerPlan.md`) and the
Phase 1b closure table (`plans/done/ollamarunPhase1bPlan.md`). The
CLI/TUI must keep working and `python3 test_ollamarun.py` stays green
at every step.

## Status

Implemented and on `main`: the envelope vocabulary (`events.py`) and
the turn manager (`turns.py`). The lifecycle invariants called out in
the concurrency and replay sections are the Phase 2 completion work:
enforce them with regression tests before the HTTP layer relies on
them.

## Constraints

- **No TUI/Textual work in this plan.** The `--ui tui` front end is
  covered by separate user plans; nothing here fixes, changes or adds
  tests for Textual behavior. Structural mentions stay (the renumbered
  architecture list, modules that happen to cover the TUI, the "don't
  break the CLI/TUI" acceptance), but any TUI-specific fix is dropped on
  sight and left to those plans.

## Goal

- Every runtime event gets one deterministic, JSON-ready envelope.
- A `TurnManager` owns turn records, worker threads, subscribers,
  replay, buffers, TTL, cancellation, session exclusivity and a global
  concurrency cap.
- **Nothing changes in the runtime's emission API** — the manager wraps
  it. The CLI keeps `repl.event_sink` semantics; no CLI path imports
  the manager.
- No HTTP, no approvals parking, no wall-clock enforcement in this
  phase (assigned below).

## Deliverable: `ollamarun_lib/turns.py`

### Event envelope

```python
@dataclass(frozen=True)
class TurnEvent:
    turn_id: str
    session_id: str
    seq: int          # 1-based, +1 per event, single writer per turn
    ts: str           # helpers.now_iso()
    type: str         # events.py vocabulary
    data: dict        # JSON-ready payload

    @property
    def id(self) -> str: return f"{self.turn_id}:{self.seq}"

    def to_dict(self) -> dict:  # fixed key order: id, seq, type,
                                # turn_id, session_id, ts, data
```

- No random `evt_…` ids: `<turn_id>:<seq>` is unique, ordered,
  dedupable, and is exactly the `id:` the SSE layer will send. (Small
  refinement of the merged plan's example, recorded here.)
- `TurnResult` already exists in `runtime.py`; the manager stores one per
  turn and returns it from the snapshot.

### Vocabulary delta

- Add `STREAM_GAP = "stream.gap"` to `events.py`. It is transport-level:
  only the manager's replay path emits it, payload
  `{reason, requested_seq, oldest_seq, latest_seq}` with
  `reason: "truncated_buffer" | "slow_subscriber"`.
- Nothing else is added or renamed; `turn.warning` stays as shipped.

### `TurnManager` API

```python
EventSink = Callable[[RuntimeEvent], None]

class TurnManager:
    def __init__(self, *, max_active=4, buffer_events=10_000,
                 completed_ttl=86_400.0, subscriber_queue=1_000): ...
    def submit(self, *, session_id, work, turn_id=None, cancel=None,
               context=None) -> TurnHandle
    def get(self, turn_id) -> Optional[TurnSnapshot]
    def active_for(self, session_id) -> Optional[str]
    def subscribe(self, turn_id, after=None) -> Subscription
    def sweep(self, now=None) -> int          # drops expired completed turns
    def shutdown(self, *, cancel=True, timeout=10.0) -> None

class TurnHandle:
    id: str; session_id: str
    def cancel(self) -> None
    def snapshot(self) -> TurnSnapshot

class Subscription:
    def get(self, timeout=None) -> Optional[TurnEvent]  # None = try again
    def close(self) -> None
    # closed / gap_pending / at_terminal properties for the SSE layer

class TurnSnapshot:
    id, session_id, status, created_at, updated_at,
    seq, oldest_seq, latest_seq, result, error, cancelled
```

- `work: Callable[[EventSink, CancelToken], TurnResult]` — the host
  closure builds its hooks/presenter around the sink and runs
  `AgentRuntime`. The manager never imports `Repl`.
- `cancel` is the turn's `CancelToken`; the closure must hand it to
  `TurnHooks.cancel`. `handle.cancel()` sets it.
- `context: Callable[[], dict] | None` — see decoration below.

Typed errors (mapped to HTTP in Phase 4):

| Error | Phase 4 mapping |
|---|---|
| `SessionBusy(active_turn_id)` | 409 `session_busy` |
| `TooManyTurns(limit)` | 429 `rate_limit_error` |
| `UnknownTurn(turn_id)` | 404 |
| `TurnExpired(turn_id, status)` | 410 `gone` (events endpoint) |

### Lifecycle and terminal semantics

- `status`: `queued` → `running` → `completed` / `failed` / `cancelled`.
  `waiting_approval` is reserved for Phase 4 (the enum value exists, the
  manager never sets it yet).
- The worker thread is the only writer for its turn, so `seq` order is
  free; the manager is the only seql assigner.
- Runtime terminals (`turn.completed` / `turn.error` / `turn.cancelled`)
  are observed, not re-emitted. If the closure raises or returns without
  a terminal event, the manager synthesizes `turn.error` with
  `code: "internal_error"` (flag it in the snapshot as synthetic).
- Terminal event is the last one; subscribers see `at_terminal` and end.
- Completed turns are retained (events + snapshot) for
  `serve_turn_event_ttl`, then swept. Active turns never expire.
- **Durability**: turn records and event buffers are in-memory; a
  server restart destroys a live turn record even though the
  conversation stays in the session store. The session JSON is the
  durable record: clients must not treat turns as durable jobs, and
  `GET /v1/agent/turns/{id}` (Phase 4) answers 404/410 for whatever a
  restart or a sweep removed.

### Buffer and replay semantics

- Ring buffer per turn, capacity `serve_turn_event_buffer` (10,000).
  Overflow drops the oldest events and advances `oldest_seq`; the gap is
  recorded (`truncated_before`).
- `subscribe(turn_id, after=None)`:
  - `after=None`: live only (the caller already has history).
  - `after >= oldest_seq - 1`: replay `after+1 … latest`, then live.
  - `after < oldest_seq - 1`: first event is `stream.gap`
    (`reason: "truncated_buffer"`, with requested/oldest/latest), then
    replay from `oldest_seq`; the client must treat missed events as
    unknown and re-read the turn/session status.
  - terminal + buffer already swept: `TurnExpired`.
- Retention is enforced on every entry point: `subscribe()` sweeps
  expired records before its lookup, exactly like `get()` and
  `submit()`, so a completed turn past its TTL cannot be subscribed to
  just because nothing else triggered a sweep.
- Slow subscribers: a bounded queue; on overflow the subscription is
  sent a final `stream.gap` (`reason: "slow_subscriber"`) and closed.
  The turn continues; a missing subscriber never stalls it.

### Concurrency, cancellation, shutdown

- One daemon worker thread per turn; registry mutation under one lock.
- One active turn per session: a second `submit` raises `SessionBusy`.
- Global cap `serve_max_active_turns` (4): excess raises `TooManyTurns`.
- `cancel()` sets the token; the runtime stops at its next checkpoint and
  emits `turn.cancelled` (synthesized if it cannot).
- `shutdown(cancel=True, timeout=...)` cancels active turns, joins with a
  timeout, sweeps; idempotent.
- Every managed turn owns a `CancelToken`: `submit()` creates one when
  the caller does not supply it, so `handle.cancel()` and `shutdown()`
  always have a token to cancel.
- Worker startup is atomic with respect to shutdown: the worker thread
  starts while the registry lock is held, so `shutdown()` can never
  skip an accepted record because its worker had not started yet.
- Wall-clock turn timeout is **not** enforced here — Phase 4 owns it
  (`serve_turn_timeout`), because it belongs with the request lifecycle.

### Context decoration (closes Phase 1b item 6)

With a `context` provider the manager adds
`data["context"] = {"used", "total", "source"}` to `turn.started` and to
the terminal `turn.completed`, and stores it on the snapshot. The
runtime stays unaware; the server (Phase 4) passes
`repl.context_report`-derived data, which also retires
`HeadlessRenderer.context_total` as a source of truth.

### Config

Add to `DEFAULT_CONFIG` (`constants.py`): `serve_max_active_turns: 4`,
`serve_turn_event_buffer: 10000`, `serve_turn_event_ttl: 86400`
(seconds). Bind/port/token keys stay in Phase 4.

### Placement

- New `turns.py` as stage 17 (after `runtime.py`), shifting `repl.py`
  → 18, `tui.py` → 19, `run.py` → 20, `cli.py` → 21; update the
  AGENTS/CLAUDE numbering and the cross-module paragraph.
- `turns.py` imports `events`, `helpers`, `session`, `runtime`; nothing
  in the CLI path imports it; `__init__.py` exports it after `runtime`.

## Wiring (the seam Phase 4 will use, proven in Phase 2 tests)

```python
def work(sink, cancel):
    hooks = repl._turn_hooks()
    hooks.on_event = sink
    hooks.cancel = cancel
    presenter = HeadlessRenderer(on_event=sink, ...)
    hooks.present_round = make_presenter(presenter)   # stream -> deltas
    result = AgentRuntime(hooks).run()
    return runtime.to_turn_result(result, session_id=repl.session.id)

handle = manager.submit(session_id=repl.session.id, work=work,
                        context=lambda: context_block(repl))
```

## Tests

| Test | What it proves |
|---|---|
| `test_turn_event_envelope` | deterministic `to_dict`/`id`, seq starts at 1, ts/type/data |
| `test_turn_manager_lifecycle` | running → completed, result stored, terminal is last |
| `test_turn_manager_synthesized_error` | a raising closure still gets a terminal `turn.error` |
| `test_turn_manager_replay` | `after=None/0/k` replay windows and live tail |
| `test_turn_buffer_truncation` | small capacity → oldest dropped, `stream.gap` first on replay |
| `test_turn_expired` | swept terminal turn raises `TurnExpired` from subscribe |
| `test_turn_manager_session_busy` | second submit on the same session rejected with the active id |
| `test_turn_manager_max_active` | cap enforced with `TooManyTurns`, different sessions fine |
| `test_turn_manager_cancel` | cancel → `turn.cancelled`, status cancelled |
| `test_turn_subscriber_overflow` | slow subscriber gets `slow_subscriber` gap and closes; turn completes |
| `test_turn_manager_ttl_sweep` | TTL drop + idempotent `shutdown` |
| `test_turn_manager_context` | started/completed carry the context block |
| `test_turn_manager_headless_turn` | the full closure: fake client + headless Repl through the manager, enveloped events + `TurnResult` |
| existing suite | CLI/TUI unchanged, all green |

## Out of scope (assigned)

- HTTP endpoints, SSE framing, 409/410 mapping: **Phase 3/4**.
- Interactive approvals parking/timeout: **Phase 4**.
- Wall-clock turn timeout (`serve_turn_timeout`): **Phase 4**.
- Durable turn records: session JSON remains the record; Phase 4 decides
  whether to persist turn summaries.
- Server auto-compaction after turns: **Phase 4**.
- pyflakes gate, real 3.10 smoke: **Phase 7**. Textual/test-harness items
  are owned by the separate TUI plan (out of scope here).

## Acceptance

- CLI/TUI unchanged; full suite green after every commit.
- Envelopes deterministic; replay ordered and dedupable; a truncated
  history is distinguishable from a replay (`stream.gap`); terminal
  events are final.
- A turn survives subscriber disconnect; a slow subscriber cannot stall
  a turn.
- One active turn per session and the global cap enforced with typed
  errors.
- Cancellation is a property of every managed turn, shutdown cannot
  miss an accepted one, and retention applies to subscriptions too;
  each has a regression test.

## Implementation order

1. `events.py`: `stream.gap` + the `TurnEvent` envelope (exact tests).
2. `turns.py`: registry, worker, buffers, subscribe/replay, cancel, TTL,
   typed errors (the bulk of the tests).
3. Context decoration + the three `serve_*` config keys + the end-to-end
   headless closure test.
4. Lifecycle invariants (tokenless cancellation, startup/shutdown
   atomicity, retention on subscribe) + regression tests.
5. Version bump 1.47 + changelog entry.
6. Docs (AGENTS/CLAUDE numbering and notes).
