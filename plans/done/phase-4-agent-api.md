# Phase 4 — Native agent endpoints

Your routes, with the turn lifecycle made first-class:

| Endpoint | Purpose |
|---|---|
| `POST /v1/agent/turns` | Start a full agent turn; returns `202` with turn id, status, event URL |
| `GET /v1/agent/turns/{id}` | Status snapshot (`queued/running/waiting_approval/completed/failed/cancelled`) |
| `GET /v1/agent/turns/{id}/events` | SSE subscribe/replay (`Last-Event-ID` or `?after=`) |
| `POST /v1/agent/turns/{id}/cancel` | Request cancellation |
| `POST /v1/agent/approvals/{id}` | Resolve a pending approval |
| `GET/POST /v1/sessions` | List/create sessions |
| `GET/DELETE /v1/sessions/{id}` | Retrieve/delete a session |
| `PATCH /v1/sessions/{id}` | Optional: title/model/system/tool settings |

**Turn status:** `GET /v1/agent/turns/{id}` returns the snapshot
(status, session, timestamps, latest event sequence, terminal result
and error); 404 for a turn that never existed, 410 for one whose
retained record was swept (distinct from 404). After a `stream.gap`
this endpoint is the recovery path: clients re-read it instead of
inferring state from incomplete events.

**Turn request:**

```json
{
  "session_id": "optional",
  "model": "optional; defaults to session or serve default",
  "messages": [ { "role": "user", "content": "..." } ],
  "persist": true,
  "settings": {
    "tools": true,
    "max_rounds": 8,
    "approval": "interactive",
    "system": "optional override",
    "parameters": {},
    "cancel_on_disconnect": false
  }
}
```

For an existing session, `messages` is the delta for this turn, not the full history. For a new session, it seeds the history. This avoids the client-history-vs-harness-tool-history mismatch. Agent turns persist by default; `persist: false` explicitly opts into ephemeral execution (the turn result is still returned, nothing is written to the session store).

**Approvals:** `approval.required` carries `approval_id`, the tool call, arguments, a human-readable detail, the reason, and the same choices the CLI/TUI offer (`run once`, `remember exact`, `allow pattern`, `auto`, `deny`) with rule payloads. Resolve via `POST /v1/agent/approvals/{id}`.

- Headless hosts never prompt a terminal. `approval: "interactive"` waits on the API: the turn enters `waiting_approval` and emits the event. If no client resolves it, it stays pending until `approval_timeout` (default 300s, configurable), then resolves as `action: "timeout"` and the tool result is `ok: false` with “not run: approval timed out”; the model continues.
- `approval: "policy"` skips the prompt and applies the serve/session tool policy directly: allowlisted or rule-covered calls run, everything else is denied with a note.
- **Never auto-approve on timeout or on a missing client.** Approval state is explicit (`pending` → `resolved`) and cannot be confused with successful execution.
- Each pending approval has its own ID tied to one turn and one exact
  tool operation; the turn enters `waiting_approval` only while a
  decision is genuinely pending. The first resolution wins: a
  duplicate, expired or already-resolved decision is rejected and
  must not execute the operation a second time.
- The worker waits for a decision without holding the manager’s
  registry lock; pending approvals count toward the global active-turn
  limit like any running turn; cancelling or shutting down while
  waiting resolves the approval as cancelled and stops the worker.

**Session concurrency:** exactly one active turn per session. A competing `POST /v1/agent/turns` for a busy session is rejected with `409 Conflict` and `{error: {type: "conflict_error", code: "session_busy", active_turn_id: "turn_..."}}`. No silent queueing, no concurrent mutation of the session or its tool rules. The one exception is resolving a pending approval, which is not a turn.

**Cancellation:** `POST /v1/agent/turns/{id}/cancel` sets the turn’s cancel token, resolves any pending approval with `action: "cancelled"`, aborts the in-flight inference stream through the session’s own `OllamaClient.abort_stream()`, and kills cancellable subprocesses (process group). The runtime checks the token between rounds and between tool calls; a tool that is already running may finish its current operation. Arbitrary subprocesses cannot always be interrupted immediately — document this in the API docs. The turn emits `turn.cancelled` and its terminal state is persisted.

**Disconnect semantics:** the turn continues and buffers events; results are saved to the session. The UI sets `cancel_on_disconnect: true` for interactive turns. On process shutdown, active turns are cancelled and their state is persisted. No orphaned workers.

**Durability:** turn records and event buffers are in-memory. A
server restart destroys a live turn record even though session
history persists; clients must not treat turns as durable jobs, and
an expired or missing turn is recovered from the session store.

**Session exclusivity:** the manager enforces one active turn per
session inside its own process. The initial contract is
single-writer: a persistent session must not be actively modified by
the CLI and the API at the same time. Shared cross-process session
locking is a later upgrade; distributed coordination is out of scope.

## Recorded implementation decisions

- **The headless `Repl` is the seam.** A turn builds a headless `Repl`
  (a default CLI-args namespace) with its `cancel_token`/`tool_ctx.cancel`
  rewired to the turn's token and `tools_enabled`/`tools_max_rounds`/
  `no_save` from the request, then runs `AgentRuntime` through
  `TurnManager`. `hooks.execute_tools` is cleared so the runtime's own
  tool execution carries the API approver and event sink into every round
  (Repl's wrapper would rebuild hooks without them).
- **Turn request:** `settings.approval` defaults to `"policy"`;
  `"interactive"` is the client's explicit opt-in. The serve tool policy
  is never `auto` (a config `auto` degrades to allowlist) and `--yes`
  stays the only CLI auto path. Messages are the delta for an existing
  session and the seed for a new one; `model`/`system`/`parameters` are
  session mutations (persisted), while `tools`/`max_rounds`/`approval`/
  `cancel_on_disconnect` are per-turn.
- **Approvals:** parking lives in `agent_api` (`PendingApproval` +
  `_first_decision`, first wins) with the wait loop inside the worker;
  the manager's new `mark_waiting_approval()` publishes
  `waiting_approval` (and unparking happens before the next event). A
  resolution reuses the CLI's session mutations (`tool_allow`,
  `tool_rules`); `auto` is scoped to the rest of that turn. The timeout
  (`serve_approval_timeout`, default 300 s, 0 = none) resolves as
  `action: "timeout"`, note "approval timed out", and cancellation as
  `action: "cancelled"`.
- **Cancellation/timeout:** `CancelToken.reason` (new) carries why and
  the runtime reports it on `turn.cancelled`. `POST .../cancel` and the
  `serve_turn_timeout` watchdog (0 = disabled) set the reason, call
  `abort_stream()` and resolve pending approvals as cancelled.
- **Registration and lifecycle:** one router serves both surfaces
  (`add_agent_routes()` appends to `openai_api.build_router()`'s);
  `serve_main()` shuts the agent API down after `run_forever` (pending
  approvals → cancelled, then manager shutdown). `TurnManager.expired()`
  distinguishes 410 from 404. Session CRUD goes through `SessionStore`
  (delete = trash-put; a failed trash is a sanitized 500), and a session
  with an active turn cannot be deleted (409).
- **Phase 1b item 13 closed:** the context warning is emitted as a
  `turn.warning` event and `auto_compact: "auto"` compacts before the
  first round, through the new `TurnHooks.on_started` hook;
  `Repl.context_warning()`/`maybe_auto_compact()`/`summarize()` are now
  headless-safe while the CLI prints exactly as before.
- **Auth and limits:** the endpoints ride the locked LAN/no-auth
  contract. `serve_agent`, the `serve_models` allowlist and the other
  Phase 6 policy keys stay for Phase 6.

## Acceptance evidence

`test_agent_turn_lifecycle`, `test_agent_turn_controls`,
`test_agent_approvals`, `test_agent_approval_edge`, `test_agent_policy_mode`,
`test_agent_disconnect_policy`, `test_agent_turn_timeout`,
`test_agent_sessions_api`, `test_agent_turn_validation`,
`test_agent_turn_expiry`, `test_agent_context_warning`; live-checked
against Ollama 0.40 with a real policy-mode turn, an interactive
park/resolve, event replay and session cleanup.

## Post-review fixes (v1.50)

The review of the landing confirmed three defects; all were fixed with
regressions before the phase was closed:

- **Retention on the events endpoint.** `TurnManager.subscribe()` now
  sweeps expired records before its lookup, exactly like `get()` and
  `submit()`, so `GET /v1/agent/turns/{id}/events` answers 410 for a turn
  past `serve_turn_event_ttl` even when no other request swept it first.
  The Phase 2 `test_turn_expired` had masked the gap by calling `wait()`
  (which sweeps through `get()`); it now joins the worker directly, and
  the HTTP expiry test calls the events route before any status read.
- **Sanitized internal errors.** The manager's synthesized error terminals
  (`_run`'s crash path and `_synthesize_terminal`) publish the shared
  stable `INTERNAL_ERROR_MESSAGE`; the raw exception text goes to the
  `--debug` log and stays on the internal `TurnResult.error`. The SSE
  stream no longer exposes what the status endpoint (`turn_body`) was
  already sanitizing, and `turn_body` now uses the shared constant.
- **Session edits during a turn.** `PATCH /v1/sessions/{id}` is refused
  with the same 409 `session_busy` (active turn id included) as DELETE
  while the session has an active turn, closing a lost-update race
  against the worker's session saves.

Minor observations were recorded, with no behavior change: session routes
resolve refs like the CLI (exact id, id prefix or title), the
"cancelled by user" approval note on timeout/disconnect resolution, and
`Last-Event-ID` resume having no dedicated test.
