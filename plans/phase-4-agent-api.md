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
