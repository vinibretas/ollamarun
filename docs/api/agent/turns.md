# Turns

The same `ollamarun serve` process exposes ollamarun's own endpoints:
instead of one inference call, a client submits a **full agent turn** — the
internal tool loop, approvals and session persistence run server-side — and
follows it over Server-Sent Events.

| Endpoint | What it does |
| --- | --- |
| `POST /v1/agent/turns` | Start a full agent turn (`202` + the turn id and its URLs). |
| `GET /v1/agent/turns/{id}` | Snapshot: status, seq, result, error (`404` unknown, `410` swept). |
| `GET /v1/agent/turns/{id}/events` | SSE turn events — see [Event stream](events.md). |
| `POST /v1/agent/turns/{id}/cancel` | Request cancellation. |

## Starting a turn

```json
POST /v1/agent/turns
{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
  "settings": {"tools": false, "system": "n/a"}
}
```

The response is immediate — the turn runs in a worker thread:

```json
{
  "id": "turn_2e7a284c4149",
  "object": "turn",
  "session_id": "20261018-131729-6386",
  "status": "queued",
  "created_at": "2026-10-18T13:17:29.948845-03:00",
  "status_url": "/v1/agent/turns/turn_2e7a284c4149",
  "events_url": "/v1/agent/turns/turn_2e7a284c4149/events",
  "cancel_url": "/v1/agent/turns/turn_2e7a284c4149/cancel"
}
```

For an existing session (`session_id`) the messages are the **delta**; for a
new one they seed the history. Turns persist by default (`persist: false`
runs ephemeral — the results still land in-memory and artifacts still
register). `model` is required when creating a session; continuing one may
omit it.

### `settings`

| Key | Default | Meaning |
| --- | --- | --- |
| `tools` | true | Run the tool loop. |
| `max_rounds` | 8 | Tool rounds cap. |
| `approval` | `"policy"` | `"policy"` decides via the allowlist/rules (denying the rest), `"interactive"` parks undecided calls — see [Approvals](approvals.md). |
| `system` | null | System-prompt override (`"auto"`, text, or absent for none). |
| `parameters` | {} | Validated Ollama options merged into the request. |
| `num_ctx` | — | A window, or `"max-gpu"`/`"max-cpu"` to estimate at turn start (see [Context fits](fits.md)). |
| `cpu_percent` | — | With `num_ctx: "max-cpu"`. |
| `gpu_layers` | — | Placement spec; a change unloads the resident model. |
| `cancel_on_disconnect` | false | Cancel when the events stream drops (default: keep running). |

`parameters` validation is strict: known Ollama options get type and range
checks, unknown names pass through bounded to scalars and small lists, and
non-finite numbers or nested structures are refused with a 400.

## The snapshot

The status endpoint gives the same result with no stream:

```json
{
  "id": "turn_2e7a284c4149",
  "object": "turn",
  "session_id": "20261018-131729-6386",
  "status": "completed",
  "seq": 4,
  "oldest_seq": 1,
  "latest_seq": 4,
  "truncated": false,
  "context": {"used": null, "total": 8192, "source": "loaded runner"},
  "error": null,
  "result": {
    "status": "completed",
    "content": "pong",
    "thinking": null,
    "tool_calls": null,
    "rounds": 0,
    "usage": {"prompt_tokens": 32, "completion_tokens": 2, "total_tokens": 34},
    "stats": {"total_duration": 1.658, "load_duration": 0.003, "prompt_tokens": 32,
              "completion_tokens": 2, "eval_tps": 12.27, "prompt_tps": 21.51, "done_reason": "stop"},
    "artifacts": [],
    "fit": null
  }
}
```

- `status` is `queued` / `running` / `waiting_approval` / `completed` /
  `failed` / `cancelled`.
- `seq`/`oldest_seq`/`latest_seq`/`truncated` describe the event ring (see
  [Replay and gaps](events.md#replay-and-gaps)) — a late client can resume
  exactly.
- `context` is the live window (used/total/source) when known.
- `result.artifacts` lists published [artifacts](artifacts.md);
  `result.fit` carries a turn-start [fit](fits.md) when one ran.
- A failed turn keeps a stable `error` message; the raw crash text goes to
  the `--debug` log, not the client.

## Cancelling

```console
$ curl -s -X POST http://127.0.0.1:11440/v1/agent/turns/turn_d1b7a7128a1b/cancel
{"id": "turn_d1b7a7128a1b", "object": "turn", "cancel_requested": true, "status": "running"}
```

The response reflects the *request*; the cancel resolves asynchronously into
a `turn.cancelled` terminal and the snapshot's `status: "cancelled"`. A
partial answer stays on the session. Cancellation aborts the in-flight
generation, kills cancellable subprocess tools and resolves a parked
approval as cancelled.

## Records and limits

- Turn records live in memory — a restart loses live turns (`404` unknown,
  `410` for swept ones), while session history stays in the store.
- One active turn per session: a second submission gets `409 session_busy`
  with the `active_turn_id`; editing or deleting the session is refused the
  same way while the turn runs (or a fit holds it).
- A global cap (`serve_max_active_turns`, default 4) answers `429`:

  ```json
  {"error": {"message": "too many active turns", "type": "rate_limit_error",
   "param": null, "code": "rate_limit_error", "limit": 4}}
  ```

- Completed turns are kept for `serve_turn_event_ttl` (default 24 h; `0`
  keeps them) so late clients can still replay.
- `serve_turn_timeout` (default 0 = off) is a wall-clock cap; hitting it
  cancels the turn with reason `timeout`.

- [Event stream](events.md) — following a turn.
- [Approvals](approvals.md) — resolving parked tool calls.
- [Sessions](sessions.md) — the durable record.
