# Turns

The same `ollamarun serve` process also exposes ollamarun's own endpoints:
instead of one inference call, a client submits a **full agent turn** — the
internal tool loop, approvals and session persistence run server-side — and
follows it over Server-Sent Events.

| Endpoint | What it does |
| --- | --- |
| `POST /v1/agent/turns` | start a full agent turn (`202` + the turn id and its URLs) |
| `GET /v1/agent/turns/{id}` | snapshot: `queued`/`running`/`waiting_approval`/`completed`/`failed`/`cancelled`, seq, result, error (404 unknown, 410 swept) |
| `GET /v1/agent/turns/{id}/events` | SSE turn events (`id: <turn>:<seq>`, typed `event:` names; replay with `?after=N` or `Last-Event-ID`) |
| `POST /v1/agent/turns/{id}/cancel` | request cancellation |

```json
POST /v1/agent/turns
{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "Clean up /tmp/build"}],
  "persist": true,
  "settings": {
    "tools": true, "max_rounds": 8,
    "approval": "interactive",
    "system": null, "parameters": {},
    "cancel_on_disconnect": false
  }
}
```

For an existing session (`session_id`) the messages are the delta; for a new
one they seed the history. Turns persist by default (`persist: false` runs
ephemeral). `settings.parameters` — and the sessions' `parameters` field,
including the PATCH merge — is validated: known Ollama options get their
type and range checked (the same ranges the OpenAI mapper enforces),
unknown names pass through but are bounded to scalars and small lists with
size caps, and non-finite numbers or nested structures are refused with a
400.

Concrete requests — a new session seeds the history, an existing one gets the
delta (`persist` defaults to true) — and the turn's URLs come back immediately:

```console
$ curl -s http://127.0.0.1:11440/v1/agent/turns \
    -H 'Content-Type: application/json' -d '{
      "session_id": "ses_…",
      "messages": [{"role": "user", "content": "now file it under reports"}],
      "settings": {"approval": "interactive"}
    }'
{"id": "trn_…", "object": "turn", "session_id": "ses_…", "status": "queued",
 "created_at": "…", "status_url": "/v1/agent/turns/trn_…",
 "events_url": "/v1/agent/turns/trn_…/events",
 "cancel_url": "/v1/agent/turns/trn_…/cancel"}

$ curl -s http://127.0.0.1:11440/v1/agent/turns \
    -H 'Content-Type: application/json' -d '{
      "model": "qwen3:8b",
      "messages": [{"role": "user", "content": "Summarize this repo"}],
      "persist": false,
      "settings": {"max_rounds": 4, "system": "One paragraph, no lists.",
                   "num_ctx": "max-gpu", "gpu_layers": "80%"}
    }'
```

The snapshot endpoint gives the same result with no stream (and a finished
turn has no live events, so this — or `?after=0` — is how a late client
catches up):

```console
$ curl -s http://127.0.0.1:11440/v1/agent/turns/trn_…
{"id": "trn_…", "object": "turn", "session_id": "ses_…", "status": "completed",
 "seq": 5, "result": {"status": "completed", "content": "…", "rounds": 2,
 "usage": {"prompt_tokens": 412, "completion_tokens": 96, "total_tokens": 508},
 "artifacts": [], "fit": null}, "error": null}

$ curl -s -X POST http://127.0.0.1:11440/v1/agent/turns/trn_…/cancel
{"id": "trn_…", "object": "turn", "cancel_requested": true, "status": "running"}
```

## Records and limits

Turn records live in memory — a restart loses live turns (404/410), while
session history stays under `~/etc/data/ollamarun/sessions/`. One active
turn per session (`409 session_busy` with `active_turn_id`; editing or
deleting the session is the same 409 while the turn runs), a global cap
(config `serve_max_active_turns`, `429`), and an optional wall-clock limit
per turn (config `serve_turn_timeout`, `0` = no limit).

- [Event stream](events.md) — following a turn.
- [Approvals](approvals.md) — resolving parked tool calls.
- [Sessions](sessions.md) — the durable record.
- [Disconnects and cancellation](../index.md#lifecycle).
