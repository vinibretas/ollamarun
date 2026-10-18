# Event stream

`GET /v1/agent/turns/{id}/events` is the SSE stream for a turn. Each frame
is named after its event type and carries a `<turn_id>:<seq>` id, so a
reconnect resumes with `?after=N` or `Last-Event-ID`.

A real capture of a completed simple turn (`?after=0` replay), frames
trimmed to their payloads:

```console
$ curl -N 'http://127.0.0.1:11440/v1/agent/turns/turn_2e7a284c4149/events?after=0'
id: turn_2e7a284c4149:1
event: turn.started
data: {"id": "turn_2e7a284c4149:1", "seq": 1, "type": "turn.started", "turn_id": "turn_2e7a284c4149",
       "session_id": "20261018-131729-6386", "ts": "…",
       "data": {"model": "llama3.2:3b", "session_id": "…",
                "context": {"used": null, "total": 8192, "source": "loaded runner"}}}

id: turn_2e7a284c4149:2
event: message.delta
data: {… "data": {"channel": "content", "text": "pong"}}

id: turn_2e7a284c4149:3
event: message.completed
data: {… "data": {"round": 0, "content": "pong", "thinking": null,
                  "tool_calls": 0, "cancelled": null, "error": null}}

id: turn_2e7a284c4149:4
event: turn.completed
data: {… "data": {"content": "pong", "thinking": null, "tool_calls": 0, "rounds": 0,
                  "stats": {"total_duration": 1.658, "eval_tps": 12.27, …},
                  "context": {…}}}
```

Every payload shares the envelope: `id`, `seq`, `type`, `turn_id`,
`session_id`, `ts`, and a `data` object per type.

## Event vocabulary

| Type | When | Notable `data` |
| --- | --- | --- |
| `turn.started` | Worker picked the turn up. | `model`, `session_id`, `context`. |
| `message.delta` | Each streamed chunk. | `channel` (`content`/`thinking`), `text`. |
| `message.completed` | One round finished. | `round`, `content`, `thinking`, `tool_calls` count, `cancelled`, `error`. |
| `tool.call` | Model asked for a tool. | `name`, `args`, `detail`, `round`. |
| `tool.result` | The tool finished (or was denied). | `name`, `ok`, `text`, `duration`, and `artifacts` ids when files were published. |
| `approval.required` | A call parked for a decision. | The full call + `reason` + `suggestions` + `approval_id` — see [Approvals](approvals.md). |
| `approval.resolved` | A decision landed. | `action`, `run`, `note`. |
| `artifact.created` | A file was published. | The full [ArtifactRef](artifacts.md). |
| `session.fit` | A turn-start context fit ran. | The fit result — see [Context fits](fits.md). |
| `turn.warning` | Non-fatal advice (context 80%, tool-support missing). | `message`. |
| `turn.completed` / `turn.error` / `turn.cancelled` | Terminal events. | `content`, `rounds`, `stats`, `context`, `result` shape. |

A tool call and its result come **before** the round's `message.completed`
for the round that requested them; an `artifact.created` rides between the
approval decision and the `tool.result` that references it.

## Replay and gaps

- A subscription **without** `?after=`/`Last-Event-ID` is live-only: nothing
  is replayed, so a finished turn yields an empty stream. Catching-up
  clients pass `after=0` or read the [status snapshot](turns.md#the-snapshot).
- `Last-Event-ID: <turn>:<seq>` resumes exactly; `?after=N` does the same.
- If the ring has overwritten the client's position
  (`serve_turn_event_buffer`, default 10 000 events), the stream emits a
  `stream.gap` frame and the client recovers from the status snapshot.
- Heartbeats are `: ping` comment lines every `serve_sse_ping_interval`
  (default 15 s) — they keep intermediaries from timing the stream out and
  are safe for any SSE client to ignore.

## Disconnects

A dropped stream is an **unsubscribe**, not a cancellation: the turn keeps
running and its results land in the session (checked live — status
`completed` after the client vanished). With
`"settings": {"cancel_on_disconnect": true}` the drop cancels instead
(checked live — status `cancelled` within a couple of seconds). Explicit
cancellation is `POST .../cancel` at any time.
