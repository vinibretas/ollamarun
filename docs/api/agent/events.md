# Event stream

`GET /v1/agent/turns/{id}/events` is the SSE stream for a turn. Each frame is
named after its event type and carries a `<turn_id>:<seq>` id, so a reconnect
resumes with `?after=N` or `Last-Event-ID`:

```console
$ curl -N 'http://127.0.0.1:11440/v1/agent/turns/trn_…/events?after=0'
id: trn_…:1
event: turn.started
data: {"id": "trn_…:1", "seq": 1, "type": "turn.started", "turn_id": "trn_…",
       "session_id": "ses_…", "ts": "…",
       "data": {"model": "qwen3:8b", "session_id": "ses_…"}}
id: trn_…:2
event: message.delta
data: {"id": "trn_…:2", "seq": 2, "type": "message.delta", "turn_id": "trn_…",
       "session_id": "ses_…", "ts": "…",
       "data": {"channel": "content", "text": "The repo is"}}
id: trn_…:3
event: tool.call
data: {"id": "trn_…:3", "seq": 3, "type": "tool.call", "turn_id": "trn_…",
       "session_id": "ses_…", "ts": "…", "data": {"name": "run_shell",
       "args": {"command": "ls -la"}, "detail": "ls -la", "round": 1}}
id: trn_…:4
event: tool.result
data: {"id": "trn_…:4", "seq": 4, "type": "tool.result", "turn_id": "trn_…",
       "session_id": "ses_…", "ts": "…", "data": {"name": "run_shell",
       "ok": true, "text": "total 42\n…", "duration": 0.03, …}}
id: trn_…:5
event: turn.completed
data: {"id": "trn_…:5", "seq": 5, "type": "turn.completed", "turn_id": "trn_…",
       "session_id": "ses_…", "ts": "…",
       "data": {"content": "…", "tool_calls": 1, "rounds": 2, "stats": {…}, …}}
```

## Event vocabulary

The stream carries `turn.started`, `message.delta`, `tool.call`/`tool.result`,
`approval.required`/`approval.resolved`, `turn.warning` and the terminal
`turn.completed`/`turn.error`/`turn.cancelled`, each with a deterministic
`<turn_id>:<seq>` id.

## Replay and gaps

A reconnecting client resumes with `Last-Event-ID`; a truncated buffer is
reported as `stream.gap`, and the [turn status](turns.md) endpoint is the
recovery path afterwards. A subscription without `?after=`/`Last-Event-ID` is
live-only — nothing is replayed, so a finished turn yields an empty stream;
catching-up clients pass `after=0` or read the status snapshot instead.

A dropped stream is an unsubscribe, not a cancellation — see
[Disconnects and cancellation](../index.md#lifecycle).
