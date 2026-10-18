# Chat (planned)

> **Planned, not built.**

The chat page is the terminal loop in a browser: type a message, submit a
turn, watch it stream back, and keep going in the same session.

## The shape it will follow

1. **Submit** — `POST /v1/agent/turns` with the message (and a `session_id`
   once one exists). The `202` returns the turn id and URLs immediately.
2. **Follow** — `GET /v1/agent/turns/{id}/events` feeds the transcript:
   `message.delta` chunks append text as they arrive, `tool.call` /
   `tool.result` render inline cards, and the terminal event closes the
   pending bubble.
3. **Reconnect** — an `EventSource`-style client reconnects with
   `Last-Event-ID`; the [status snapshot](../api/agent/turns.md#the-snapshot)
   is the recovery path for anything longer than the event ring.
4. **Stop** — a cancel button is `POST .../cancel`; the stream ends in
   `turn.cancelled` and the partial answer stays on the session.
5. **Continue** — the next message reuses the session; the server keeps the
   history, so the client never has to.

## Choices already made for it

- **Thinking models** surface reasoning on `message.delta` with `channel:
  "thinking"`; the UI can collapse it exactly like the terminal's
  [thinking panel](../cli/rendering/thinking.md).
- **Tool output** is a first-class event, not a transcript string — a card
  can show arguments, status, duration and artifacts without parsing text.
- **Disconnects** are configurable per turn (`cancel_on_disconnect`), so a
  page closing a tab can either stop the turn or let it finish server-side.
- **Markdown and math** render client-side; the server always sends the raw
  text (the terminal's 2D math engine is a CLI presentation detail, not part
  of the protocol).

See [Turns](../api/agent/turns.md) and
[Event stream](../api/agent/events.md) for the exact shapes.
