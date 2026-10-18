# Turns and events (planned)

> **Planned, not built.**

The browser's transport is the [typed event stream](../api/agent/events.md);
this page is the plan for what the UI does with it.

## Rendering the stream

| Event | Planned UI |
| --- | --- |
| `turn.started` | Mark the turn pending; show `context.used/total` in a side gutter. |
| `message.delta` | Append to the active bubble (`channel: "content"`), or to a collapsible reasoning fold (`channel: "thinking"`). |
| `message.completed` | Freeze the round; a tool-calling round shows as intermediate. |
| `tool.call` | Open a card with the tool name and pretty-printed arguments. |
| `tool.result` | Fill the card in place, `ok`/failed, duration; link any artifact ids. |
| `approval.required` | Swap the card's footer for decision buttons — see [Approvals](approvals.md). |
| `artifact.created` | Attach the file to the conversation with a download link. |
| `turn.warning` | Inline notice (context pressure, missing tool support). |
| Terminal events | Settle the turn: completed, failed (with the stable error), cancelled. |

## Replay, gaps and reconnects

- `id: <turn>:<seq>` makes every frame resumable; the client persists the
  last id and reconnects with `Last-Event-ID`.
- `stream.gap` tells the page it fell behind the ring; the documented
  recovery is the [status snapshot](../api/agent/turns.md#the-snapshot),
  then a replay with `?after=`.
- The status snapshot doubles as the "reload the page" path: a turn that
  finished while the tab was closed is fetched whole, and its full replay
  (`?after=0`) rebuilds the transcript losslessly.

## Why a typed stream

A single SSE channel carries thinking, prose, tool calls, approvals and
artifacts as distinct events with stable names, so the client never parses
prose to decide what happened. That is also what makes an audit view
possible later: the stream *is* the record, and the session store keeps it.
