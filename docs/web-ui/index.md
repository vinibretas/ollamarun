# Web UI (planned)

> **Planned, not built.** This section describes the intended design so the
> API surface it depends on is documented in the same place. Nothing here
> runs today; the terminal harness and the [API](../api/index.md) do.

The web interface is the main item on the roadmap: a browser client on top
of `ollamarun serve`, speaking the [native agent API](../api/agent/turns.md)
as its only backend. The API already has the shape a browser needs — submit
a turn, follow its event stream, answer parked approvals, manage sessions
and files — because that is the same loop the terminal drives in-process.
Building the client on it is what makes it a *second front end* instead of a
parallel implementation.

Why this shape:

- **One contract.** The event stream, the approval protocol and the session
  store are already typed and tested; a browser never learns anything the
  API does not expose.
- **No in-process coupling.** The [TUI](../tui/index.md) talks to the
  runtime directly and is deprecated for exactly that reason. A web client
  that can only see HTTP keeps the boundary honest.
- **Local-first.** `ollamarun serve` binds to the LAN/tailnet by default and
  has no authentication; the web UI inherits that trust model — it is a
  client for your own machine, not a hosted service.

## The planned pages

| Page | Backed by |
| --- | --- |
| [Chat](chat.md) | `POST /v1/agent/turns` + the [event stream](../api/agent/events.md) |
| [Turns and events](turns-and-events.md) | the typed `turn.*`/`message.*`/`tool.*` events |
| [Approvals](approvals.md) | `approval.required` / `POST /v1/agent/approvals/{id}` |
| [Sessions](sessions.md) | the `/v1/sessions` routes and the shared store |

## What it changes for the CLI

Nothing, by design: the API layer is additive, and the terminal keeps using
the same runtime the API runs. The deprecation note on the TUI stands
independently — that rewrite is about its own coupling, not a prerequisite
for the web UI.
