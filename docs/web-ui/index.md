# Web UI (planned)

The main thing on the roadmap is a web interface on top of `ollamarun serve`
as its backend. The native agent API already has the shape a browser client
needs — submit a turn, follow its event stream, answer parked approvals,
manage sessions and files — because that is the same loop the terminal drives.
See the [Agent API](../api/agent/turns.md) for the surface it will build on.

The planned client mirrors the terminal loop, so the pages here will follow
the same features:

- [Chat](chat.md) — submit a turn and stream the answer.
- [Turns and events](turns-and-events.md) — the typed event stream.
- [Approvals](approvals.md) — answer parked tool decisions.
- [Sessions](sessions.md) — browse, resume and manage conversations.

The existing Textual [TUI](../tui/index.md) is deprecated in the same light:
it will be rewritten as a pure client of that API instead of reaching into the
runtime in-process, so every front end speaks one interface.
