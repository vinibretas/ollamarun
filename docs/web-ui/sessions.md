# Sessions (planned)

> **Planned, not built.**

The web UI's session list is the [storage](../cli/sessions/storage.md) view
over HTTP — the same files the CLI writes, served by the same routes.

## The routes it will use

| Route | Planned use |
| --- | --- |
| `GET /v1/sessions` | The sidebar list: title, model, updated time, tags, message counts. |
| `POST /v1/sessions` | "New chat": create up front (title, model, tags) before the first turn. |
| `GET /v1/sessions/{id}` | Open a session: summary + `history` + stats + settings, without replaying turns. |
| `PATCH /v1/sessions/{id}` | Rename, retag, switch model, or set `num_ctx`/`gpu_layers` per the form. |
| `DELETE /v1/sessions/{id}` | Trash a session (recoverable, `trash-put` underneath). |
| `GET /v1/agent/turns/{id}/events?after=0` | Replay a stored turn's transcript when opening recent sessions. |

## Behaviours it inherits

- **Busy sessions** — a session with an active turn or fit refuses edits and
  deletes with `409 session_busy`; the UI should disable those controls and
  say why, the same way the CLI refuses.
- **First-class context** — `num_ctx` and `gpu_layers` are editable session
  fields; a placement change unloads the resident model server-side, so the
  form can save and the next turn reloads correctly.
- **Artifacts are not cascade-deleted** — deleting a session keeps its
  files until the artifact TTL expires, so a "deleted" link may still
  resolve for a while; the UI should say so rather than 404 silently.
- **The store is shared** — the CLI, the TUI and the web UI read and write
  the same files, and the index is locked against concurrent writers. A
  session started in the terminal appears in the browser untouched.

See [Sessions API](../api/agent/sessions.md) for the exact fields and
[Search and export](../cli/sessions/search-export.md) for what the CLI
offers on the same data (a future web search/filter would mirror
`--search`).
