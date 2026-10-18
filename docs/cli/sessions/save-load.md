# Save and load

- Start/resume: `-c/--continue [ID]`, `--session NAME`, `-r/--resume` picker,
  `--new` to force a fresh session, `--title TEXT`, `--tag TAG` (repeatable),
  `--auto-title` to ask the model for a title after the first turn.

In the REPL, `/load [MODEL|SESSION]` opens a saved session or switches model,
`/save NAME` creates an Ollama model from the session, `/sessions [filter]`
browses saved sessions, and `/clear`/`/new` start over. See
[Slash commands](../commands.md).
