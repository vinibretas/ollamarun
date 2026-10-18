# Storage

Every conversation is one atomic JSON file plus `index.json` under
`~/etc/data/ollamarun/sessions/` (prompt history in `~/etc/data/ollamarun/`,
honoring `OLLAMA_NOHISTORY`); this is a session store, not a cache — nothing
is deleted without going through `trash-put`.

- `--no-save` keeps a run out of the store entirely; `--json`/`--raw` cover
  scripting.
- `/set`, `/show`, `/load`, `/save` (`/api/create`), `/clear`, `/new`, `/bye`
  mirror ollama's REPL; `/undo`, `/retry`, `/edit`, `/branch`, `/context`,
  `/summarize`, `/image`, `/copy`, `/export`, `/stats`, `/title`, `/tag`,
  `/sessions` and the rest are described in [Slash commands](../commands.md).

Paths and the debug log are listed under
[Data and logs](../../config/data-and-logs.md).
