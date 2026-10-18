# Storage

Sessions are the durable half of the harness. One conversation is one JSON
file plus a shared index — a store, not a cache, and nothing is deleted
without going through the trash.

## Where things live

| Path | What |
| --- | --- |
| `~/etc/data/ollamarun/sessions/<id>.json` | One session file per conversation. |
| `~/etc/data/ollamarun/index.json` | The list view's index: id, title, model, host, timestamps, counts, tags. |
| `~/etc/data/ollamarun/index.json.lock` | The flock file guarding index read-modify-write. |
| `~/etc/data/ollamarun/history` | Prompt input history (unless `OLLAMA_NOHISTORY`). |
| `~/etc/data/ollamarun/artifacts/` | The serve artifact store ([Artifacts](../../api/agent/artifacts.md)). |
| `~/etc/data/ollamarun/ctxfit-calibration.json` | Context-fit measurements ([Probes and calibration](../context/probes.md)). |

Session ids look like `20261018-122556-a496` (timestamp plus a short random
suffix). Anywhere an id is taken — `--show`, `--export`, `-c`, `/load` — a
unique prefix or an unambiguous title reference works, so you rarely type the
whole thing.

## The session file

```json
{
  "id": "20261018-121947-845b",
  "schema": 1,
  "model": "llama3.2:3b",
  "host": "http://127.0.0.1:11434",
  "remote": null,
  "title": "demo",
  "system": null,
  "parameters": {},
  "options": {},
  "think": null,
  "keepalive": "2h",
  "format": null,
  "gpu_layers": null,
  "tags": [],
  "stats": {
    "turns": 7,
    "prompt_tokens": 1087,
    "completion_tokens": 14,
    "duration_s": 10.55,
    "last_eval_tps": 13.75,
    "last_prompt_tokens": 207,
    "last_completion_tokens": 2
  },
  "tool_allow": [],
  "tool_rules": [],
  "tools_disabled": [],
  "artifacts": [],
  "created_at": "2026-10-18T12:19:47.745015-03:00",
  "updated_at": "2026-10-18T12:19:52.281753-03:00",
  "messages": []
}
```

Messages carry `role`, `content` and `ts`, plus whatever the turn produced —
the same data `--show` and the exports read back:

| Field | On | Meaning |
| --- | --- | --- |
| `thinking` | assistant | The reasoning text a thinking model produced. |
| `tool_calls` | assistant | The call array the model asked for. |
| `tool_name` | tool | Which tool produced this result. |
| `images` | user | Attached images (path/name/sha1/size, base64 when embedded). |
| `files` | user | `@file` mentions that were inlined. |
| `cancelled` | either | The turn was interrupted here. |

Settings live on the session too: `system`, `parameters`, `think`,
`keepalive`, `format`, `gpu_layers`, `tools_disabled`, the approval
`tool_allow`/`tool_rules`, and the tts fields — so `/load` restores the
whole working context, not just the chat.

## Writes are atomic, and the index is locked

Saving goes through a temp file named per writer plus `os.replace`, so an
interrupted write can never leave a half session. The index read-modify-write
runs under a shared `RLock` **and** an exclusive `flock` on
`index.json.lock`: a CLI saving while `ollamarun serve` runs cannot lose
entries, and `--reindex` rebuilds the index from the session files if it ever
drifts.

## Deletes go to the trash

`--delete ID` (asks; `-y` skips) and the store's delete move the session
file to the XDG trash via `trash-put` — never a raw unlink:

```console
$ ollamarun --delete 20261018-111147-a9d6 -y
trashed 20261018-111147-a9d6
```

The file is recoverable from `~/.local/share/Trash/` (with a `.trashinfo`
recording where it came from), and `--list` no longer shows it. In the REPL,
`/clear` empties the context but keeps the session's settings and starts a new
id under the same file; `/new` saves the current session and starts a fresh
one.

## Not every run is saved

`--no-save` keeps a run out of the store entirely — useful for scripting and
for throwaway checks — and `--json`/`--raw` runs still respect it. The REPL's
prompt history is the only other thing persisted by default, and
`OLLAMA_NOHISTORY` disables that too.

- [Save and load](save-load.md) — resuming conversations.
- [Search and export](search-export.md) — getting data back out.
