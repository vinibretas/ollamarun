# Save and load

Every interactive session and every one-shot (unless `--no-save`) lands in
the store, and any of them can be picked back up — from the command line or
from inside the REPL.

## Resuming from the command line

| Form | Meaning |
| --- | --- |
| `-c [ID]`, `--continue [ID]` | Continue the most recent session, or the given one (an id, an id prefix, or an exact title followed by a separate prompt). |
| `-r`, `--resume` | Browse saved sessions and pick one. |
| `--session NAME` | Continue the session named `NAME`, or create it. |
| `--new` | Force a fresh session even with `-c`/`--session`. |

If the session is still running in a [background host](background.md) — the
terminal was closed or detached — `-c` attaches to it instead of starting a
new process: a running turn is replayed and continues, and a turn that
finished while away is shown. A second terminal takes the session over.

A named session keeps its name as its title, and later runs can omit the
model — it comes from the session:

```console
$ ollamarun --no-tools --session demo llama3.2:3b "Remember the number 41. Reply with OK only."
OK
$ ollamarun --no-tools --session demo "What number did I ask you to remember? Answer with the number only."
41
```

`-c` continues the most recently updated session, with its model:

```console
$ ollamarun --no-tools -c "Now add one to that number. Answer with the number only."
42
```

!!! info "`-c` and prompt-like titles"

    Stored titles are derived from prompts, so `-c` only reads a token as a
    session reference when it is an exact id, an id prefix, or an exact title
    with a separate prompt following it (`-c "demo session" "and the next
    question"`). A lone token that equals (or contains) a title stays the
    prompt, so reusing prompt wording continues the latest session.

`-r` opens the same browser as `/sessions`: a table where you type a number,
an id, a filter to narrow, or `q` to quit.

Resuming also reconnects to the machine the session last ran on, unless
`--host`/`--remote` override it — see [Remote](../remote.md).

## `/load` and switching

- `/load SESSION` resumes a saved session; when its model differs from the
  current one the harness asks which model to run and unloads the current
  one before switching.
- `/load MODEL` (or `/model NAME`) switches the model instead, unloading the
  previous resident unless it is an embedding model.

## Saving a model out of a session

`/save NAME` bakes the current conversation into an Ollama model, using the
session as its messages and the effective system prompt as its `SYSTEM` — a
snapshot you can `ollamarun NAME` later:

```text
llama3.2:3b › /save olladocs-test:latest
creating olladocs-test:latest from llama3.2:3b…
  using existing layer sha256:dde5aa3fc5ff…
  creating new layer sha256:7b935f8b1107…
  writing manifest
  success
created olladocs-test:latest — `ollamarun olladocs-test:latest`
```

(Created model above was removed again with the API's `/api/delete` after the
check.)

## Starting over

| Command | Effect |
| --- | --- |
| `/clear` | Empties the context; keeps settings, keeps the file's identity (`context cleared`), starts a new id so the old file stays. |
| `/new [TITLE]` | Saves the current session and starts a brand-new one (`new session 20261018-122626-02cd`). |
| `/branch` | Forks instead of starting over — see [Branching](branching.md). |

Both are also reachable by just starting another run with `--new` or a
different `--session`.

## Titles and tags

Created automatically from the first prompt, settable with `--title`,
`--tag`, `/title` and `/tag`, or asked from the model with `--auto-title`:
see [Titles and tags](titles-tags.md).
