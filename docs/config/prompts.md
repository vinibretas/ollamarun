# Prompts

The text sent to the model as system prompts and the text used to ask it for
titles and summaries are files, not code:

| File (`ollamarun_lib/defs/prompts/`) | Used for |
| --- | --- |
| `system.md` | The built-in default system prompt. |
| `system_tools.md` | Tool, web-search, math-in-code and tool-result rules, appended when tools are on. |
| `title.md` | `--auto-title` / `/title --auto` — wraps the conversation as `{{content}}`. |
| `summarize.md` | `/summarize` and auto-compaction — wraps the transcript as `{{content}}`. |
| `pdf_vision.md` | The instruction `read_pdf` sends with each page image of a scanned PDF. |

A same-named file under `~/etc/config_files/ollamarun-defs/prompts/`
overrides the shipped one, so prompts are tunable without touching Python.
Missing or unreadable files warn and the prompt reads as none — a bad edit
never breaks the harness.

## The system prompt contract

Every request carries a leading `system` message built from the session:

| Session `system` | What is sent |
| --- | --- |
| `null` (unset) | The built-in default — `system.md`, plus `system_tools.md` when tools are on. |
| `""` (empty) | Nothing; no system prompt at all. |
| text | Exactly that text. |

It is assembled at request time and deliberately **not** stored in
`session.messages`, so `/system` changes land on the next request and
exports keep the history clean.

Ways in and out, all sharing that one contract:

| Where | Effect |
| --- | --- |
| `--system TEXT` / `--system-file FILE` | Set it for a run. |
| `--no-system`, config `system: ""` | Disable it. |
| `/system` | Show what is in effect (panel, labelled with its source). |
| `/system TEXT` | Set it (`system prompt set`). |
| `/system off` | Disable it (`system prompt disabled`). |
| `/system clear` / `/system default` | Back to the built-in default (`system prompt reset to the built-in default`). |
| `/set system TEXT` | Same as `/system TEXT`. |

An override in place is visible in the banner
(`system prompt: <text>`) and takes effect with no code change — verified by
dropping a `system.md` containing *"You are a terse assistant. Answer in at
most three words."* into the override directory: the next run answered
`Endless blue waters.` to "Describe the ocean.".

## Tool wording

The other half of the model-facing text is per-tool: descriptions,
parameter docs, safety flags and aliases in
`ollamarun_lib/defs/tools/*.json`, with the same override directory — see
[Tool definitions](../cli/tools/definitions.md).

## Editing tips

- The defaults are deliberately short; the tool rules in `system_tools.md`
  are what keep models on rails (compute with `run_python`, search again
  instead of giving up, report tool results honestly). If you replace them,
  keep those three.
- `{{content}}` is the only placeholder; both title and summarize prompts
  use it exactly once.
- The files are read once per process and cached, so a running REPL keeps
  its prompts — restart to pick up edits.
