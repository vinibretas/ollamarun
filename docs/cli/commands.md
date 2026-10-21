# Slash commands

Everything interactive is a slash command, dispatched identically in the
inline REPL and the TUI. `/help` prints the live table; `/help shortcuts`
lists the keys. Syntax below is the canonical form — prefixes and
convenience spellings are noted per command.

## Model and session lifecycle

### `/model`, `/models [NAME | --list | --sort KEY]`
Without an argument: the interactive [picker](models/picker.md). With a
name: switch model (unloading the current one). `--list` prints the table
instead, `--sort KEY` orders it (`modified`, `name`, `size`, `parameters`,
`context`).

```text
llama3.2:3b › /model --list --sort size
```

### `/load [MODEL | SESSION]`
`/load SESSION` resumes a saved session; if its model differs from the
current one, it asks *current or last* and unloads on the switch. `/load
MODEL` is a model switch (`/model` with a different verb):

```text
llama3.2:3b › /load llama3.2:3b
already using that model
```

See [Save and load](sessions/save-load.md).

### `/save NAME`
Creates an Ollama model from the session — messages plus the effective
system prompt:

```text
llama3.2:3b › /save olladocs-test:latest
creating olladocs-test:latest from llama3.2:3b…
  … writing manifest
  success
created olladocs-test:latest — `ollamarun olladocs-test:latest`
```

### `/clear`, `/new [TITLE]`
`/clear` empties the context but keeps settings (a fresh session id under
the same file); `/new` saves the current session and starts a brand-new one.

```text
llama3.2:3b › /clear
context cleared
llama3.2:3b › /new fresh start
new session 20261018-122626-02cd
```

### `/sessions [filter]`
The session browser: a numbered table, then `session number, ID, filter or
'q' to quit`. Typing text filters; a number or id switches.
See [Search and export](sessions/search-export.md).

### `/bye`, `/exit`
Explicitly close the session and its background host (Ctrl+D does the same).
Unloads the session model unless `--keep-loaded`/[housekeeping](models/housekeeping.md).
Closing the window or detaching with Ctrl+C does **not** run this.

```text
llama3.2:3b › /bye
unloaded llama3.2:3b
session 20261018-124236-e402 saved · ~/etc/data/ollamarun/sessions
```

### `/title [TEXT | --auto]`
Show, set, or ask the model to (re)title the session — see
[Titles and tags](sessions/titles-tags.md).

### `/tag [list | add TAG… | rm TAG]`
Session tags: `add`/`rm` take several at once.

```text
llama3.2:3b › /tag add demo docs
tags demo, docs
```

### `/branch [TITLE]`
Fork the conversation into a new session (settings, rules and tags included)
and switch to it — see [Branching](sessions/branching.md).

```text
llama3.2:3b › /branch side quest
branched 20261018-122252-de60
```

## Editing the conversation

### `/undo`, `/retry`, `/edit`

```text
llama3.2:3b › /undo
undone
llama3.2:3b › /retry
retrying…
llama3.2:3b › /edit
resending edited message…
```

`/undo` drops the last exchange as a unit; `/retry` keeps the question and
regenerates; `/edit` reopens the last user message in `$OLLAMA_EDITOR` and
resends the edited text. Full semantics in
[Undo, retry and edit](sessions/undo-retry-edit.md) and
[Editor](editor.md).

### `/copy [N]`
Copies the last response (or its Nth code block) to the clipboard. Needs
`pyperclip`; reports `pyperclip is not installed` otherwise.

## The model and the request

### `/set parameter NAME [VALUE]`, `/set system TEXT`
Set a model option (values are coerced: `0.2` → float, `true` → bool, JSON
passes through); no value clears it. `/set system` is the system prompt.

```text
llama3.2:3b › /set parameter temperature 0.2
set temperature = 0.2
```

### `/params`
The session's parameters table — see [Parameters](models/parameters.md).

### `/think [on | off | low | medium | high]`
The **model's** thinking mode (sent as Ollama's `think`). `/thinking`
controls the display instead — see [Thinking panel](rendering/thinking.md).

```text
llama3.2:3b › /think low
think low
```

### `/show [info | license | modelfile | parameters | system | template]`
Model details, `ollama show`-style. `info` prints the model table and the
session parameters together:

```text
llama3.2:3b › /show info
             model · llama3.2:3b
╭────────────────┬───────────────────────────╮
│ field          │ value                     │
├────────────────┼───────────────────────────┤
│ family         │ llama                     │
│ parameter size │ 3.2B                      │
│ quantization   │ Q4_K_M                    │
│ format         │ gguf                      │
│ capabilities   │ completion, tools         │
│ context length │ 131072                    │
│ modified       │ 2026-09-14T01:41:20-03:00 │
╰────────────────┴───────────────────────────╯
             session parameters
…
```

(Transcript replay is the CLI's `--show ID` — see
[Transcript replay](rendering/transcript.md).)

### `/pull [MODEL]`
Pull or update a model on the Ollama machine with live progress — see
[Pulling models](models/pulling.md).

```text
llama3.2:3b › /pull llama3.2:3b
llama3.2:3b is installed — checking for updates
pulling llama3.2:3b · pulling manifest …
```

### `/ps`
Resident models (`ollama ps`) with VRAM, quant and keep-alive countdown.

```text
llama3.2:3b › /ps
                                ollama ps
│ model       │ size    │ VRAM    │ quant  │ until               │
│ llama3.2:3b │ 3.2 GiB │ 2.7 GiB │ Q4_K_M │ 2026-10-18 12:51:04 │
```

## Context and placement

### `/num-ctx [N | --max-gpu [--probe] | --max-cpu N [--probe]]`
The window size, the GPU fit or the DRAM fill — [num-ctx](context/num-ctx.md)
and the [context overview](context/overview.md).

```text
qwen3:0.6b › /num-ctx 8192
num_ctx set to 8192
qwen3:0.6b › /num-ctx 99999999
99999999 exceeds qwen3:0.6b's max context (40960) — capped to 40960
```

With no argument it prints the current value, the model's maximum and the
two fit forms.

### `/gpu-layers [N | N% | +N | -N | all | cpu | auto]`
Pin the GPU/CPU layer split — [GPU layers](context/gpu-layers.md).

```text
qwen3:0.6b › /gpu-layers 50%
gpu layers 50% → 15 of 29 layers
unloaded qwen3:0.6b
the model will reload with the new setting
```

### `/context`
Window usage: message counts, last request's tokens, effective length.

```text
llama3.2:3b › /context
│ messages               │     2 │
│ last prompt tokens     │    94 │
│ context used           │   100 │
│ context length         │  8192 │
```

### `/summarize`
Compact the older half of the conversation into a summary —
[Compaction](context/compaction.md).

## Tools

### `/tools [SPEC]`
The whole harness lives here:

| Form | Effect |
| --- | --- |
| `/tools` (or `list`/`status`) | The state table and the policy line. |
| `/tools on \| off` | Enable/disable the harness for the session. |
| `/tools policy auto\|allowlist\|ask` | Global approval policy. |
| `/tools enable NAME… \| disable NAME…` | Toggle several tools. |
| `/tools NAME on \| off` | Toggle one tool (aliases resolve). |
| `/tools NAME help` | Generated from the definition JSON. |
| `/tools NAME call ARGS` | Run it yourself through the approval gate. |
| `/tools allow` | The safe shell-command allowlist. |
| `/tools rules [add\|rm]` | Stored approval rules. |
| `/tools help` | The reference table. |

See [Tools](tools/index.md) and
[Approvals and policy](tools/approvals.md).

## Display and output

### `/render [markdown | plain | raw]`
Response rendering — [Streaming markdown](rendering/markdown.md).

```text
llama3.2:3b › /render plain
render plain
```

### `/math [on | off]`
LaTeX → Unicode conversion — [2D math](rendering/math.md).

```text
llama3.2:3b › /math off
math off
```

### `/thinking [show | collapse | hide | toggle | full]`
Thinking display; `full` prints the stored reasoning after the fact —
[Thinking panel](rendering/thinking.md).

### `/tps [on | off]`
The live tokens/s + context footer — [Footer](rendering/footer.md).

```text
llama3.2:3b › /tps off
tps footer off
```

### `/pager [on | off]`
Buffer responses into `$PAGER` — [Pager](pager.md).

### `/image PATH… | clear`
Attach images to the **next** message; `clear` drops pending ones —
[Images and vision](images.md).

## Sessions and longer arcs

### `/export [--format md|json|html] [FILE]`
Export the current session, mid-run — same formats as the CLI
[`--export`](sessions/search-export.md):

```text
llama3.2:3b › /export --format md /tmp/session.md
wrote /tmp/session.md
```

### `/stats`
Session and last-generation stats: turns, tokens, duration, last eval rate.

```text
llama3.2:3b › /stats
                   stats
╭───────────────────┬──────────────────────╮
│ field             │                value │
├───────────────────┼──────────────────────┤
│ session           │ 20261018-130942-c98d │
│ model             │          llama3.2:3b │
│ turns             │                    0 │
│ messages          │                    0 │
│ prompt tokens     │                    0 │
│ completion tokens │                    0 │
│ generation time   │                 0 ms │
╰───────────────────┴──────────────────────╯
```

### `/queue [list | clear | rm N]`
What is waiting while a turn runs — the queue is the deprecated
[TUI](../tui/index.md)'s typing-ahead list; the inline REPL reads one line
at a time.

```text
llama3.2:3b › /queue list
queue is empty
```

### `/tts [on | off | voice NAME | lang CODE | tools on|off | voices]`
Auto-speak final answers through `tts-remote`; `tools` toggles the model's
speech tools — see [Tools](tools/reference.md#speech) and the config keys
`tts_*`.

### `/help [topic]`, `/?`
The command table; `/help shortcuts` the key bindings.

## Shortcuts

```text
╭─────────┬────────────────────────────────────────────────────────────────╮
│ Ctrl+D  │ close the session (and its background host)                    │
│ Ctrl+C  │ clear input; twice within 10s while busy detaches the terminal │
│ Ctrl+Q  │ stop the current generation (inline; stays attached)           │
│ Ctrl+T  │ expand or collapse the thinking panel while a response streams │
│ Ctrl+G  │ $OLLAMA_EDITOR for the current prompt                          │
│ Ctrl+J  │ insert newline                                                 │
│ Ctrl+R  │ search prompt history                                          │
│ Up/Down │ history navigation                                             │
│ Tab     │ complete commands, ids, models, @files                         │
│ @path   │ inline a file into the prompt                                  │
╰─────────┴────────────────────────────────────────────────────────────────╯
```
