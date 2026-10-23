# The REPL

With no `PROMPT` argument, `ollamarun` starts the interactive front end: the
inline prompt_toolkit REPL (the default, `--ui inline`). A model-less run
opens the [picker](../cli/models/picker.md) first.

## Starting up

```console
$ ollamarun llama3.2:3b
```

The banner names what is in effect:

```text
╭─ ollamarun ──────────────────────────────────────────────────────────╮
│ model   llama3.2:3b · 3.2B · Q4_K_M · tools · thinking               │
│ session 20261018-105614-b08e · new session                           │
│ host    http://127.0.0.1:11434                                       │
│ window  4096 tok (OLLAMA_CONTEXT_LENGTH=4096) · model max 131072     │
│ usage   12 messages · 3.2k used (78%)                                │
│ render  markdown · thinking collapsed · 95 local models              │
│ tools   allowlist · 8 rounds                                         │
│ gpu     auto (ollama decides)                                        │
│ stats   6 turns · 41.2k prompt tok · 3.4k completion tok             │
│ background pid 1234 · survives detach                                │
│ /help for commands · Ctrl+D closes the session · Ctrl+C twice        │
│ detaches · Ctrl+T thinking                                           │
╰──────────────────────────────────────────────────────────────────────╯
llama3.2:3b › 
```

The `window` line names the window the model will *actually run with* and
where that number came from — the session's `num_ctx` (`num_ctx=8192`), the
loaded runner (`loaded runner`, from `/api/ps`), the model's own Modelfile
default (`model num_ctx=8192`) or the ollama service environment
(`OLLAMA_CONTEXT_LENGTH=4096`, read from the unit) — the same value
[`/context`](commands.md#context) and the footer use. The model's trained
maximum is never presented as the window; it is a separate `model max` fact
next to it, and a cold model whose load context cannot be measured reads
`not loaded yet`. The line below carries the message count and the
colour-coded share in use (yellow at 70%, red at 90%). In a resumed session
the stored messages are replayed below the banner and the bar carries the
conversation's turn and token totals, so the session picks up where it left
off.

The model starts loading in the background the moment the session opens —
the prompt is available immediately, with a `waiting for <model>…` spinner on
its right while the load runs. When the model becomes resident, a one-time
[`ollama ps`](../cli/models/housekeeping.md) table prints what is loaded (its
`context` column carries the window the runner actually uses) and for how
long.

## Typing prompts

- Plain text goes to the model and streams back with the live
  [footer](rendering/footer.md) and, for thinking models, the collapsed
  [thinking panel](rendering/thinking.md).
- `@path` inlines a file's contents — see
  [@file mentions](file-mentions.md).
- `/command` runs a slash command — the full set is in
  [Slash commands](commands.md).
- **Tab** completes commands, session ids, model names and `@files`.
- **Up/Down** walks the prompt history (kept in
  `~/etc/data/ollamarun/history`; set `OLLAMA_NOHISTORY` to disable it).
- **Ctrl+J** inserts a newline for multi-line prompts.
- **Ctrl+R** searches the prompt history.
- **Ctrl+G** opens `$OLLAMA_EDITOR` on the current input — see
  [Editor](editor.md).
- **Ctrl+D** closes the session (equivalent to `/bye`).

## While an answer streams

The input line becomes a rule above the live view; keystrokes are not read
until the turn ends. Two keys work mid-stream:

- **Ctrl+T** expands the thinking panel in place; press it again to collapse.
  The stream never pauses either way, and the choice sticks for the rest of
  the running REPL (later rounds, turns and a reattached host included).
- **Ctrl+C** arms a detach: the first press prints a warning, and a second
  press within 10 s closes the terminal while the turn keeps running in the
  background. The host is not cancelled — use `/bye`, `Ctrl+D` or
  `--close-session` to stop it. At an idle prompt `Ctrl+C` still clears the
  input line.
- **Ctrl+Q** stops the current generation and stays attached: the whole turn
  is cancelled (a running tool or parked approval included), the turn is
  recorded as cancelled, and the prompt returns. At an idle prompt it does
  nothing. In the TUI, Ctrl+Q remains *quit*; use Esc Esc to stop a turn.

The detach warning names the explicit alternatives:

```text
still running — press Ctrl+C again within 10s to detach; the session keeps
going in the background
Ctrl+D terminates the session · Ctrl+Q stops the current generation
```

The Ctrl+Q part is only shown while a response is actually streaming; a
command such as `/pull` has no generation to stop. Ctrl+D closes the session
from the prompt, and in the background client it also closes immediately
while a response streams.
```

See [Background sessions](sessions/background.md) for reattaching, replay and
take-over.

(The queueing behavior — typing ahead while the model works and having the
message land at the next sub-task boundary — belongs to the deprecated
[Textual TUI](../tui/index.md); the inline REPL reads one line at a time.)

## Help inside the session

`/help` prints the command table; `/help shortcuts` the key bindings:

```text
╭─────────┬────────────────────────────────────────────────────────────────╮
│ key     │ action                                                         │
├─────────┼────────────────────────────────────────────────────────────────┤
│ Ctrl+D  │ close the session (and its background host)                    │
│ Ctrl+C  │ clear input; twice within 10s while busy detaches the terminal │
│ Ctrl+Q  │ stop the current generation (inline; stays attached)           │
│ Ctrl+T  │ expand or collapse the thinking panel (sticks for the session) │
│ Ctrl+G  │ $OLLAMA_EDITOR for the current prompt                          │
│ Ctrl+J  │ insert newline                                                 │
│ Ctrl+R  │ search prompt history                                          │
│ Up/Down │ history navigation                                             │
│ Tab     │ complete commands, ids, models, @files                         │
│ @path   │ inline a file into the prompt                                  │
╰─────────┴────────────────────────────────────────────────────────────────╯
```

## Leaving

`/bye` (or `/exit`, or Ctrl+D) explicitly closes the session and its
background host. The session is saved on the way out, and — unless
`--keep-loaded` / config `unload_on_exit: false` — the model is unloaded:

```text
llama3.2:3b › /bye
session 20261018-105614-b08e saved · ~/etc/data/ollamarun/sessions
```

Sessions are [persistent](sessions/storage.md): `-c` continues the most
recent one and `--session NAME` continues a named one.

!!! note "Without prompt_toolkit"

    `prompt_toolkit` powers the input line. If the package is missing (or the
    output is not a terminal), the REPL still runs, but with plain
    `input()` prompts: no Tab completion, no history, no Ctrl+G/Ctrl+R.
    Install it as shown in [Installation](../getting-started/installation.md)
    to get the real thing.

## The other front end

`--ui tui` starts the Textual TUI — panes, an always-available input box and
a message queue. It is [deprecated](../tui/index.md) ahead of the web-UI
rewrite; the inline REPL is the supported front end.
