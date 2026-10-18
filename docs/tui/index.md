# Textual TUI (deprecated)

> **Deprecated.** The Textual front end still works, but it is not the future:
> it talks to the runtime in-process, and it will be rewritten as a pure client
> of the `ollamarun serve` API — the same direction as the planned web UI. The
> inline REPL is the supported front end meanwhile.

`--ui tui` (config `"ui": "tui"`, or `--ui auto` to use it when the optional
`textual` package is installed) separates the interactive session into panes:
the transcript up top, a rule, a status bar (spinner + state, model and remote,
thinking/tools/render modes, context tokens and %, live tok/s, queued count,
key hints) and an input box that stays available while the model works.

- **Type while it works.** Enter queues the message and it is handed to the
  model at the next sub-task boundary — right after a tool result, or in a
  follow-up round when an answer finishes streaming — so the model continues
  with the new information instead of dropping what it was doing. Commands
  that only read state or change display knobs run immediately; the rest
  queue and run once the turn ends, in order (`/queue` shows/edits the list).
- **Ctrl+B** sends what is typed right now: it cuts a streaming answer short
  but never kills a tool that is already running (the command keeps running
  and the model gets its result with your message in the next round).
- **Esc Esc** stops the turn (the first Esc arms it). The partial answer is
  kept as a cancelled message and anything queued then runs as the next turn.
- **Ctrl+T** thinking, **Ctrl+G** `$EDITOR`, **Ctrl+L** clear the transcript,
  **Ctrl+Q** quit (a second press when busy or with queued items), Tab
  completion and Up/Down history shared with the inline REPL. Tool approvals
  ask in a modal (`y`/`a`/`n`/`A`).

`OLLAMARUN_TUI_FULLSCREEN=1` runs the app on the alternate screen (the
transcript disappears on exit instead of staying in the terminal), and
`OLLAMA_TUI_DISABLE` keeps `--ui tui` on the inline REPL. The inline REPL is
also the fallback whenever Textual is missing or the output is not a terminal.
