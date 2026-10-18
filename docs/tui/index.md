# Textual TUI (deprecated)

> **Deprecated.** The Textual front end still works and is smoke-tested, but
> it is not the future: it talks to the runtime **in-process**, and it will
> be rewritten as a pure client of the `ollamarun serve` API — the same
> direction as the planned [web UI](../web-ui/index.md). The inline REPL is
> the supported front end meanwhile.

`--ui tui` (config `"ui": "tui"`, or `--ui auto` to use it when the optional
`textual` package is installed) splits the terminal into panes: the
transcript on top, a rule, a status bar (spinner + state, model and remote,
thinking/tools/render modes, context tokens and %, live tok/s, queued count,
key hints) and an input box that stays available while the model works.

## What is different from the inline REPL

- **Type while it works.** Enter queues the message and it is handed to the
  model at the next sub-task boundary — right after a tool result, or in a
  follow-up round when an answer finishes streaming — so the model continues
  with the new information instead of dropping what it was doing. Commands
  that only read state or change display knobs run immediately; the rest
  queue and run once the turn ends, in order.
- **Ctrl+B** sends what is typed right now: it cuts a streaming answer short
  but never kills a tool that is already running (the command finishes and
  the model gets its result with your message in the next round).
- **Esc Esc** stops the turn (the first Esc arms it). The partial answer is
  kept as a cancelled message and anything queued then runs as the next
  turn.
- **Ctrl+T** thinking, **Ctrl+G** `$EDITOR`, **Ctrl+L** clear the
  transcript, **Ctrl+Q** quit (a second press when busy or with queued
  items), Tab completion and Up/Down history shared with the inline REPL.
  Tool approvals ask in a modal with the same numbered choices (`y`/`a`/`n`
  /`A`) as the console prompt — both are built from one list.

## Running it

```console
$ ollamarun --ui tui llama3.2:3b
```

(Check: a smoke run starts the app, renders the banner into the transcript
pane and quits on Ctrl+Q.)

If the front end cannot run, the harness prints the reason and stays inline:

| Condition | Message |
| --- | --- |
| `textual` missing | `--ui tui needs the textual package (1.0+) — using the inline REPL` |
| `textual` too old | `textual X is too old for the TUI (need 1.0+) — using the inline REPL` |
| not a terminal / `OLLAMA_TUI_DISABLE` | the same warning, inline REPL |

`OLLAMARUN_TUI_FULLSCREEN=1` runs the app on the alternate screen (the
transcript disappears on exit instead of staying in the terminal);
`OLLAMA_TUI_DISABLE` keeps even `--ui tui` on the inline REPL. `--ui auto`
picks the TUI when it is usable, the inline REPL otherwise.

## Why it is deprecated

The TUI reaches into the runtime in-process, which means two front ends to
keep in sync and a UI that cannot be driven from outside the process. The
[agent API](../api/agent/turns.md) already exposes everything it renders —
turns, typed events, parked approvals, sessions — so the rewrite is a client
against a stable surface, not new machinery.
