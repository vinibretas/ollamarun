# CLI overview

`ollamarun` is the terminal harness. Every `ollama run` flag (ollama 0.34) is
accepted and extended, so it still doubles as a drop-in replacement for the
CLI it started as: run it one-shot with a prompt, or with none for the
interactive REPL.

- **Agent loop, not a pipe.** The model calls tools through Ollama function
  calling — shell, Python, files, web, your saved sessions, system stats,
  speech, the chemical inventory — in multi-round loops. Read-only calls run
  unattended; everything else asks, with numbered choices and remembered
  rules. See [Tools](tools/index.md).
- **Rendering that keeps up.** Markdown re-renders while tokens stream,
  display math is laid out in 2D Unicode by a self-contained engine, thinking
  streams live in a collapsible panel, and a `~tok/s · ctx` footer ticks with
  every token (`--no-tps`/`/tps` switch it off). See
  [Streaming markdown](rendering/markdown.md).
- **Sessions that persist.** One atomic JSON per conversation plus an index
  under `~/etc/data/ollamarun/`, with search, export (md/json/html), branch,
  tags and `trash-put` deletes. See [Storage](sessions/storage.md).
- **Two front ends.** The prompt_toolkit inline REPL (default) or `--ui tui`
  (deprecated — see [TUI](../tui/index.md)): transcript pane, status bar, and
  an input box that stays usable while the model works.
- **Any machine on the tailnet.** `--remote NAME` runs all inference and model
  management on another machine — a name defined in config `remote_hosts`, or
  any `host[:port]` — while sessions, files and tools stay local. See
  [Remote inference](remote.md).
- **Context and placement fitting.** `/num-ctx --max-gpu` finds the largest
  window that stays fully on the GPU, `/num-ctx --max-cpu N` fills DRAM to N%,
  `--probe` verifies by loading and calibrates later estimates, and
  `/gpu-layers` pins layer counts. See
  [Context and placement](context/overview.md).
- **Model-facing text is data.** Tool definitions and prompts live as
  JSON/Markdown under `ollamarun_lib/defs/`, overridable per user without
  touching Python. See [Tool definitions](tools/definitions.md) and
  [Prompts](../config/prompts.md).

Start with [Usage and flags](usage.md), or jump straight to
[The REPL](repl.md).
