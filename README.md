# ollamarun

Full `ollama run` wrapper: every ollama 0.34 run flag plus markdown rendering with a live re-render while streaming and 2D Unicode math — display math (`$$…$$`, `\[…\]`) is laid out by a self-contained engine (stacked fractions, sqrt overlines, sum/prod/lim limits above-below, integral side-limits, matrices, cases, aligned, `\boxed`, `\binom`, tall delimiters, accents, `\mathbb`/`\mathcal`/`\mathfrak`/`\mathbf`) centered in cyan, with flattened Unicode for inline math and pylatexenc as the never-crashing fallback. A `~tok/s` footer streams live (thinking + answer) with the 5s windowed rate, its cumulative average, the streamed token count and a live context counter — `~24.0 tok/s (avg 21.4) · 45 tok · ctx 12.345k/32.768k (38.123%)`, the ctx figure ticking with every token, its denominator the window the model is actually loaded with (`/api/ps`; a session `num_ctx` wins, the architectural maximum is only the fallback), and its counts reading raw below 1k / 3-decimal k above — and the TUI status bar shows the same rate, average and live ctx (`--no-tps`/`/tps` switch it off). Response timings (`⏱ load … · prompt … · eval … · total …`) print under every answer on a terminal — `--verbose` is obsolete there and now only adds them to piped/redirected output; `--raw`/`--json` never get the line, and config `show_stats: false` turns it off. Thinking streams live and is collapsible — collapsed by default as an animated `thinking · N chars · Ctrl+T to expand` line; Ctrl+T expands a tail-capped boxed preview in place (press it again to collapse) and `/thinking full` prints the stored thinking panel later. A tool harness gives the model `run_shell`, `run_python`, `read_file`, `write_file`, `edit_file` (exact-fragment replacement with an atomic write; the call panel shows a coloured diff before the change is approved), `list_dir`, `make_dir`/`move_file`/`copy_file`/`delete_file` (trash-put only, never a permanent unlink), `fetch_url`, `web_search` (when the `ddgs` package is installed), `search_sessions` (searches the saved conversations), `list_models` (installed models with the loaded ones marked by VRAM and unload time), `system_status` (RAM, GPU VRAM and model-disk space of the machine running Ollama — ssh-probed under `--remote`, `target="local"` for this machine), `context_status`/`session_info` (the model's own context budget — window size and usage — plus the date, model capabilities and tool round budget) and, when the `chemInventoryAPI` CLI is installed, the read-only `chem_search`/`chem_container`/`chem_locations` through Ollama function calling, plus tts-remote's `speak`/`list_voices`. `fetch_url` returns readable page text instead of raw markup — HTML as text plus the page's links, JSON pretty-printed, PDFs text-extracted, failed fetches as one actionable line rather than the site's error page, a URL already fetched in the session answered from its cache — and `web_search` returns titles/URLs/snippets (cached too, with search-engine URLs rerouted to it), so the model no longer has to guess URLs. Invented tool names (`search`, `curl`, ...) are resolved onto the real tools, the allowlist policy still applies (read-only calls run unattended, allowlisted shell commands and provably-harmless python run unattended, everything else lists numbered choices — run once, always-allow that exact call, or always-allow a suggested generalized rule (`shell: mkdir -p :*`, `path: ~/proj/**`, `python: imports os`, stored for the session) — and you type one, unless `-y` or `/tools policy auto`), and multi-round tool loops keep calls/results in the session (a finished loop is separated from the committed answer by a bright centered `response` divider); every tool's call and result panels are colour-coded — one fixed colour per tool, a random palette colour (remembered for the run) for anything unmapped, and config `tool_colors` overrides; the result box wears the same colour as the call, with ok/failed printed green/red in its title, and the speak tool's box also names where the voice is generated and where it will be heard (`optiplex → g3`, with `(cached)` for a phrase served from disk). tts-remote can also read every final answer aloud (`--speak`, config `tts_auto`, `/tts on` — the `speaking` line ends with the same `X → Y`) and `/tts` manages voice, language and whether the model may use the speech tools at all. Model housekeeping: `/model` and `/load --model` unload the current model before switching, `/load SESSION` prompts current-vs-last model and unloads when switching, REPL exit unloads (`--keep-loaded`/`--unload` override; one-shots stay warm by default; embedding models are never unloaded). Conversations persist as one atomic JSON per session in `~/etc/data/ollamarun` (`-c/--continue`, `-r/--resume` picker, `--list/--show/--search/--export/--delete`, export as md/json/html, deletes via `trash-put`), and the prompt_toolkit REPL's slash commands are a superset of ollama's (`/set`, `/show`, `/load`, `/save` → `/api/create`, `/clear`, `/bye`) with `/undo`, `/retry`, `/edit`, `/branch`, `/context`, `/summarize`, `/image`, `/copy`, `/export`, `/stats`, `/tools`, `@file` mentions and tab completion. Embedding models route to `/api/embed` honoring `--dimensions`/`--truncate`; `--json`/`--raw`/`--no-save` cover scripting. Raw HTTP, so it works through `llm-jukebox` on `:11434`. `--remote NAME` runs the whole thing on a tailnet machine — `pi`, `kiddo`, `g3`, `optiplex7070` (alias `optiplex`) or `guri`, or any `host[:port]`, resolved through `tailscale status --json` — so models load, generate and unload there while sessions, files, images and tools stay local; a short `/api/version` preflight fails fast with the tailscale node state when the machine is down, and resuming a session reconnects to the machine it last ran on. A model that is not installed on the target machine (local or remote) offers to pull it after printing that machine's free disk space, with live per-layer progress (`--pull` pulls without asking, `/pull [MODEL]` in the REPL, and declining the prompt opens the model picker). `--gpu-layers SPEC` / `/gpu-layers` chooses how many layers run on the GPU — `20`, `50%` of the model's layers, `+4`/`-4` adjustments, `all`, `cpu` or `auto` — stored in the session, re-resolved per model, effective on the next load. `/num-ctx --max-gpu` picks the largest context that still keeps the model fully on the GPU: the KV-cache size per token comes from `/api/show`'s GGUF metadata (exactly, hybrid per-layer attention included), the model size from `/api/tags`, and free VRAM from nvidia-smi — locally or over one ssh probe for `--remote` machines — minus llama.cpp's fit margin and an overhead allowance; resident models' VRAM is credited back (it is released when they reload at the new context, `max_gpu_credit_residents: false` limits the credit to the model being sized) and a margin-only infeasible placement reports the context it would fit without the reserve; a pinned `/gpu-layers` is honoured. `/num-ctx --max-gpu --probe` (CLI `--num-ctx max-gpu --probe`) then verifies by loading the model from that estimate and reading each real GPU split back from `/api/ps`, solving the measured bytes-per-token line for the capacity instead of stepping bisection midpoints, and leaves the model warm at the largest 100%-GPU context. `/num-ctx --max-cpu N` (CLI `--num-ctx --max-cpu N [--probe]`) fills DRAM instead: it raises the context until the model's host RAM reaches N% (1-100) of the DRAM available to it — `/proc/meminfo`'s MemAvailable on the ollama machine (one ssh probe for `--remote` hosts) plus the RAM the loaded model already holds (`/api/ps` `size - size_vram`, back when it reloads; other programs and models keep theirs), less a `max_cpu_overhead_mb` allowance — some layers staying on the GPU unless `/gpu-layers` is cpu (a pinned layer count is honoured and only the host side counts); `--probe` loads candidates, reads the real footprint back from `/api/ps` and solves the measured footprint line for the target (`max_cpu_probes`, default 8), never choosing a context above it, and leaves the model warm there.

```console
$ ollamarun MODEL [PROMPT] [--host HOST | --remote NAME] [--pull] [--sort KEY] [--gpu-layers SPEC] [--num-ctx [N|max-gpu | --max-cpu N] [--probe]] [--think [STR]] [--render MODE] [--option K=V]
                  [--image PATH] [--system TEXT | --no-system] [-c [ID] | --session NAME | -r] [--new]
                  [--tools | --no-tools] [-y] [--list | --show ID | --search TEXT | --export ID]
                  [--export-format FMT] [--delete ID [-y]] [--out FILE] [--no-save]
                  [--json | --raw] [--verbose] [--speak | --no-speak]
                  [--show-thinking | --hidethinking] [--no-tps] [--unload | --keep-loaded]
                  # Ctrl+T: expand/collapse the thinking panel while streaming
                  # /model (or /models): switch, list (--list) or sort the
                  #   installed models -- --sort name|size|parameters|context|modified
                  #   also works at the picker prompt, and --sort KEY before a
                  #   model-less run orders the picker it opens
                  # /tools: harness status, policy, tool toggles and approval rules
                  # /tts: auto-speak, voice, language, model access to tts tools
```

## TUI (`--ui tui`)

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

The inline prompt_toolkit REPL remains the default (`--ui inline`) and is the
fallback whenever Textual is missing or the output is not a terminal.

## Prompts and tool definitions

The model-facing text lives in files, not code:

- `ollamarun_lib/defs/tools/*.json` — one JSON per tool: `name`, the
  `description` and JSON-schema `parameters` the model sees, `read_only` /
  `always_safe` safety flags, `requires` (`search` for `web_search`'s ddgs
  backend, `tts` for the speech tools), the model-invented `aliases` that
  resolve onto it, and the `handler` that runs it.
- `ollamarun_lib/defs/prompts/` — `system.md` (the built-in default system
  prompt), `system_tools.md` (the tool, web-search and math-in-code rules,
  appended when tools are on), `title.md` and `summarize.md` (both take the
  text to wrap as `{{content}}`).

A same-named file under `~/etc/config_files/ollamarun-defs/` overrides the
shipped one, so prompts and tool wording can be tuned without touching
Python. Bad definition files warn and are skipped instead of breaking the
harness.

Every request carries a system prompt: the session's own (`--system`,
`--system-file`, config `system`, `/system TEXT`) when set, the built-in
default otherwise. `--no-system` / config `system: ""` / `/system off` send
none, `/system clear` returns to the default, and `/system` alone shows what
is in effect. It is sent as a leading system message and never enters the
saved history.

## Layout

- `ollamarun` — thin executable entry point (`#! /bin/python3.10`); calls
  into `ollamarun_lib/`.
- `ollamarun_lib/` — the implementation, one module per pipeline stage
  (math rendering, HTTP client, sessions, REPL, CLI, ...) — see CLAUDE.md's
  Architecture section for what lives where.
- `test_ollamarun.py` — test suite. Run with `python3 test_ollamarun.py`.

## Usage as a submodule

This repo is checked out as a git submodule at `ollamarun/` inside
[`vinibretas/binscripts`](https://github.com/vinibretas/binscripts) (`~/bin`
on the machine it runs on). `~/bin/ollamarun` (this submodule's directory) is
added to `PATH` alongside `~/bin` itself so the `ollamarun` command resolves
straight to the script here.

See `~/bin/AGENTS.md` in binscripts for the surrounding tool conventions
(shebang pin, `rich` usage, argparse structure, config/log/data locations).
