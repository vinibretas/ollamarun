# ollamarun

Full `ollama run` wrapper: every ollama 0.34 run flag plus markdown rendering with a live re-render while streaming and 2D Unicode math — display math (`$$…$$`, `\[…\]`) is laid out by a self-contained engine (stacked fractions, sqrt overlines, sum/prod/lim limits above-below, integral side-limits, matrices, cases, aligned, `\boxed`, `\binom`, tall delimiters, accents, `\mathbb`/`\mathcal`/`\mathfrak`/`\mathbf`) centered in cyan, with flattened Unicode for inline math and pylatexenc as the never-crashing fallback. A `~tok/s` footer streams live (thinking + answer, 5s window; `--no-tps`/`/tps`). Thinking streams live and is collapsible — collapsed by default as an animated `thinking · N chars · Ctrl+T to expand` line; Ctrl+T expands a tail-capped boxed preview in place (press it again to collapse) and `/thinking full` prints the stored thinking panel later. A tool harness gives the model `run_shell`, `run_python`, `read_file`, `write_file`, `list_dir`, `fetch_url`, `web_search` (new, when the `ddgs` package is installed) and tts-remote's `speak`/`list_voices` through Ollama function calling. `fetch_url` returns readable page text instead of raw markup — HTML as text plus the page's links, JSON pretty-printed, PDFs text-extracted, failed fetches as one actionable line rather than the site's error page, a URL already fetched in the session answered from its cache — and `web_search` returns titles/URLs/snippets (cached too, with search-engine URLs rerouted to it), so the model no longer has to guess URLs. Invented tool names (`search`, `curl`, ...) are resolved onto the real tools, the allowlist policy still applies (read-only calls run unattended, allowlisted shell commands and provably-harmless python run unattended, everything else lists numbered choices — run once, always-allow that exact call, or always-allow a suggested generalized rule (`shell: mkdir -p :*`, `path: ~/proj/**`, `python: imports os`, stored for the session) — and you type one, unless `-y` or `/tools policy auto`), and multi-round tool loops keep calls/results in the session (a finished loop is separated from the committed answer by a bright centered `response` divider); every tool's call and result panels are colour-coded — one fixed colour per tool, a random palette colour (remembered for the run) for anything unmapped, and config `tool_colors` overrides; the result box wears the same colour as the call, with ok/failed printed green/red in its title. tts-remote can also read every final answer aloud (`--speak`, config `tts_auto`, `/tts on`) and `/tts` manages voice, language and whether the model may use the speech tools at all. Model housekeeping: `/model` and `/load --model` unload the current model before switching, `/load SESSION` prompts current-vs-last model and unloads when switching, REPL exit unloads (`--keep-loaded`/`--unload` override; one-shots stay warm by default; embedding models are never unloaded). Conversations persist as one atomic JSON per session in `~/etc/data/ollamarun` (`-c/--continue`, `-r/--resume` picker, `--list/--show/--search/--export/--delete`, export as md/json/html, deletes via `trash-put`), and the prompt_toolkit REPL's slash commands are a superset of ollama's (`/set`, `/show`, `/load`, `/save` → `/api/create`, `/clear`, `/bye`) with `/undo`, `/retry`, `/edit`, `/branch`, `/context`, `/summarize`, `/image`, `/copy`, `/export`, `/stats`, `/tools`, `@file` mentions and tab completion. Embedding models route to `/api/embed` honoring `--dimensions`/`--truncate`; `--json`/`--raw`/`--no-save` cover scripting. Raw HTTP, so it works through `llm-jukebox` on `:11434`. `--remote NAME` runs the whole thing on a tailnet machine — `pi`, `kiddo`, `g3`, `optiplex7070` (alias `optiplex`) or `guri`, or any `host[:port]`, resolved through `tailscale status --json` — so models load, generate and unload there while sessions, files, images and tools stay local; a short `/api/version` preflight fails fast with the tailscale node state when the machine is down, and resuming a session reconnects to the machine it last ran on. A model that is not installed on the target machine (local or remote) offers to pull it after printing that machine's free disk space, with live per-layer progress (`--pull` pulls without asking, `/pull [MODEL]` in the REPL, and declining the prompt opens the model picker). `--gpu-layers SPEC` / `/gpu-layers` chooses how many layers run on the GPU — `20`, `50%` of the model's layers, `+4`/`-4` adjustments, `all`, `cpu` or `auto` — stored in the session, re-resolved per model, effective on the next load. `/num-ctx --max-gpu` picks the largest context that still keeps the model fully on the GPU: the KV-cache size per token comes from `/api/show`'s GGUF metadata (exactly, hybrid per-layer attention included), the model size from `/api/tags`, and free VRAM from nvidia-smi — locally or over one ssh probe for `--remote` machines — minus llama.cpp's fit margin and an overhead allowance; a pinned `/gpu-layers` is honoured. `/num-ctx --max-gpu --probe` (CLI `--num-ctx max-gpu [--max-gpu-probe]`) then verifies by loading the model at candidate contexts and reading each real GPU split back from `/api/ps`, bisecting footprint bytes until the largest 100%-GPU context is bracketed, and leaves the model warm at that context.

```console
$ ollamarun MODEL [PROMPT] [--host HOST | --remote NAME] [--pull] [--sort KEY] [--gpu-layers SPEC] [--num-ctx N|max-gpu [--max-gpu-probe]] [--think [STR]] [--render MODE] [--option K=V]
                  [--image PATH] [--system TEXT] [-c [ID] | --session NAME | -r] [--new]
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
