# AGENTS.md

Guidance for coding agents (OpenCode, Codex, Cursor, Claude Code, ...) working
in this repository. `CLAUDE.md` is a mirror copy for Claude Code — **edit both
together**; they must stay identical.

## What this is

`ollamarun` is a full terminal agent harness for Ollama: a drop-in superset of
`ollama run` that can also give the model tools. Beyond every upstream flag it
adds an approval-gated tool-calling loop over Ollama function calling (22
built-in tools, defined as data under `defs/`), persistent sessions, live
markdown/2D-math rendering with a collapsible thinking panel and a tok/s +
context footer, two front ends (the inline prompt_toolkit REPL and the
optional Textual TUI), remote inference on tailnet machines, and context/GPU
placement fitting. It talks to the Ollama HTTP API directly (`requests`, not
the `ollama` Python package) so it rides transparently through `llm-jukebox`,
the scheduler this user runs in front of Ollama on `:11434`.

The harness layers, roughly top to bottom:

- **Front ends** — `cli.py` starts either the inline REPL or the Textual TUI
  (`tui.py`), both talking to the REPL through the `Ui` seam (`ui.py`).
- **Agent loop** — `repl.py`'s `Repl` drives multi-round tool calls
  (`_generate` → `_execute_tool_calls` → approval → follow-up round), plus the
  message queue, cancellation and every slash command.
- **Tool registry** — `tools.py` builds the tool set from `defs/tools/*.json`,
  classifies each call (`approval_decision`), executes it and truncates what
  goes back to the model; prompts live in `defs/prompts/`.
- **Rendering** — `renderer.py` streams markdown/2D math through `rich.Live`,
  with the collapsible thinking panel and `TpsEstimator`.
- **State** — `session.py` persists conversations; `emit.py` backs the
  non-interactive inspection flags.
- **Inference** — `client.py` speaks the Ollama HTTP API; `interactive.py`
  handles pickers/pull/GPU-layer specs; `ctxfit.py` sizes contexts.

`--remote NAME` retargets every Ollama call at a tailnet machine: the known
names are `pi`, `kiddo`, `g3`, `optiplex7070` (alias `optiplex`) and `guri`,
plus any `host[:port]`. `helpers.py` maps the name through `REMOTE_NODES` /
config `remote_hosts` and resolves it to a tailnet IP via
`tailscale status --json` (falling back to the MagicDNS name); `cli.py` builds
the client from it, probes `/api/version` with a short timeout before starting
and reconnects a resumed session to the host it last ran on. Load/unload,
streaming and all `/api/*` calls then happen on that machine while sessions,
files, images and the tool harness stay local — there is no separate remote
code path.

A model missing from the target machine is handled by `interactive.py`'s
`ensure_model()`: it prints the machine's free disk space (`shutil.disk_usage`
locally, a BatchMode ssh `df` probe — reading `OLLAMA_MODELS` from the ollama
unit — for `--remote` hosts, degrading to "unknown" when ssh fails) and offers
to pull it. `--pull` skips the question for one-shots/scripts, declining opens
the model picker, and `/pull` pulls or updates from the REPL. Pulls stream
`/api/pull` through `OllamaClient.pull_model()` behind a rich per-layer
progress display.

`--gpu-layers SPEC` / `/gpu-layers` set how many model layers run on the GPU
(`N`, `P%`, `+N`/`-N`, `all`, `cpu`, `auto`). The spec lives in the session
(`session.gpu_layers`), resolves per model against `/api/show`'s
`<arch>.block_count` (via `interactive.py`'s `resolve_gpu_layers()`), wins over
`session.parameters["num_gpu"]` in `Repl._payload()`, and a change unloads a
resident model so the next request reloads with the new placement.

The `ollamarun` file at the repo root is a thin, shebang-pinned entry point
(`#! /bin/python3.10`); the implementation lives in the `ollamarun_lib/`
package, one module per pipeline stage — see Architecture below. The entry
point adds its own directory to `sys.path` and calls `ollamarun_lib.cli.main()`,
so `ollamarun_lib` must stay next to it (it does, since both live at the repo
root). `ollamarun_lib/__init__.py` re-exports every submodule's public names
at the top level, so `import ollamarun_lib as orun` behaves like the old
single-file script for anything that imports it (tests, one-off scripts).

This repo is checked out as a git submodule at `~/bin/ollamarun` inside the
`vinibretas/binscripts` repo (`~/bin`). `~/bin/ollamarun` is added to `PATH`
alongside `~/bin` itself so the `ollamarun` command resolves straight to the
script here. See `~/bin/AGENTS.md` for the surrounding house style this
script must follow (shebang pin, `rich` usage, argparse structure, version
docstring convention) — it is binding, not optional, for edits in this repo.

## Commands

- Run tests: `python3 test_ollamarun.py` (plain script, no framework; does
  `sys.path.insert(0, ...)` then `import ollamarun_lib as orun`). Add new
  cases to the relevant list/function near the top of the file (e.g.
  `MATH_CASES` for math-rendering regressions) rather than writing ad hoc
  scripts; there's also a `test_*` function per subsystem (client HTTP,
  session storage, REPL commands, CLI parsing, ...) run from `__main__` at
  the bottom — add a new `test_*` function there for a new subsystem and
  wire it into the `total = (...)` sum.
- No build/lint config exists. Verify manually: `./ollamarun --help`,
  `./ollamarun serve --help`, or a real one-shot call against a running
  Ollama (`./ollamarun MODEL "prompt"`).
- The entry-point script is executable and shebang-pinned to
  `#! /bin/python3.10` (system Python). Do not use stdlib features newer
  than 3.10 without a try/except fallback (see the `HAS_PYLATEXENC`/
  `HAS_PYPERCLIP`/`HAS_PROMPT_TOOLKIT` guards in `ollamarun_lib/constants.py`
  for the existing pattern — the `except` branches bind the names to `None`
  or, when the name is subclassed or called at module level (prompt_toolkit's
  `Completer`, `PromptSession`, …), to stand-in classes, so importing a name
  never raises just because the optional dependency is missing).

## House style specific to this file

- Every edit that changes behavior must bump `VERSION` (module-level
  constant in `ollamarun_lib/constants.py`, currently `"1.47"` — keep this
  line current when you bump it) and prepend a new bullet list under
  `Summary of changes in this version:` in the top-level `ollamarun` entry
  script's module docstring — read the existing entries there before adding
  one; that block is the changelog, there is no separate CHANGELOG file.
  Docs-only commits (README/CLAUDE.md/AGENTS.md) don't need a bump.
- Config lives at `~/etc/config_files/ollamarun.json`, created from
  `DEFAULT_CONFIG` on first run. Session state is one atomic JSON file per
  conversation plus `index.json` under `~/etc/data/ollamarun/sessions/`.
  Debug logs go to `~/etc/logs/ollamarun.log` (`--debug`). Deletes always go
  through `trash-put`, never a raw unlink.
- Model-facing text lives in files under `ollamarun_lib/defs/`, not in code:
  `defs/tools/*.json` (one per tool) and `defs/prompts/*.md` (`system.md`,
  `system_tools.md`, `title.md`, `summarize.md`). Edit those instead of
  hardcoding strings; a same-named file in
  `~/etc/config_files/ollamarun-defs/{tools,prompts}/` overrides the shipped
  one (the loaders warn and skip/ignore bad files rather than crash). The
  system-prompt contract: unset session `system` = built-in default, `""` =
  no system prompt (`--no-system`, `/system off`), text = replaces it
  (`/system clear` returns to the default).
- Docs move with behavior: `README.md` is the feature reference (tool table,
  command table, config), `CLAUDE.md`/`AGENTS.md` are the working guidance —
  update them in the same change, keeping the two guidance files identical.
- Inspection-style flags (`--list`, `--show`, `--search`, `--export`) share
  one output convention: rich table by default, `--json`/`--csv`/`--markdown`
  plus `--out FILE` to redirect — implemented once in `emit()`
  (`ollamarun_lib/emit.py`) and reused by each `cli_*` function; new
  inspection commands should route through it rather than hand-rolling
  output.
- Every module-level `console` used across the package is the same singleton
  `rich.Console` instance, created once in `ollamarun_lib/constants.py` and
  imported (`from .constants import console`) wherever it's printed to —
  don't construct a second `Console()` for real output, only for tests that
  need to redirect/record it.

## Architecture

`ollamarun_lib/` is organized as pipeline stages, one module per stage; read
a module's own docstring for what lives there before editing:

1. **`constants.py`** — shared stdlib/`rich`/optional-dependency imports,
   `VERSION`, config/session/log paths, `THEME`, the shared `console`
   singleton, `DEFAULT_CONFIG`, and the tool-panel colours (`TOOL_COLORS`/
   `TOOL_COLOR_PALETTE`, `tool_color()` with its random per-run fallback,
   `set_tool_colors()` for the config `tool_colors` overrides). Every other
   module sits on top of this one.
2. **`helpers.py`** — small helpers: ISO timestamps, id generation,
   bool/host/think coercion, `--remote` resolution (`resolve_remote`,
   `remote_node`, `tailscale_map`, `tailscale_peer_state`, `is_local_host`),
   disk-space probes (`local_free_disk`, `ssh_free_disk` — the ssh probes
   take a `cancel` token), `run_cancellable()`/`CommandCancelled` (the
   process-group-killing subprocess runner every tool shares), config load
   (`load_config`) and the response-timings policy (`resolve_show_stats`),
   logging setup, and the count formatters shared by both front ends
   (`human_tokens`/`format_context` for the ctx counter, `human_chars`,
   `human_duration`).
3. **`mathrender.py`** — LaTeX → Unicode, combined in one module because the
   two halves call into each other: regex-level conversion for inline math
   (`convert_math`, `latex_to_unicode`) falls through to the 2D layout
   engine's `render_inline_math` for inline math, and the layout engine
   falls back to `convert_math`/`latex_to_unicode` for anything unsupported.
   The layout engine is a self-contained parser/box-layout system for
   *display* math (`$$...$$`, `\[...\]`): `_tokenize_math` → `_MathParser` →
   an AST of `M*` node classes (`MFrac`, `MSqrt`, `MScript`, `MBigOp`,
   `MMatrix`, `MCases`, `MAligned`, `MBoxed`, etc.) → `_layout()` builds a
   `_Box` (list-of-strings + width) recursively → `_flatten()` is the
   plain-text fallback path. `render_display_math()`/`render_inline_math()`
   are the public entry points; both must never raise — anything unsupported
   (unknown environment, over-wide, over-budget) falls through to
   `pylatexenc`. This module also has `render_content()`/`render_thinking()`
   (markdown + centered display math, used by the renderer and REPL).
   Changes here need care: the layout is hand-rolled string-grid arithmetic,
   not a real typesetting library.
4. **`files.py`** — `@file` mentions: `expand_file_mentions()` inlines file
   contents into a prompt.
5. **`images.py`** — `encode_image`/`image_payload`, base64 embedding with
   `max_file_kb` truncation.
6. **`renderer.py`** — `Renderer` class drives a `rich.Live` view:
   incremental markdown re-render while tokens stream, the collapsible
   thinking panel (`render_thinking`, Ctrl+T expand/collapse in place;
   `/thinking full` prints the stored panel directly), and
   `TpsEstimator` (`windowed()`/`average()`/`rate()`) for the live footer:
   windowed rate, cumulative average, streamed tokens and a ctx counter
   that counts the round's chunks on top of the `context_used`/
   `context_total` the REPL passes in, and the per-response `⏱` timings
   line whose visibility comes from `resolve_show_stats()` (on by default
   on a terminal, piped output only with `--verbose`, config
   `show_stats: false` as the off-switch). `StreamSink` is the seam the
   TUI uses: with a sink, `stream()` pushes the same frames to a widget
   instead of a `rich.Live`, so both front ends share one streaming loop
   (and `stream(cancel=…)` honors a `CancelToken`). Also has the chat-bubble
   helpers (`user_message_box`, small right-aligned box for a user turn;
   `model_message_panel`, full-width titled box for a model turn) and the
   tool panels -- `tool_call_panel()`/`tool_result_panel()`, both coloured
   by `constants.tool_color()` so every tool keeps one identity across the
   live view and the transcript (`tool_result_panel` also takes a `note`
   that lands in the title -- the speak tool's `optiplex → g3`, persisted
   on the tool message as `tool_note`) -- plus
   `render_transcript()` (replays a saved session the way it looked live,
   used by `/show`/`--show`). This is the piece that talks to the terminal
   frame-by-frame; changes to streaming UX go here. `consume_chunk()` is
   the shared per-chunk accumulator and `HeadlessRenderer` is the
   terminal-free presenter (it emits `message.delta` events instead of
   printing), used by the runtime's headless turns.
7. **`client.py`** — `OllamaClient` wraps `/api/chat`, `/api/generate`,
   `/api/embed`, `/api/ps`, `/api/pull`, `/api/create` etc. over raw
   `requests`, streaming response bodies line-by-line (including pull
   progress events). `build_chat_payload()` assembles
   the request body from CLI args + session history.
8. **`tools.py`** — the tool harness: `ToolSpec`/`default_registry()` (shell,
   python, files — read/write/list plus `edit_file` and the
   `make_dir`/`move_file`/`copy_file`/`delete_file` chores — `web_search`
   when `ddgs` is installed, `fetch_url`, `search_sessions`, `list_models`,
   `system_status`, `context_status`/`session_info`, tts-remote
   speak/list_voices, and the chemInventoryAPI
   lookup tools when that CLI is present), the allowlist policy
   (`shell_safety`, `python_safety`, `approval_decision`), the approval-rule
   engine (`ToolRule`, `suggest_rules`, `rule_matches`, `dangerous_command`),
   argument normalization, output truncation and `execute_tool()`. The tool
   *declarations* load from the definition files — one JSON per tool under
   `defs/tools/` (`ToolDefinition`, `load_tool_definitions()`; `HANDLERS`
   maps the JSON's `handler` name onto the function, `requires` gates a tool
   on a capability, `tool_aliases()` is built from the `aliases` lists so
   invented names like `search`/`curl` resolve onto the real tools; bad files
   warn and are skipped). `ToolSpec` keeps each definition's `requires` and
   `aliases` (`REQUIREMENT_HINTS` words them for the help pages).
   `ToolContext` carries the runtime seams the REPL
   injects: `client`/`store` (list_models, system_status, search_sessions),
   `ssh_target`/`ssh_user` for remote probes, `chem_inventory` (the CLI
   path override) and `context_report`/`session_report` (the plain dicts
   Repl.context_report()/session_report() assemble for the awareness tools).
   The web half
   is self-contained: `_HTMLTextExtractor`/`extract_html_text` turn pages into
   readable text plus links, `extract_pdf_text` uses pypdf or pdftotext,
   `format_fetch_response` maps a response onto model-readable text (never the
   site's error page), `run_web_search` wraps ddgs. The speech half speaks
   through `run_tts_say()` (an older
   `tts-remote` without `--json` is retried without it) and turns the
   `say --json` plan into the shared `title_note` via `speech_hosts()`.
   The agent loop that
   calls into it lives in `runtime.py` (approve → execute → append a
   `role: tool` message → ask the model again), auto-speak lives in
   `Repl.speak_text`, and rendering of calls/results is `renderer.py`.
   `ToolContext.cancel` reaches every subprocess-backed tool (shell,
   python, the ssh probes, chemInventoryAPI, tts-remote, trash-put)
   through `helpers.run_cancellable()`, and `ToolRunner` abandons network
   tools (`fetch_url`, `web_search`) on cancel so the result is discarded
   instead of waited on.
9. **`prompts.py`** — the prompt files: `load_prompt()` (override layer first,
   shipped default after, cached per process; missing/unreadable warns and
   reads as None), `default_system_prompt()` (composes `system.md` +
   `system_tools.md` when tools are on), `title_prompt()`/`summarize_prompt()`
   (both expand `{{content}}`) and `render_prompt()`. Defaults live in
   `defs/prompts/`.
10. **`events.py`** — the runtime event vocabulary: `RuntimeEvent` plus
   the `turn.*`/`message.*`/`tool.*`/`approval.*` type constants and
   `TERMINAL_EVENTS`. A leaf module (no imports from the package) shared
   by `renderer.py`, `runtime.py` and the HTTP layer; envelopes, sequence
   numbers and replay are added on top of the same types, so nothing
   here depends on transport.
11. **`session.py`** — `Session` (in-memory conversation model: messages,
   model, system prompt, options) and `SessionStore` (atomic read/write of
   the JSON files + `index.json`, list/search/export/delete; its index
   read-modify-write runs under a shared `RLock` plus an exclusive `flock`
   on `<index>.lock` (so a CLI saving while serve runs cannot lose entries)
   and its temp files are named per writer.
   `export_markdown`/`export_json`/`export_html` are the three export
   formats.
12. **`emit.py`** — `emit()` plus the `cli_list`/`cli_search`/`cli_show`/
   `cli_export`/`cli_delete` functions that implement the non-interactive
   inspection flags.
13. **`ui.py`** — the front-end seam: `Ui` (the interactive hooks `repl.py`
   and `interactive.py` call: `ask`, `confirm`, `editor`, `approval`,
   `notify`, and `interactive` — whether the front end can actually ask),
   `ConsoleUi` (the inline implementation — the same rich prompts as
   always, notifications are a no-op; `interactive` follows
   `console.is_terminal`), the tool-approval choice list
   (`approval_entries`/`approval_lines`) and the numbered typed prompt
   `_prompt_approval` (moved here so the console prompt and the TUI modal are
   built from one list), and the status-bar model (`StatusSnapshot`,
   `format_status()`, `spinner_frame()`): plain data plus a pure formatter
   (context text via `format_context`, tps + `tps_avg`), unit-testable
   without a terminal.
14. **`interactive.py`** — `open_in_editor` (`$OLLAMA_EDITOR`),
   `pick_model`/`pick_session` (fuzzy pickers; `ui=` routes the question to
   the TUI front end), `model_context_length`
   (reads context window from `/api/show`), and the missing-model flow:
   `ensure_model()` (free-space line + y/N pull prompt, `--pull` skip,
   picker on decline), `pull_with_progress()` and `disk_free_line()`.
   Model-list sorting lives here too: `MODEL_SORT_KEYS`/`parse_model_sort`
   (`modified`|`name`|`size`|`parameters`|`context`, plus aliases),
   `model_parameter_count` (parses `details.parameter_size`, MoE totals
   included), `sort_models`/`model_table` (shared by the picker and
   `/model --list`) and `fetch_context_lengths`/`cached_context_length`
   (one `/api/show` per uncached model, in parallel behind a status line,
   cached per host); `pick_model` takes `sort=`/`on_sort=` and its prompt
   accepts `--sort KEY`/`-s KEY`/`sort KEY` to re-order the table in place.
   Also
   the GPU-layer spec logic: `model_block_count()`,
   `parse_gpu_layers_spec()`, `resolve_gpu_layers()`, `gpu_layers_label()`.
15. **`ctxfit.py`** — the context fits behind `/num-ctx --max-gpu` and
   `/num-ctx --max-cpu N`: `kv_bytes_per_token()` (llama.cpp's KV formula
   applied to `/api/show`'s GGUF metadata, per-layer `head_count_kv` arrays
   included), `machine_state_for()`/`probe_machine_gpu()` (free VRAM plus
   the ollama unit env) and `machine_ram_for()`/`probe_machine_ram()`
   (available DRAM from /proc/meminfo), locally or over one ssh probe each;
   `estimate_max_gpu_ctx()` (weights from `/api/tags`, llama.cpp's fit
   margin, resident models' `/api/ps` VRAM credited back, pinned
   `/gpu-layers`); `estimate_max_cpu_ctx()` (the budget is MemAvailable plus
   the resident model's own host RAM -- a reload returns it -- and the
   host footprint is the weights plus per-token KV of the layers llama.cpp's
   fit leaves on the CPU, `max_cpu_overhead_mb` reserved); the
   `/api/ps`-verified `probe_max_gpu_ctx()`/`probe_max_cpu_ctx()` that solve
   each next candidate from the measured affine footprint line instead of
   bisecting (`FitRequest`/`GpuFit`/`CpuFit`/`ProbeResult`), the rich
   reports and `max_gpu_context()`/`max_cpu_context()`, which the REPL
   command and the CLI `--num-ctx max-gpu` / `--num-ctx --max-cpu N` flags
   both call. The probes themselves live in `helpers.py` (`local_gpu_probe`,
   `ssh_gpu_probe`, `local_ram_probe`, `ssh_ram_probe`, `host_is_self`).
16. **`runtime.py`** — the agent turn runtime: `AgentRuntime` (the loop
   and tool execution that used to live in `Repl._generate`/
   `Repl._execute_tool_calls`), `TurnSettings`, `TurnHooks` (the host
   seam: payload builder, presenter, approver, event sink, boundary
   drain), `ApprovalRequest`/`ApprovalDecision`, `TurnResult`, and the
   shared `build_payload()`/`context_report()`/`session_report()`
   helpers. It never prints and never touches a terminal; the CLI
   supplies a `rich` presenter and keeps thin wrappers with the old
   signatures, while `HeadlessRenderer` + an event sink drive it with no
   terminal at all. The runtime owns the tool-support capability check
   (once per model, after `turn.started`, warned as a `turn.warning`), and
   `quiet_tools` is read live so `/render raw` mid-turn takes effect.
17. **`turns.py`** — the turn manager: `TurnEvent` envelopes (turn id,
   1-based `seq`, ISO timestamp, deterministic `id`), `TurnManager`
   (a worker thread per turn, one active turn per session, a global
   cap, per-turn buffers, replay with `stream.gap`, TTL retention,
   cancellation, shutdown), `TurnHandle`/`Subscription`/
   `TurnSnapshot`, and the typed errors (`SessionBusy`/`TooManyTurns`/
   `UnknownTurn`/`TurnExpired`) the HTTP layer maps to 409/429/404/410.
   It imports no front end: a host closure runs the runtime and
   returns a `TurnResult`; `wait()` joins a worker for tests and
   shutdown, and a subscriber that falls behind is closed, never
   stalling the turn.
18. **`repl.py`** — `SlashCompleter`/`completion_candidates` (completion for
   `/commands`, session ids, models, `@files`, shared with the TUI input),
   the `Repl` class (the CLI adapter over `runtime.AgentRuntime`:
   `_generate` and `_execute_tool_calls` are thin wrappers that build the
   per-turn `TurnHooks` and render its events, `_approve_tool_call` is the
   policy + prompt approver, and the `MessageQueue`/`CancelToken`
   machinery is unchanged). `headless` + `event_sink` make the same
   object drivable with no terminal. `_payload()` is the one chat-
   request builder: it prepends the effective system prompt as a leading
   `role: system` message — `Repl.effective_system()` returns the session's
   text when set, the built-in default otherwise, None when disabled (empty
   string) — deliberately *not* stored in `session.messages`, so `/system`
   changes land on the next request and exports keep the history clean.
   `Repl.context_length()` is the one effective-window resolver behind the
   footer/status-bar ctx counter, `/context` and the 80%/85% warnings: a
   session `num_ctx` first, then the loaded runner's `/api/ps`
   `context_length`, then `/api/show`'s model default; a show-only answer
   stays provisional until `/api/ps` reports the runner. `model_show()` is
   the run-long `/api/show` cache the length/block-count/awareness readers
   share, and `context_report()`/`session_report()` assemble the plain
   dicts the `context_status`/`session_info` tools format (window + usage,
   identity + settings + tool round).
   `/system` semantics live in `_apply_system` (`clear`/`default` → built-in,
   `off` → disabled, text → custom) plus `_show_system`, shared with
   `/set system`. Then the `cmd_*` functions.
   The numbered tool-approval prompt itself and the entry list it is built
   from (`approval_entries`/`_prompt_approval`, storing generalized
   `ToolRule`s in the session) now live in `ui.py`, re-exported here so
   `_prompt_approval` keeps resolving from this module. Each slash command
   (`/model`/`/models`, `/set`, `/show`, `/load`, `/save`, `/pull`,
   `/num-ctx`, `/gpu-layers`, `/params`, `/tools`, `/tts`, `/think`,
   `/thinking`, `/system`, `/undo`, `/retry`, `/edit`, `/branch`,
   `/context`, `/summarize`, `/image`, `/render`, `/math`, `/pager`,
   `/tps`, `/copy`, `/export`, `/stats`, `/queue`, ...) is a standalone
   `cmd_NAME(repl, argument)` function dispatched by name through
   `COMMAND_HANDLERS`; `COMMAND_INFO` is the authoritative description table
   (the REPL `/help`, the README's command table and the `cli.py` epilog all
   mirror it). Add a new slash command by adding one function here and
   wiring it into `COMMAND_INFO`/`COMMAND_HANDLERS` (and the hand-maintained
   `build_parser()` epilog in `cli.py`).
   `/tools` takes a tool name (or alias) as its first token for per-tool
   control — `NAME on|off|help|call ARGS`: `_find_tool` resolves aliases and
   falls back to `tool_definitions()` so an uninstalled tool still gets a
   page, `_print_tool_help` renders state/description/parameters/approval
   from the definition JSON, and `_cmd_tools_call` runs the tool through the
   normal approval gate without touching the conversation.
   `completion_candidates`/`_tools_completion` Tab-complete the whole
   command (subcommands, names and aliases, per-tool actions, policy modes,
   rule kinds) in both front ends.
   `ps_table()`/`resolve_num_ctx()` also live here since `cmd_ps`/
   `cmd_num_ctx` are their only callers (`resolve_num_ctx` is reused by
   `main()` in `cli.py` for `--num-ctx`). `ApprovalPicker` and its runners are
   re-exported from `ui.py` for compatibility.
19. **`tui.py`** — the optional Textual front end (`--ui tui`): the app
   (transcript pane / rule / status bar / always-on `PromptInput`),
   `TuiStreamSink` (the `StreamSink` that paints the streaming answer),
   `TranscriptWriter` (a file-like sink capturing everything printed through
   the shared `console` into the transcript), the modal screens (confirm,
   ask, and tool approval offering the same numbered choices as the console
   prompt) and `TuiUi` (the `Ui` implementation `repl.py` talks to). Turns and commands run in thread
   workers that marshal widget updates back with `call_from_thread`;
   `run_tui()` is the entry point. Textual is optional — `cli.py`'s
   `wants_tui()` checks it and falls back to the inline REPL with a note.
20. **`run.py`** — `run_once()` (non-REPL path, built on an
   already-constructed `Repl`; its `--json` branch runs the tool loop
   through `AgentRuntime` directly) and `run_embed()` (routes
   embedding-capable models to `/api/embed` instead of chat).
21. **`server.py`** — the generic HTTP plumbing behind `ollamarun serve`:
   `Router` (method + `{param}` path patterns, with distinct 404/405),
   the `Request`/`Response`/`StreamResponse` model, `HttpServer` (a
   `ThreadingHTTPServer` wrapper with a body cap checked before the body
   is read, socket timeouts and daemon handler threads) and `SseWriter`
   (one HTTP chunk per event, flushed; a `: ping` heartbeat from a
   companion thread; a failed write marks the stream broken exactly once
   and fires the response's `on_disconnect`). `ApiError` renders as the
   sanitized `{"error": {...}}` envelope; any other exception is a bare
   500 and its traceback goes only to the `--debug` log.
   `run_forever()` serves on a worker thread with SIGINT/SIGTERM mapped
   to a clean `shutdown()`. It knows nothing about OpenAI or Ollama.
22. **`openai_api.py`** — the OpenAI-compatible `/v1` surface
   (`build_router()` registers `GET /v1/models`, `GET /v1/models/{id}`,
   `POST /v1/chat/completions`). Model turns only: `parse_chat_request()`
   validates the OpenAI schema (explicit 400s for unknown fields, `n>1`,
   `logprobs`, `logit_bias`, `audio`, non-text modalities,
   `tool_choice: "required"`), converts messages verbatim (assistant
   `tool_calls`, `role: tool` + `tool_call_id`→`tool_name` history,
   data-URI images) and `ollama_payload()` builds the one `/api/chat`
   call; client `tools` pass through and never execute (the agent loop,
   tools registry and sessions are not reachable from this module).
   `completion_body()`/`_stream_completion()`/`stream_chunk()` shape the
   OpenAI response and SSE (role chunk, deltas, thinking as
   `delta.reasoning_content`, the tool calls as one indexed array chunk,
   finish reason, optional usage chunk, `[DONE]`); `OpenAIApi` caches the
   model list and capabilities briefly and builds one `OllamaClient` per
   request so a disconnecting client cancels only its own stream. The
   `system` extension (`"auto"` = built-in default, text = replace) is
   off unless asked. `translate_error()` maps upstream failures onto
   404/502 envelopes.
23. **`cli.py`** — `build_parser()` (the full argparse surface, mirroring
   every `ollama run` flag), `resolve_model`/`resolve_session` (figuring out
   what model/session a given invocation means, including the `-c`/`-r`/
   `--session` disambiguation in `normalize_argv`), `ui_mode()`/`wants_tui()`
   (which front end to start), `build_serve_parser()`/`serve_main()` (the
   `ollamarun serve` subcommand: upstream resolution, banner, listener) and
   `main()` as the entry point (called by the top-level `ollamarun` script),
   which dispatches `serve` before the normal parser.

Model housekeeping (unload-before-switch, unload-on-exit, embedding models
always resident) is cross-cutting logic that lives in the REPL commands
(`cmd_model`, `cmd_load`, both in `repl.py`) and `main()`'s exit path
(`cli.py`), coordinated through `/api/ps` lookups in `OllamaClient` — not a
separate module. The model picker those commands (and a model-less
`ollamarun` run) open is ordered by the session's `model_sort`, seeded from
`--sort` and updated whenever a sort is chosen at the prompt.

Cross-module imports follow the numbering above (a module only imports from
one earlier in the list, plus `constants.py`/`helpers.py`/`events.py` from
anywhere) — keep new code on that side of the dependency graph rather than
reaching "backwards" (e.g. don't import from `repl.py` inside
`renderer.py`); use `typing.TYPE_CHECKING` for a backwards-pointing type
hint only, the way `renderer.py` does for `Session` (defined later, in
`session.py`).

### Where to change what

- **New tool**: add `defs/tools/NAME.json` (description, JSON schema, safety
  flags, optional `requires`/`aliases`, `handler` name), the handler function
  and its `HANDLERS` entry in `tools.py`, and a row in the README tool table.
  Registration is data-driven — `default_registry()` picks the file up
  automatically, and `read_only`/`always_safe` feed `approval_decision()`.
- **New slash command**: one `cmd_NAME(repl, argument)` function in
  `repl.py`, wired into `COMMAND_INFO`/`COMMAND_HANDLERS`; complete the
  `build_parser()` epilog and the README command table when it is notable.
- **New config key**: add it to `DEFAULT_CONFIG` in `constants.py` and to the
  README's selected-keys list; user config merges over the defaults.
- **New inspection output** (`--list`-style): build it into a `cli_*`
  function and route through `emit()` (`emit.py`), never a hand-rolled table.
- **New interactive prompt/hook**: extend the `Ui` protocol and `ConsoleUi`
  in `ui.py`, then implement it in `TuiUi` (`tui.py`) — nothing below the
  `repl.py` layer may know which front end is running. Set `interactive`
  on an implementation that can actually ask; the approval gate reads it.
- **New runtime event**: add the type to `events.py`, emit it from
  `runtime.py`, and teach the consumers (`Repl`'s event printer,
  `HeadlessRenderer`, the future HTTP layer) — the event stream is the
  contract between the runtime and every front end.
- **New `/v1` endpoint**: register it in `openai_api.build_router()` and keep
  transport concerns in `server.py` (body caps, SSE framing, error
  envelopes); handlers raise `ApiError` for structured errors and return a
  `Response` or `StreamResponse`. Native agent endpoints belong to
  `turns.py`'s manager plus a headless `Repl`, not to the OpenAI layer.
- **New model-facing text**: tool wording in `defs/tools/*.json`, prompt text
  in `defs/prompts/`; never hardcode it in Python.
- **Streaming/rendering UX**: `renderer.py` (the TUI consumes the same frames
  through `StreamSink`), and `ui.py`'s `StatusSnapshot`/`format_status()` when
  it shows up in the status bar.

## Testing notes

`test_ollamarun.py` is table-driven: some sections are a `CASES`-style list
of `(name, input, predicate)` tuples run through a small harness, others are
a `test_*() -> int` function that prints `PASS`/`FAIL` lines and returns a
failure count, mirroring the pattern documented in `~/bin/AGENTS.md`
(`test_modelbenchmark.py` is the canonical example there); all `test_*`
functions are summed in the `if __name__ == "__main__":` block at the
bottom. The math-rendering tests (`MATH_CASES`) are the largest block and
the most likely place a change to the layout engine (`mathrender.py`) will
need new coverage. `OllamaClient` is tested against a fake `requests.Session`
(`_FakeSession`/`_FakeResponse`) rather than a real HTTP mock library;
REPL/CLI-level tests use `_FakeClient` (a lighter double with the handful of
methods those layers actually call) instead. Console output assertions that
capture the shared `console` singleton (`with orun.console.capture() as cap:`)
should go through the `_plain()` helper before substring-matching, since
`rich`'s automatic number/keyword highlighting can split a literal substring
across ANSI color codes. The Phase 1/1b runtime tests (`test_runtime_events`,
`test_headless_renderer`, `test_session_store_lock`,
`test_session_store_cross_process_lock`, `test_ui_interactive`,
`test_tool_cancellation`, `test_cancellation_coverage`,
`test_file_mention_inlining`, `test_tool_support_warning`,
`test_quiet_tools_live`, `test_approval_event_contract`,
`test_python310_syntax`, `test_turn_result_mapping`) cover the
`AgentRuntime` event stream, the terminal-free turn path, the
in-process/cross-process store locks, turn cancellation (including the
ssh/chem/tts/trash coverage), `@file` inlining, the approval event
contract and the 3.10 syntax guard. The audit guards
(`test_runtime_event_contract`, `test_approval_runtime_contract`,
`test_unattended_turn_denial`, `test_cancel_reaches_production_tools`,
`test_quiet_tools_real_turn`, `test_session_store_process_gate`,
`test_session_store_same_session_writers`,
`test_session_store_without_fcntl`, `test_cancellation_destructive_guard`)
pin the event vocabulary/serialization and terminal-event contract, the
approval order with its interrupt/stopped paths, the cancel token actually
reaching a production turn's tool, the store flock gating another process,
and the destructive-tool refusal after a stop. The Phase 2 tests
(`test_turn_event_envelope`, `test_turn_manager_lifecycle`,
`test_turn_manager_synthesized_terminal`, `test_turn_buffer_truncation`,
`test_turn_expired`, `test_turn_manager_session_busy`,
`test_turn_manager_max_active`, `test_turn_manager_cancel`,
`test_turn_subscriber_overflow`, `test_turn_manager_shutdown`,
`test_turn_manager_context`, `test_turn_manager_headless_turn`) cover
envelopes, lifecycle, replay/gap/truncation, session exclusivity, the
active cap, cancellation, overflow and shutdown. The Phase 3 serve tests
(`test_openai_mapping`, `test_serve_http`, `test_serve_disconnect`,
`test_serve_cli`, `test_openai_sdk`) pin request validation and
message/tool mapping, the live HTTP surface over a fake backend (models,
non-stream and SSE chat, tool calls, the vision gate, the body cap and the
error envelopes), client-disconnect cancellation, the serve parser,
config defaults and `main()` dispatch, and the official OpenAI SDK
round-trip (the SDK test skips when the package is absent; it is a
test-only dependency).
