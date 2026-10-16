# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ollamarun` is a Python CLI that fully wraps `ollama run`: every upstream
flag plus persistent sessions, a prompt_toolkit REPL, live markdown/2D-math
rendering, and token/s stats. It talks to the Ollama HTTP API directly
(`requests`, not the `ollama` Python package) so it rides transparently
through `llm-jukebox`, the scheduler this user runs in front of Ollama on
`:11434`.

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
- No build/lint config exists. Verify manually: `./ollamarun --help`, or a
  real one-shot call against a running Ollama (`./ollamarun MODEL "prompt"`).
- The entry-point script is executable and shebang-pinned to
  `#! /bin/python3.10` (system Python). Do not use stdlib features newer
  than 3.10 without a try/except fallback (see `HAS_PYLATEXENC`/
  `HAS_PYPERCLIP` guards in `ollamarun_lib/constants.py` for the existing
  pattern — note that the `except` branch there also binds the names to
  `None` so `from .constants import LatexNodes2Text` etc. never raises an
  `ImportError` in another module just because the optional dependency is
  missing).

## House style specific to this file

- Every edit that changes behavior must bump `VERSION` (module-level
  constant in `ollamarun_lib/constants.py`, currently `"1.28"`) and prepend a
  new bullet list under `Summary of changes in this version:` in the
  top-level `ollamarun` entry script's module docstring — read the existing
  entries there before adding one; that block is the changelog, there is no
  separate CHANGELOG file.
- Config lives at `~/etc/config_files/ollamarun.json`, created from
  `DEFAULT_CONFIG` on first run. Session state is one atomic JSON file per
  conversation plus `index.json` under `~/etc/data/ollamarun/sessions/`.
  Debug logs go to `~/etc/logs/ollamarun.log` (`--debug`). Deletes always go
  through `trash-put`, never a raw unlink.
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
   disk-space probes (`local_free_disk`, `ssh_free_disk`), config load
   (`load_config`), logging setup, and the count formatters shared by both
   front ends (`human_tokens`/`format_context` for the ctx counter,
   `human_chars`, `human_duration`).
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
   `context_total` the REPL passes in. `StreamSink` is the seam the
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
   frame-by-frame; changes to streaming UX go here.
7. **`client.py`** — `OllamaClient` wraps `/api/chat`, `/api/generate`,
   `/api/embed`, `/api/ps`, `/api/pull`, `/api/create` etc. over raw
   `requests`, streaming response bodies line-by-line (including pull
   progress events). `build_chat_payload()` assembles
   the request body from CLI args + session history.
8. **`tools.py`** — the tool harness: `ToolSpec`/`default_registry()` (shell,
   python, files, `web_search` when `ddgs` is installed, `fetch_url`, tts-remote
   speak/list_voices), the allowlist policy
   (`shell_safety`, `python_safety`, `approval_decision`), the approval-rule
   engine (`ToolRule`, `suggest_rules`, `rule_matches`, `dangerous_command`),
   argument normalization, output truncation and `execute_tool()`. The web half
   is self-contained: `_HTMLTextExtractor`/`extract_html_text` turn pages into
   readable text plus links, `extract_pdf_text` uses pypdf or pdftotext,
   `format_fetch_response` maps a response onto model-readable text (never the
   site's error page), `run_web_search` wraps ddgs and `TOOL_ALIASES`/
   `resolve_tool_alias` map invented tool names (`search`, `curl`, ...) onto
   the real ones. The speech half speaks through `run_tts_say()` (an older
   `tts-remote` without `--json` is retried without it) and turns the
   `say --json` plan into the shared `title_note` via `speech_hosts()`.
   The agent loop that
   calls into it lives in `repl.py` (approve → execute → append a `role: tool`
   message → ask the model again), auto-speak lives in `Repl.speak_text`, and
   rendering of calls/results is `renderer.py`.
9. **`session.py`** — `Session` (in-memory conversation model: messages,
   model, system prompt, options) and `SessionStore` (atomic read/write of
   the JSON files + `index.json`, list/search/export/delete).
   `export_markdown`/`export_json`/`export_html` are the three export
   formats.
10. **`emit.py`** — `emit()` plus the `cli_list`/`cli_search`/`cli_show`/
   `cli_export`/`cli_delete` functions that implement the non-interactive
   inspection flags.
11. **`ui.py`** — the front-end seam: `Ui` (the interactive hooks `repl.py`
   and `interactive.py` call: `ask`, `confirm`, `editor`, `approval`,
   `notify`), `ConsoleUi` (the inline implementation — the same rich prompts
   as always, notifications are a no-op), the tool-approval choice list
   (`approval_entries`/`approval_lines`) and the numbered typed prompt
   `_prompt_approval` (moved here so the console prompt and the TUI modal are
   built from one list), and the status-bar model (`StatusSnapshot`,
   `format_status()`, `spinner_frame()`): plain data plus a pure formatter
   (context text via `format_context`, tps + `tps_avg`), unit-testable
   without a terminal.
12. **`interactive.py`** — `open_in_editor` (`$OLLAMA_EDITOR`),
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
13. **`ctxfit.py`** — the GPU context fit behind `/num-ctx --max-gpu`:
   `kv_bytes_per_token()` (llama.cpp's KV formula applied to `/api/show`'s
   GGUF metadata, per-layer `head_count_kv` arrays included),
   `machine_state_for()`/`probe_machine_gpu()` (free VRAM plus the ollama
   unit env, locally or over one ssh probe), `estimate_max_gpu_ctx()`
   (weights from `/api/tags`, llama.cpp's fit margin, resident models'
   `/api/ps` VRAM credited back, pinned `/gpu-layers`), the `/api/ps`-verified
   `probe_max_gpu_ctx()` that solves each next candidate from the measured
   affine footprint line instead of bisecting (`FitRequest`/`GpuFit`/
   `ProbeResult`), the rich reports and
   `max_gpu_context()`, which the REPL command and the CLI
   `--num-ctx max-gpu` flag both call. The probes themselves live in
   `helpers.py` (`local_gpu_probe`, `ssh_gpu_probe`, `host_is_self`).
14. **`repl.py`** — `SlashCompleter`/`completion_candidates` (completion for
   `/commands`, session ids, models, `@files`, shared with the TUI input),
   the `Repl` class (the multi-round tool loop
   `_generate` → `_execute_tool_calls`, the approval flow
   `_approve_tool_call`, the `MessageQueue` of what was typed while a turn was
   running and `CancelToken`-driven stopping), then the `cmd_*` functions.
   The numbered tool-approval prompt itself and the entry list it is built
   from (`approval_entries`/`_prompt_approval`, storing generalized
   `ToolRule`s in the session) now live in `ui.py`, re-exported here so
   `_prompt_approval` keeps resolving from this module. Each slash command
   (`/model`/`/models` -- switch, `--list`, `--sort KEY`, `/set`, `/show`,
   `/load`, `/save`, `/pull`, `/undo`, `/retry`, `/edit`,
   `/branch`, `/context`, `/summarize`, `/image`, `/render`, `/math`,
   `/pager`, `/tps`, `/ps`, `/num-ctx`, `/gpu-layers`, `/copy`, `/export`, `/stats`,
   `/thinking`, `/tools`, `/tts`, `/queue`, ...) is a standalone
   `cmd_NAME(repl, argument)` function dispatched by name through
   `COMMAND_HANDLERS`; add a new slash command by adding one function here
   and wiring it into `COMMAND_INFO`/`COMMAND_HANDLERS` (and the
   `build_parser()` epilog in `cli.py` if it's worth advertising there).
   `ps_table()`/`resolve_num_ctx()` also live here since `cmd_ps`/
   `cmd_num_ctx` are their only callers (`resolve_num_ctx` is reused by
   `main()` in `cli.py` for `--num-ctx`). `ApprovalPicker` and its runners are
   re-exported from `ui.py` for compatibility.
15. **`tui.py`** — the optional Textual front end (`--ui tui`): the app
   (transcript pane / rule / status bar / always-on `PromptInput`),
   `TuiStreamSink` (the `StreamSink` that paints the streaming answer),
   `TranscriptWriter` (a file-like sink capturing everything printed through
   the shared `console` into the transcript), the modal screens (confirm,
   ask, and tool approval offering the same numbered choices as the console
   prompt) and `TuiUi` (the `Ui` implementation `repl.py` talks to). Turns and commands run in thread
   workers that marshal widget updates back with `call_from_thread`;
   `run_tui()` is the entry point. Textual is optional — `cli.py`'s
   `wants_tui()` checks it and falls back to the inline REPL with a note.
16. **`run.py`** — `run_once()` (non-REPL path, built on an already-constructed
   `Repl`) and `run_embed()` (routes embedding-capable models to
   `/api/embed` instead of chat).
17. **`cli.py`** — `build_parser()` (the full argparse surface, mirroring
   every `ollama run` flag), `resolve_model`/`resolve_session` (figuring out
   what model/session a given invocation means, including the `-c`/`-r`/
   `--session` disambiguation in `normalize_argv`), `ui_mode()`/`wants_tui()`
   (which front end to start) and `main()` as the entry point (called by the
   top-level `ollamarun` script).

Model housekeeping (unload-before-switch, unload-on-exit, embedding models
always resident) is cross-cutting logic that lives in the REPL commands
(`cmd_model`, `cmd_load`, both in `repl.py`) and `main()`'s exit path
(`cli.py`), coordinated through `/api/ps` lookups in `OllamaClient` — not a
separate module. The model picker those commands (and a model-less
`ollamarun` run) open is ordered by the session's `model_sort`, seeded from
`--sort` and updated whenever a sort is chosen at the prompt.

Cross-module imports follow the numbering above (a module only imports from
one earlier in the list, plus `constants.py`/`helpers.py` from anywhere) —
keep new code on that side of the dependency graph rather than reaching
"backwards" (e.g. don't import from `repl.py` inside `renderer.py`); use
`typing.TYPE_CHECKING` for a backwards-pointing type hint only, the way
`renderer.py` does for `Session` (defined later, in `session.py`).

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
across ANSI color codes.
