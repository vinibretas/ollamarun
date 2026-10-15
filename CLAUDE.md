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
  constant in `ollamarun_lib/constants.py`, currently `"1.24"`) and prepend a
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
   (`load_config`), logging setup.
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
   `TpsEstimator` for the live `~tok/s` footer. Also has the chat-bubble
   helpers (`user_message_box`, small right-aligned box for a user turn;
   `model_message_panel`, full-width titled box for a model turn) and the
   tool panels -- `tool_call_panel()`/`tool_result_panel()`, both coloured
   by `constants.tool_color()` so every tool keeps one identity across the
   live view and the transcript -- plus
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
   (`shell_safety`, `python_safety`, `approval_decision`), argument
   normalization, output truncation and `execute_tool()`. The web half is
   self-contained: `_HTMLTextExtractor`/`extract_html_text` turn pages into
   readable text plus links, `extract_pdf_text` uses pypdf or pdftotext,
   `format_fetch_response` maps a response onto model-readable text (never the
   site's error page), `run_web_search` wraps ddgs and `TOOL_ALIASES`/
   `resolve_tool_alias` map invented tool names (`search`, `curl`, ...) onto
   the real ones. The agent loop that
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
11. **`interactive.py`** — `open_in_editor` (`$OLLAMA_EDITOR`),
   `pick_model`/`pick_session` (fuzzy pickers), `model_context_length`
   (reads context window from `/api/show`), and the missing-model flow:
   `ensure_model()` (free-space line + y/N pull prompt, `--pull` skip,
   picker on decline), `pull_with_progress()` and `disk_free_line()`. Also
   the GPU-layer spec logic: `model_block_count()`,
   `parse_gpu_layers_spec()`, `resolve_gpu_layers()`, `gpu_layers_label()`.
12. **`repl.py`** — `SlashCompleter` (prompt_toolkit completion for
   `/commands`, session ids, models, `@files`), the `Repl` class (including
   the multi-round tool loop `_generate` → `_execute_tool_calls` and the
   y/a/n approval prompt), then the `cmd_*` functions. Each slash command
   (`/set`, `/show`, `/load`, `/save`, `/pull`, `/undo`, `/retry`, `/edit`,
   `/branch`, `/context`, `/summarize`, `/image`, `/render`, `/math`,
   `/pager`, `/tps`, `/ps`, `/num-ctx`, `/gpu-layers`, `/copy`, `/export`, `/stats`,
   `/thinking`, `/tools`, `/tts`, ...) is a standalone
   `cmd_NAME(repl, argument)` function dispatched by name through
   `COMMAND_HANDLERS`; add a new slash command by adding one function here
   and wiring it into `COMMAND_INFO`/`COMMAND_HANDLERS` (and the
   `build_parser()` epilog in `cli.py` if it's worth advertising there).
   `ps_table()`/`resolve_num_ctx()` also live here since `cmd_ps`/
   `cmd_num_ctx` are their only callers (`resolve_num_ctx` is reused by
   `main()` in `cli.py` for `--num-ctx`).
13. **`run.py`** — `run_once()` (non-REPL path, built on an already-constructed
   `Repl`) and `run_embed()` (routes embedding-capable models to
   `/api/embed` instead of chat).
14. **`cli.py`** — `build_parser()` (the full argparse surface, mirroring
   every `ollama run` flag), `resolve_model`/`resolve_session` (figuring out
   what model/session a given invocation means, including the `-c`/`-r`/
   `--session` disambiguation in `normalize_argv`), and `main()` as the
   entry point (called by the top-level `ollamarun` script).

Model housekeeping (unload-before-switch, unload-on-exit, embedding models
always resident) is cross-cutting logic that lives in the REPL commands
(`cmd_model`, `cmd_load`, both in `repl.py`) and `main()`'s exit path
(`cli.py`), coordinated through `/api/ps` lookups in `OllamaClient` — not a
separate module.

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
