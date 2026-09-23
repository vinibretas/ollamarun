# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ollamarun` is a single-file Python CLI (`ollamarun`, ~4300 lines) that fully
wraps `ollama run`: every upstream flag plus persistent sessions, a
prompt_toolkit REPL, live markdown/2D-math rendering, and token/s stats. It
talks to the Ollama HTTP API directly (`requests`, not the `ollama` Python
package) so it rides transparently through `llm-jukebox`, the scheduler this
user runs in front of Ollama on `:11434`.

This repo is checked out as a git submodule at `~/bin/ollamarun` inside the
`vinibretas/binscripts` repo (`~/bin`). `~/bin/ollamarun` is added to `PATH`
alongside `~/bin` itself so the `ollamarun` command resolves straight to the
script here. See `~/bin/AGENTS.md` for the surrounding house style this
script must follow (shebang pin, `rich` usage, argparse structure, version
docstring convention) — it is binding, not optional, for edits in this repo.

## Commands

- Run tests: `python3 test_ollamarun.py` (plain script, no framework; loads
  `ollamarun` via `importlib.machinery.SourceFileLoader` since the file has
  no `.py` extension). Add new cases to the relevant list near the top of
  the file (e.g. `MATH_CASES` for math-rendering regressions) rather than
  writing ad hoc scripts.
- No build/lint config exists. Verify manually: `./ollamarun --help`, or a
  real one-shot call against a running Ollama (`./ollamarun MODEL "prompt"`).
- The script is executable and shebang-pinned to `#! /bin/python3.10`
  (system Python). Do not use stdlib features newer than 3.10 without a
  try/except fallback (see `HAS_PYLATEXENC`/`HAS_PYPERCLIP` guards near the
  top for the existing pattern).

## House style specific to this file

- Every edit that changes behavior must bump `VERSION` (module-level
  constant, currently `"1.2"`) and prepend a new bullet list under
  `Summary of changes in this version:` in the module docstring — read the
  existing entries at the top of `ollamarun` before adding one; that block
  is the changelog, there is no separate CHANGELOG file.
- Config lives at `~/etc/config_files/ollamarun.json`, created from
  `DEFAULT_CONFIG` on first run. Session state is one atomic JSON file per
  conversation plus `index.json` under `~/etc/data/ollamarun/sessions/`.
  Debug logs go to `~/etc/logs/ollamarun.log` (`--debug`). Deletes always go
  through `trash-put`, never a raw unlink.
- Inspection-style flags (`--list`, `--show`, `--search`, `--export`) share
  one output convention: rich table by default, `--json`/`--csv`/`--markdown`
  plus `--out FILE` to redirect — implemented once in `emit()` (~line 2531)
  and reused by each `cli_*` function; new inspection commands should route
  through it rather than hand-rolling output.

## Architecture

The file is organized top-to-bottom as pipeline stages, marked by `# ════`
banner comments — read the banner nearest your line number to orient:

1. **Small helpers** (~L220) — ISO timestamps, id generation, bool/host/think
   coercion, config load (`load_config`), logging setup.
2. **Math: LaTeX → Unicode** (~L405) — regex-level conversion for inline math
   and simple cases, with code fences/spans protected from the passes.
3. **2D math layout engine** (~L540–1605) — a self-contained parser/box-layout
   system for *display* math (`$$...$$`, `\[...\]`): `_tokenize_math` →
   `_MathParser` → an AST of `M*` node classes (`MFrac`, `MSqrt`, `MScript`,
   `MBigOp`, `MMatrix`, `MCases`, `MAligned`, `MBoxed`, etc.) → `_layout()`
   builds a `_Box` (list-of-strings + width) recursively → `_flatten()` is
   the plain-text fallback path. `render_display_math()`/`render_inline_math()`
   are the public entry points; both must never raise — anything unsupported
   (unknown environment, over-wide, over-budget) falls through to
   `pylatexenc` via `convert_math()`. Changes here need care: the layout is
   hand-rolled string-grid arithmetic, not a real typesetting library.
4. **`@file` mentions** (~L1608) — `expand_file_mentions()` inlines file
   contents into a prompt.
5. **Images** (~L1646) — `encode_image`/`image_payload`, base64 embedding
   with `max_file_kb` truncation.
6. **Streaming renderer** (~L1678) — `Renderer` class drives a `rich.Live`
   view: incremental markdown re-render while tokens stream, the collapsible
   thinking panel (`render_thinking`, Ctrl+T pager handoff), and
   `TpsEstimator` for the live `~tok/s` footer. This is the piece that talks
   to the terminal frame-by-frame; changes to streaming UX go here.
7. **Ollama HTTP client** (~L2078) — `OllamaClient` wraps `/api/chat`,
   `/api/generate`, `/api/embed`, `/api/ps`, `/api/create` etc. over raw
   `requests`, streaming response bodies line-by-line. `build_chat_payload()`
   assembles the request body from CLI args + session history.
8. **Session storage** (~L2237) — `Session` (in-memory conversation model:
   messages, model, system prompt, options) and `SessionStore` (atomic
   read/write of the JSON files + `index.json`, list/search/export/delete).
   `export_markdown`/`export_json`/`export_html` are the three export
   formats.
9. **Output convention** (~L2529) — `emit()` plus the `cli_list`/`cli_search`/
   `cli_show`/`cli_export`/`cli_delete` functions that implement the
   non-interactive inspection flags.
10. **Interactive helpers** (~L2768) — `open_in_editor` (`$OLLAMA_EDITOR`),
    `pick_model`/`pick_session` (fuzzy pickers), `model_context_length`
    (reads context window from `/api/show` + GGUF header heuristics),
    `SlashCompleter` (prompt_toolkit completion for `/commands`, session ids,
    models, `@files`).
11. **REPL** (~L2844 `Repl` class, then ~L3128–3820 `cmd_*` functions) — each
    slash command (`/set`, `/show`, `/load`, `/save`, `/undo`, `/retry`,
    `/edit`, `/branch`, `/context`, `/summarize`, `/image`, `/render`,
    `/math`, `/pager`, `/tps`, `/copy`, `/export`, `/stats`, `/thinking`,
    ...) is a standalone `cmd_NAME(repl, argument)` function dispatched by
    name; add a new slash command by adding one function here and wiring it
    into the dispatch table in `Repl`.
12. **One-shot / embedding runs** (~L3822) — `run_once()` (non-REPL path) and
    `run_embed()` (routes embedding-capable models to `/api/embed` instead of
    chat).
13. **CLI** (~L3915) — `build_parser()` (the full argparse surface, mirroring
    every `ollama run` flag), `resolve_model`/`resolve_session` (figuring out
    what model/session a given invocation means, including the `-c`/`-r`/
    `--session` disambiguation in `normalize_argv`), and `main()` as the
    entry point.

Model housekeeping (unload-before-switch, unload-on-exit, embedding models
always resident) is cross-cutting logic that lives in the REPL commands
(`cmd_model`, `cmd_load`) and `main()`'s exit path, coordinated through
`/api/ps` lookups in `OllamaClient` — not a separate module.

## Testing notes

`test_ollamarun.py` is table-driven: most sections are a `CASES`-style list
of `(name, input, predicate)` tuples run through a small harness at the
bottom of the file, mirroring the pattern documented in `~/bin/AGENTS.md`
(`test_modelbenchmark.py` is the canonical example there). The math-rendering
tests (`MATH_CASES`) are the largest block and the most likely place a change
to the layout engine (stage 3 above) will need new coverage.
