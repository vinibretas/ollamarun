# Architecture

--8<-- "AGENTS.md:architecture"

## Repository layout

- `ollamarun` — thin executable entry point (`#! /bin/python3.10`); adds its
  own directory to `sys.path` and calls into `ollamarun_lib/`.
- `ollamarun_lib/` — the implementation, one module per pipeline stage (math
  rendering, HTTP client, sessions, REPL, TUI, CLI, ...) — the module list
  above is the map.
- `ollamarun_lib/defs/` — the model-facing tool definitions and prompts.
- `test_ollamarun.py` (and the serve half it imports) — the test suite; run
  with `python3 test_ollamarun.py`.
- `docs/` + `mkdocs.yml` — this documentation site.

`CLAUDE.md` and `AGENTS.md` (mirror copies) carry the working guidance for
agents editing this repo.

See [Contributing](contributing.md) for where to change what and the test
suite.
