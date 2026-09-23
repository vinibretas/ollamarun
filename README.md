# ollamarun

Full `ollama run` wrapper: every ollama 0.34 run flag plus markdown rendering with a live re-render while streaming and 2D Unicode math — display math (`$$…$$`, `\[…\]`) is laid out by a self-contained engine (stacked fractions, sqrt overlines, sum/prod/lim limits above-below, integral side-limits, matrices, cases, aligned, `\boxed`, `\binom`, tall delimiters, accents, `\mathbb`/`\mathcal`/`\mathfrak`/`\mathbf`) centered in cyan, with flattened Unicode for inline math and pylatexenc as the never-crashing fallback. A `~tok/s` footer streams live (thinking + answer, 5s window; `--no-tps`/`/tps`). Thinking streams live and is collapsible — collapsed by default as an animated `thinking · N chars · Ctrl+T to expand` line; Ctrl+T opens the full text in the pager (scroll/search; the stream resumes on quit) and expands a tail-capped live preview, while `/thinking full` re-opens it later. Model housekeeping: `/model` and `/load --model` unload the current model before switching, `/load SESSION` prompts current-vs-last model and unloads when switching, REPL exit unloads (`--keep-loaded`/`--unload` override; one-shots stay warm by default; embedding models are never unloaded). Conversations persist as one atomic JSON per session in `~/etc/data/ollamarun` (`-c/--continue`, `-r/--resume` picker, `--list/--show/--search/--export/--delete`, export as md/json/html, deletes via `trash-put`), and the prompt_toolkit REPL's slash commands are a superset of ollama's (`/set`, `/show`, `/load`, `/save` → `/api/create`, `/clear`, `/bye`) with `/undo`, `/retry`, `/edit`, `/branch`, `/context`, `/summarize`, `/image`, `/copy`, `/export`, `/stats`, `@file` mentions and tab completion. Embedding models route to `/api/embed` honoring `--dimensions`/`--truncate`; `--json`/`--raw`/`--no-save` cover scripting. Raw HTTP, so it works through `llm-jukebox` on `:11434`.

```console
$ ollamarun MODEL [PROMPT] [--host HOST] [--think [STR]] [--render MODE] [--option K=V]
                  [--image PATH] [--system TEXT] [-c [ID] | --session NAME | -r] [--new]
                  [--list | --show ID | --search TEXT | --export ID [--export-format FMT]]
                  [--delete ID [-y]] [--out FILE] [--no-save] [--json | --raw] [--verbose]
                  [--show-thinking | --hidethinking] [--no-tps] [--unload | --keep-loaded]
                  # Ctrl+T: thinking pager + expand/collapse while streaming
```

## Layout

- `ollamarun` — the tool itself (`#! /bin/python3.10`).
- `test_ollamarun.py` — test suite. Run with `python3 test_ollamarun.py`.

## Usage as a submodule

This repo is checked out as a git submodule at `ollamarun/` inside
[`vinibretas/binscripts`](https://github.com/vinibretas/binscripts) (`~/bin`
on the machine it runs on). `~/bin/ollamarun` (this submodule's directory) is
added to `PATH` alongside `~/bin` itself so the `ollamarun` command resolves
straight to the script here.

See `~/bin/AGENTS.md` in binscripts for the surrounding tool conventions
(shebang pin, `rich` usage, argparse structure, config/log/data locations).
