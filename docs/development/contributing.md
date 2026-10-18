# Contributing

This is a personal project, but the internals are documented for anyone
reading along. `CLAUDE.md` and `AGENTS.md` (mirror copies) describe the module
layout, the conventions (Python 3.10, `rich`, argparse in `main()`) and where
to change what. There is no contribution process; issues are welcome, but the
roadmap is whatever I am learning next — currently the web UI.

Run the tests with `python3 test_ollamarun.py`. The suite doubles as the
compatibility contract for both API surfaces — it round-trips the official
`openai` SDK and exercises a real `ollamarun serve` process against a stub
backend — and `--live` adds a pass against a real Ollama server.

## Where to change what

--8<-- "AGENTS.md:where-to-change"

## Testing notes

--8<-- "AGENTS.md:testing"
