# Installation

Needs Linux, Python 3.10+ and a reachable Ollama server (default
`127.0.0.1:11434`). The entry point is pinned to `#! /bin/python3.10`; if that
interpreter is elsewhere, run `python3 ollamarun …` instead.

```console
$ pip install --user rich requests prompt_toolkit
```

Optional extras: `pylatexenc` (inline-math fallback), `textual` (the
deprecated `--ui tui` front end), `ddgs` (the `web_search` tool).

The test suite additionally needs `openai` and `pyflakes`:

```console
$ pip install --user openai pyflakes
$ python3 test_ollamarun.py
```
