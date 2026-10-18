# Installation

ollamarun is a single executable plus one Python package — there is no build
step and nothing to compile. It runs anywhere Linux does, as long as a Python
3.10+ and a reachable Ollama server are around.

## Requirements

| What | Why |
| --- | --- |
| Linux | The harness leans on Linux behavior: `/proc/meminfo` for the RAM fits, `nvidia-smi` for GPU probes, `trash-put` for deletes, PTYs for the inline REPL. |
| Python 3.10+ | The entry script pins `#! /bin/python3.10`; the code targets 3.10 and does not use newer stdlib without guards. |
| A reachable Ollama server | All inference, model listing and model management go to Ollama over its HTTP API. Default `127.0.0.1:11434`; see [pointing at Ollama](#pointing-at-ollama) below. |

## The code

Clone the repository somewhere it can live permanently:

```console
$ git clone https://github.com/vinibretas/ollamarun.git
$ cd ollamarun
```

The `ollamarun` file at the root is the whole entry point: it adds its own
directory to `sys.path` and calls into `ollamarun_lib/`, so the two must stay
together. If the checkout already sits on your `PATH` (this project is also a
git submodule of my personal `~/bin`), the bare `ollamarun` command resolves
straight to it.

## Dependencies

```console
$ pip install --user rich requests prompt_toolkit
```

Those three are the required runtime dependencies: `rich` for the rendering,
`requests` for the HTTP API and `prompt_toolkit` for the inline REPL. On a
PEP 668 "externally managed" distribution, add `--break-system-packages` (it
still only touches your user site) or use a virtualenv.

Everything else is optional and degrades gracefully:

| Extra | Enables | Without it |
| --- | --- | --- |
| `pylatexenc` | The best-quality LaTeX → Unicode conversion | A smaller built-in macro table does the conversion; nothing crashes. |
| `textual` | The deprecated `--ui tui` front end | `--ui tui` warns and falls back to the inline REPL. |
| `ddgs` | The `web_search` tool | The tool is not offered to the model. |
| `pypdf` | PDF text extraction for `fetch_url` | Falls back to `pdftotext` when it is installed. |
| `pyperclip` | `/copy` writes to the clipboard | `/copy` reports that pyperclip is missing. |

A few tools are wired to external commands and appear only where those
resolve: `tts-remote` (the `speak`/`list_voices` tools), `chemInventoryAPI`
(the read-only `chem_*` lookups), plus system tools the harness calls when
present — `trash-put` (deletes), `pdftotext` (PDFs), `nvidia-smi` (GPU
probes) and `tailscale` (`--remote`).

To run the test suite you additionally need the official SDK and the linter:

```console
$ pip install --user openai pyflakes
$ python3 test_ollamarun.py
```

## Verify

```console
$ ./ollamarun --version
ollamarun 2.0
```

`./ollamarun --help` prints the full flag surface (the same reference this
site keeps in [Usage and flags](../cli/usage.md)) and `./ollamarun serve
--help` covers the API server.

## First run

Nothing needs configuring up front. The first run creates the config file
from the built-in defaults, so a plain one-shot is enough to prove everything
works:

```console
$ ollamarun --no-tools llama3.2:3b "Say hello in one short sentence."
llama3.2:3b ›
Hello.
```

(That transcript is verbatim from a real run; `llama3.2:3b` happens to answer
in one or two words. Without `--no-tools` it may also call the shell — that is
the [tool harness](../cli/tools/index.md) doing its job.)

- Config: `~/etc/config_files/ollamarun.json` — created on first run; see
  [Config file](../config/file.md) for every key.
- Sessions: one JSON file plus an index under `~/etc/data/ollamarun/` — see
  [Storage](../cli/sessions/storage.md).
- Logs: `~/etc/logs/ollamarun.log` receives the debug log when `--debug` is
  set.

If the model is not installed on the target machine, the run does not fail
silently: it prints the machine's free disk space and offers to pull the
model (per-layer progress included). `--pull` skips the question, declining
opens the [model picker](../cli/models/picker.md), and `/pull` does the same
from inside the REPL.

## Pointing at Ollama

In precedence order, the upstream server comes from:

1. `--host HOST[:PORT]` on the command line.
2. `--remote NAME` for a tailnet machine — see
   [Remote inference](../cli/remote.md).
3. The `OLLAMA_HOST` environment variable.
4. The `host` config key.
5. The default, `127.0.0.1:11434`.

`ollamarun serve` has its own, separate `--host`/`--remote` for the upstream;
the API listener is chosen with `--bind`/`--port` — see
[API overview](../api/index.md).

## Running it without the checkout (optional)

The executable is self-contained with its package next to it, so putting the
checkout on `PATH` is enough:

```console
$ export PATH="$PATH:$HOME/code/ollamarun"    # then: ollamarun ...
```

The shebang `#! /bin/python3.10` matters only for the bare `./ollamarun`
form; if that interpreter lives elsewhere, call it explicitly with any 3.10+
Python instead:

```console
$ python3 /path/to/ollamarun --version
```

That works because the entry script re-adds its own directory to `sys.path`
before importing the package.
