# Tool reference

Every built-in tool, its parameters (as defined in
`ollamarun_lib/defs/tools/*.json`) and how it is approved. **Auto** means it
runs unattended; **asks** means the [approval gate](approvals.md) decides.
Tools with a `requires` entry only appear when that capability resolves.

## Shell and Python

### `run_shell` — asks (safe list)

Run a shell command (`bash -lc`), return output and exit code.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `command` | string | required | the command to run |
| `cwd` | string | optional | working directory |
| `timeout` | number | optional | seconds |

Allowlisted commands (`ls`, `cat`, `grep`, `rg`, `find`, `git status`, …)
run unattended per segment; anything else asks.

### `run_python` — asks (safe list)

Execute a short Python 3 program and return its stdout/stderr. The models are
told to compute with it, never mentally.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `code` | string | required | the python source to run |
| `timeout` | number | optional | seconds |

Safe modules only run unattended; `open`/`import os`-style code asks.

## Files

### `read_file` — auto (read-only)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `path` | string | required | file path |
| `offset` | integer | optional | first line (0-based) |
| `limit` | integer | optional | how many lines |

A PDF is refused here with a pointer to `read_pdf`.

### `read_pdf` — auto (read-only)

Read a PDF by path or URL. A text layer is extracted directly (pypdf, then
`pdftotext`); a scan or image-only PDF is rasterized with `pdftoppm` and read
by a vision model, one page at a time, with `--- page N ---` markers. The
session model does it when it reports `vision`; otherwise the best installed
vision model is picked automatically, the session model is unloaded for the
pass and the vision model is unloaded afterwards (the next round reloads the
session model). The result names the model that read the pages. Config:
`pdf_vision`, `pdf_vision_model`, `pdf_vision_switch`, `pdf_vision_pages`;
prompt: `defs/prompts/pdf_vision.md`.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `path` | string | optional | local PDF path (one of `path`/`url`) |
| `url` | string | optional | absolute http(s) PDF URL |
| `pages` | string | optional | e.g. `1-3,7` or `all` (default: the first pages) |
| `mode` | string | optional | `auto` (default), `text` (never a model), `vision` (always page images) |
| `max_chars` | integer | optional | character cap for the returned text |

### `write_file` — asks

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `path` | string | required | file path |
| `content` | string | required | text to write |
| `append` | boolean | optional | append instead of replacing |

### `edit_file` — asks

Replace one exact fragment; the match must be unique unless `replace_all`.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `path` | string | required | file path |
| `old_string` | string | required | the exact text to replace |
| `new_string` | string | required | the replacement text |
| `replace_all` | boolean | optional | replace every occurrence |

### `list_dir` — auto (read-only)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `path` | string | optional | directory (default: cwd) |

### `make_dir` — asks

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `path` | string | required | directory path |

### `move_file` / `copy_file` — ask

Refuse to replace an existing destination unless `overwrite`.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `source` | string | required | what to move/copy |
| `destination` | string | required | where to put it |
| `overwrite` | boolean | optional | replace an existing destination |

### `delete_file` — asks

Moves the target to the trash (`trash-put`) — never a permanent unlink.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `path` | string | required | file or directory to delete |

## Web

### `fetch_url` — auto (read-only)

Fetch one http(s) URL as readable content: HTML → text plus links, JSON
pretty-printed, PDFs text-extracted (a scan points at `read_pdf`). A URL
already fetched in the session is answered from the cache; failures come
back as one actionable line.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `url` | string | required | absolute http(s) URL |
| `timeout` | number | optional | seconds |

### `web_search` — auto (read-only) · requires `search` (`ddgs`)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `query` | string | required | what to search for |
| `max_results` | integer | optional | how many results (default 5, max 10) |
| `timeout` | number | optional | seconds |

## Sessions, models and the run

### `search_sessions` — auto (read-only)

Search your saved conversations — titles, models, messages and thinking.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `query` | string | required | word or phrase to look for |
| `limit` | integer | optional | max sessions (default 5, max 20) |

### `list_models` — auto (read-only)

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `loaded_only` | boolean | optional | only the resident models |

### `system_status` — auto (read-only)

RAM, GPU VRAM and model-disk free space of the machine running Ollama
(ssh-probed under `--remote`).

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `target` | string | optional | `ollama` (default) or `local` |

### `context_status` — auto (read-only)

The context budget actually in use: window (with its source), tokens the
last request carried, free space, message counts, a wrap-up note past 80%.

### `session_info` — auto (read-only)

Date/time, session id and title, model size/quantization/capabilities,
server and cwd, settings, thinking/render modes, system-prompt status, tool
policy and remaining rounds.

## Speech

### `speak` — auto · requires `tts`

Speak text through `tts-remote` (queued, returns immediately).

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `text` | string | required | what to say |
| `voice` | string | optional | voice slug |
| `lang` | string | optional | ISO language code |

### `list_voices` — auto (read-only) · requires `tts`

Lists the voices `speak` can use.

## Chemical inventory

Read-only lookups through the `chemInventoryAPI` CLI · requires `chem`.

| Tool | Parameters |
| --- | --- |
| `chem_search` | `query` (required), `type` (`name`/`cas`/`barcode`/`ghs`…), `limit` (default 10, max 50) |
| `chem_container` | `id` (required) — full container record |
| `chem_locations` | `location` (optional) — the tree, or the containers in one location |

## Artifacts

### `publish_artifact` — asks · requires `artifacts`

Publish a file (from `run_shell`/`run_python`) as a downloadable artifact in
`ollamarun serve`. The file tools publish their results automatically.

| Parameter | Type | | Description |
| --- | --- | --- | --- |
| `path` | string | required | file to publish (inside the working directory) |

## Running tools yourself

`/tools NAME call ARGS` runs a tool exactly like the model would — same
approval gate, same panels — without adding anything to the conversation.
Three argument forms:

- JSON: `/tools read_file call {"path": "/etc/hostname"}`
- keyed pairs: `/tools run_shell call command="echo hi"`
- one bare value, only when exactly one parameter is **required**:
  `/tools run_python call print(6*7)`

That last rule has a sharp edge: `list_dir`'s `path` is optional, so
`/tools list_dir call /tmp` refuses with a usage line — use
`/tools list_dir call path=/tmp`.

Tested examples, verbatim results condensed:

```text
llama3.2:3b › /tools read_file call /etc/hostname
╭─ tool · read_file · ok · 0.00s ──────────────────────────────────────────────╮
│ $ /etc/hostname                                                              │
│ /etc/hostname (1 lines shown)                                                │
│ 1: g3                                                                        │
╰──────────────────────────────────────────────────────────────────────────────╯

llama3.2:3b › /tools run_shell call echo hi from tools
╭─ tool · run_shell · ok · 0.00s ──────────────────────────────────────────────╮
│ $ echo hi from tools                                                         │
│ exit 0                                                                       │
│ hi from tools                                                                │
╰──────────────────────────────────────────────────────────────────────────────╯

llama3.2:3b › /tools run_python call print(6*7)
│ 42                                                                           │

llama3.2:3b › /tools list_models call loaded_only=true
│ 1 model(s) loaded on http://127.0.0.1:11434:                                 │
│ - llama3.2:3b · 2.6 GiB VRAM of 9.4 GiB · unloads 2026-10-18 14:08:33        │

llama3.2:3b › /tools search_sessions call color
│ 5 saved session(s) matching 'color':                                         │
```

And the awareness trio, which is what the model reads when it asks "how much
room is left?":

```text
llama3.2:3b › /tools context_status call
│ model: llama3.2:3b on this machine (http://127.0.0.1:11434)                  │
│ context window: 65536 tokens (loaded runner; model default 131072)           │
│ context used: nothing yet (no request has been sent)                         │

llama3.2:3b › /tools system_status call
│ this machine (ollama at http://127.0.0.1:11434)                              │
│ RAM: 16.2 GiB free of 31.2 GiB                                               │
│ GPU 0: 1.3 GiB free of 4.0 GiB                                               │
│ disk (models): 92.9 GiB free of 467.3 GiB at /usr/share/ollama/.ollama/models │

llama3.2:3b › /tools session_info call
│ session: 20261018-120810-fb11 "new session" · 0 messages · 0 turns           │
│ model: llama3.2:3b · 3.2B · Q4_K_M · capabilities: completion, tools        │
│ modes: thinking collapsed · render markdown · system prompt built-in default │
│ tools: on · policy allowlist · round 0 of 8 · disabled: none                 │
```

Each tool's own page (`/tools NAME help`) is the version guaranteed to match
your installed definitions and overrides.
