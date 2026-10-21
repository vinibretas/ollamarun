# ollamarun

> A terminal agent harness for Ollama — a personal project about how far local
> models go when you hand them tools.

**Documentation:** guides, reference and API docs for every surface live at
<https://vinibretas.github.io/ollamarun/>.

ollamarun began as a nicer `ollama run`: streamed markdown, 2D Unicode math,
sessions that actually persist. It has since grown into a full harness — an
approval-gated tool loop, two front ends, remote inference, context/GPU
placement fitting, and an HTTP API layer (`ollamarun serve`) that speaks both
the OpenAI API and its own native agent API.

This is not a product. It is a one-person workbench for learning how local LLM
stacks fit together, and how much of a real agent harness you can build around
one. Every `ollama run` flag (ollama 0.34) is accepted and extended, so it
still doubles as a drop-in replacement for the CLI it started as.

## What I'm using it to learn

- **Local LLM orchestration** — talking to Ollama over raw HTTP, streaming
  responses by hand, model lifecycle (`/api/ps`, load/unload), GPU placement.
- **Harness engineering** — the unglamorous middle of agents: tool schemas,
  approvals that do not get in the way, cancellation, output truncation, and
  keeping a model on the rails across multi-round tool loops.
- **API compatibility and support** — making unmodified OpenAI clients work
  against local models (strict validation, SSE, embeddings, emulated `n > 1`),
  plus a native agent API (turns, events, parked approvals, sessions, fits)
  for clients that want the whole loop.
- **Terminal UX** — live markdown re-rendering, a self-contained
  LaTeX → 2D Unicode math layout engine, a collapsible thinking panel,
  tok/s + context footers.
- **Systems math** — sizing context windows from GGUF metadata and KV-cache
  geometry, VRAM budgets, and calibrating estimates with measured probes.
- **Data-driven prompting** — tool definitions and system prompts as editable
  files, not Python strings.

## The path so far

1. **`ollama run`, but nicer** — richer rendering, 2D math, persistent
   sessions, slash commands.
2. **Tools and approvals** — Ollama function calling with a read-only
   allowlist, numbered approval prompts with remembered rules, and sessions
   that keep the whole tool loop.
3. **A real terminal app** — a Textual TUI, remote inference over Tailscale,
   GPU/context fitting, image input, artifacts.
4. **An API layer** — `ollamarun serve`: an OpenAI-compatible `/v1` surface
   plus a native agent API for full turns, with reference clients and a
   compatibility test suite.

## Next: a web UI

The main thing on the roadmap is a web interface on top of `ollamarun serve`
as its backend. The native agent API already has the shape a browser client
needs — submit a turn, follow its event stream, answer parked approvals,
manage sessions and files — because that is the same loop the terminal drives.
See [Agent API](#agent-api-turns-events-approvals-sessions) for the surface it
will build on.

The existing Textual TUI is deprecated in the same light: it will be
rewritten as a pure client of that API instead of reaching into the runtime
in-process, so every front end speaks one interface.

## Status

Early, single-developer and moving. Built against one Linux box and a small
Tailscale fleet; Python 3.10; a few tools are wired to my own side projects
and degrade gracefully when those are absent. Rough edges are part of the
deal — expect the structure to keep changing as the learning continues.

## Highlights

- **Agent loop, not a pipe.** The model calls tools through Ollama function
  calling — shell, Python, files, web, your saved sessions, system stats,
  speech, the chemical inventory — in multi-round loops. Read-only calls run
  unattended; everything else asks, with numbered choices and remembered
  rules.
- **Rendering that keeps up.** Markdown re-renders while tokens stream, display
  math is laid out in 2D Unicode by a self-contained engine, thinking streams
  live in a collapsible panel, and a `~tok/s · ctx` footer ticks with every
  token (`--no-tps`/`/tps` switch it off).
- **Sessions that persist.** One atomic JSON per conversation plus an index
  under `~/etc/data/ollamarun/`, with search, export (md/json/html), branch,
  tags and `trash-put` deletes.
- **Two front ends.** The prompt_toolkit inline REPL (default) or `--ui tui`
  (deprecated — see [Front ends](#front-ends)): transcript pane, status bar,
  and an input box that stays usable while the model works.
- **Any machine on the tailnet.** `--remote NAME` runs all inference and model
  management on another machine — a name defined in config `remote_hosts`, or
  any `host[:port]` — while sessions, files and tools stay local.
- **Context and placement fitting.** `/num-ctx --max-gpu` finds the largest
  window that stays fully on the GPU, `/num-ctx --max-cpu N` fills DRAM to N%,
  `--probe` verifies by loading and calibrates later estimates, and
  `/gpu-layers` pins layer counts.
- **An API, not just a CLI.** `ollamarun serve` exposes the OpenAI API and a
  native agent API (turns, SSE events, approvals, sessions, artifacts, context
  fits) from the same process.
- **Model-facing text is data.** Tool definitions and prompts live as
  JSON/Markdown under `ollamarun_lib/defs/`, overridable per user without
  touching Python.

## Quickstart

Needs Linux, Python 3.10+ and a reachable Ollama server (default
`127.0.0.1:11434`). The entry point is pinned to `#! /bin/python3.10`; if that
interpreter is elsewhere, run `python3 ollamarun …` instead.

```console
$ pip install --user rich requests prompt_toolkit
$ ./ollamarun qwen3:8b "explain this repo"    # one-shot
$ ./ollamarun qwen3:8b                        # interactive REPL
```

Optional extras: `pylatexenc` (inline-math fallback), `textual` (the
deprecated `--ui tui` front end), `ddgs` (the `web_search` tool). The test
suite additionally needs `openai` and `pyflakes`:

```console
$ pip install --user openai pyflakes
$ python3 test_ollamarun.py
```

Serve the API layer and drive it from the reference clients:

```console
$ ./ollamarun serve                           # OpenAI + agent API on :11440
$ python3 examples/openai_client.py --model qwen3:8b
$ python3 examples/agent_client.py --prompt "List the files in /tmp"
```

## Contents

- [The tool harness](#the-tool-harness)
- [Rendering](#rendering)
- [Front ends](#front-ends)
- [Sessions](#sessions)
- [Models and housekeeping](#models-and-housekeeping)
- [Context and placement](#context-and-placement)
- [Remote](#remote)
- [OpenAI-compatible API (`ollamarun serve`)](#openai-compatible-api-ollamarun-serve)
- [Agent API (turns, events, approvals, sessions)](#agent-api-turns-events-approvals-sessions)
- [Example clients](#example-clients)
- [Prompts and tool definitions](#prompts-and-tool-definitions)
- [Usage](#usage)
- [Slash commands](#slash-commands)
- [Config, files and logs](#config-files-and-logs)
- [Layout](#layout)
- [Development](#development)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## The tool harness

With `--tools` (the default), the request carries the 23 built-in tool
definitions (some capability-gated) and the model can call them through Ollama
function calling. Calls and results are kept in the session (so
`/show`/`--show` replay them), a finished loop is separated from the committed
answer by a centered `response` divider, and every call/result panel is
colour-coded — one fixed colour per tool, a random palette colour (remembered
for the run) for anything unmapped, with config `tool_colors` overrides. The
result box wears the call's colour, prints ok/failed green/red in its title,
and the `speak` tool's box names where the voice is generated and where it
will be heard (`gpu-box → desk`, `(cached)` for a phrase served from disk).
Invented tool names (`search`, `curl`, `mkdir`, ...) and call-id-prefixed ones
(`call_fetch_url`) resolve onto the real tools through per-tool alias lists.
With a response format set (`--format`), the harness steps aside completely:
no tool schemas are sent and the tool rules leave the system prompt too, so
the model never sees a mention of a tool it cannot call.

A few tools are wired to my own side projects: `speak`/`list_voices` drive
`tts-remote` and the `chem_*` lookups go through `chemInventoryAPI`. They are
capability-gated — they only appear when the CLI resolves — and mostly serve
as examples of the extension mechanism: adding a tool (personal or general)
is one JSON file under `ollamarun_lib/defs/tools/` plus a handler.

### Tools

| Tool | What it does | Needs |
| --- | --- | --- |
| `run_shell` | Run a shell command (`bash -lc`), return output and exit code | — |
| `run_python` | Execute a short Python 3 program; the model is told to compute with it, never mentally | — |
| `read_file` | Read a text file, optionally a line range | — |
| `write_file` | Write or append text, creating the file | — |
| `edit_file` | Replace one exact fragment (unique match unless `replace_all`), atomic write keeping the mode; the call panel shows a coloured diff before approval | — |
| `list_dir` | Directory listing with sizes and modification times | — |
| `make_dir` | `mkdir -p` | — |
| `move_file` / `copy_file` | Move/copy; refuse an existing destination unless `overwrite=true` | — |
| `delete_file` | Move to the trash (`trash-put`) — never a permanent unlink | — |
| `publish_artifact` | Publish a generated file (from `run_shell`/`run_python`) as a downloadable artifact; the file tools publish their own results automatically | `ollamarun serve` artifact store |
| `fetch_url` | Fetch one exact URL as readable content — HTML to text plus the page's links, JSON pretty-printed, PDFs text-extracted; a URL already fetched in the session is answered from cache, and failures come back as one actionable line | — |
| `web_search` | Search the web for titles/URLs/snippets (cached; search-engine `q=` URLs are rerouted here) | `ddgs` package |
| `search_sessions` | Search your saved ollamarun conversations (titles, models, messages, thinking) | — |
| `list_models` | Installed models; loaded ones marked with VRAM and unload time (`loaded_only` for just the residents) | — |
| `system_status` | RAM, GPU VRAM and model-disk space of the machine running Ollama (ssh-probed under `--remote`; `target="local"` for this machine) | — |
| `context_status` | The context budget in use: window (labelled source), tokens the last request carried, free space, message counts, wrap-up note past 80% | — |
| `session_info` | Date/time, session id/title, model size/quantization/capabilities, server + cwd, parameters, thinking/render modes, system-prompt status, tool policy and remaining rounds | — |
| `speak` / `list_voices` | Speak text aloud through tts-remote (fire-and-forget), or list its voices | tts-remote |
| `chem_search` / `chem_container` / `chem_locations` | Read-only chemical-inventory lookups through the `chemInventoryAPI` CLI | chemInventoryAPI |

`web_search` only appears when `ddgs` is installed, the speech tools when
`tts-remote` is available, the chem tools when `chemInventoryAPI` resolves
(config `chem_inventory` names a different path, `chem_tools: false` hides
them), and `publish_artifact` when `ollamarun serve` runs an artifact store
(the CLI has none). A tool's `requires` capability gates it the same way for
user-added definitions.

### Approvals and policy

Read-only tools run unattended, as do allowlisted shell commands
(`ls`/`cat`/`grep`/`rg`, read-only `git`, ... — no substitutions or
redirections) and provably-harmless Python. Everything else lists numbered
choices — run once, always allow that exact call, or always allow a suggested
generalized rule (`shell: mkdir -p :*`, `path: ~/proj/**`,
`python: imports os`) stored for the session — and you type one. Unattended
runs deny with a note unless `-y/--yes` or `/tools policy auto` is in effect.
`dangerous_command()` is the floor no rule ever covers: substitutions,
redirections, privilege/wrapper/interpreter commands, disk-level tools,
catastrophic `rm` and pipes into an interpreter always prompt again.

- `/tools` — harness-wide `on|off`, `policy MODE`, `enable`/`disable NAME…`,
  `allow`, `rules`, plus per-tool control with `/tools NAME on|off|help|call
  ARGS` (NAME also takes the invented aliases, e.g. `/tools search off`).
  `help` is generated from the tool's definition JSON — description, every
  parameter with its type and whether it is required or optional, aliases,
  approval behaviour, and what an uninstalled tool needs — and `call` runs
  the tool yourself through the same approval gate and result panel as a
  model call (`call {"query": "x"}`, `call query=x max_results=3`, or a bare
  value when a single parameter is required); a manual call never touches the
  conversation. `/tools help` (and `/help tools`) is the reference.
- `--tools` / `--no-tools` for a single run; config `tools`, `tools_policy`,
  `tools_shell_allow`, `tools_timeout`, `tools_max_output`, `tools_max_rounds`
  (default 8), `tool_colors`

## Rendering

Output streams through a `rich.Live` view with a throttled markdown re-render
(raw streaming past `max_render_chars`, default 12000); `--pager` buffers into
the pager, `--render markdown|plain|raw` picks the mode.

- **2D Unicode math.** Display math (`$$…$$`, `\[…\]`) is laid out by a
  self-contained parser/layout engine — stacked fractions, sqrt overlines,
  sum/prod/lim limits above-below, integral side-limits, matrices, cases,
  aligned, `\boxed`, `\binom`, tall delimiters, accents,
  `\mathbb`/`\mathcal`/`\mathfrak`/`\mathbf` — centered in cyan, with
  flattened Unicode for inline math and `pylatexenc` as the never-crashing
  fallback (unknown env/over-wide/over-budget). `--no-math` / `/math` disable
  the conversion.
- **Thinking, collapsible.** Thinking streams live and is collapsed by default
  as an animated `thinking · N chars · Ctrl+T to expand` line. Ctrl+T expands
  a tail-capped boxed preview in place (press again to collapse; the stream
  never pauses), and `/thinking full` prints the stored panel later.
  `--show-thinking`/`--hidethinking` and config `thinking` set the start mode.
- **Live footer.** `~24.0 tok/s (avg 21.4) · 45 tok · ctx 12.345k/32.768k
  (38.123%)` — the 5s windowed rate, its cumulative average, the streamed
  token count and a context counter that ticks with every token. Its
  denominator is the window the model is actually loaded with (`/api/ps`; a
  session `num_ctx` wins, the architectural maximum is only the fallback),
  and counts read raw below 1k / 3-decimal k above. The TUI status bar shows
  the same rate, average and live ctx.
- **Timings.** `⏱ load … · prompt … · eval … · total …` prints under every
  answer on a terminal; `--verbose` is obsolete there and now only adds the
  line to piped/redirected output. `--raw`/`--json` never get it, and config
  `show_stats: false` turns it off.

## Front ends

The inline prompt_toolkit REPL is the default (`--ui inline`). It is a
superset of ollama's own commands, with `@path` file mentions (inlined into
the prompt), Tab completion for commands/session ids/models/`@files`, prompt
history (unless `OLLAMA_NOHISTORY` is set) and Ctrl+G to open
`$OLLAMA_EDITOR` on the current input. See [Slash commands](#slash-commands).

### TUI (`--ui tui`)

> **Deprecated.** The Textual front end still works, but it is not the future:
> it talks to the runtime in-process, and it will be rewritten as a pure client
> of the `ollamarun serve` API — the same direction as the planned web UI. The
> inline REPL is the supported front end meanwhile.

`--ui tui` (config `"ui": "tui"`, or `--ui auto` to use it when the optional
`textual` package is installed) separates the interactive session into panes:
the transcript up top, a rule, a status bar (spinner + state, model and remote,
thinking/tools/render modes, context tokens and %, live tok/s, queued count,
key hints) and an input box that stays available while the model works.

- **Type while it works.** Enter queues the message and it is handed to the
  model at the next sub-task boundary — right after a tool result, or in a
  follow-up round when an answer finishes streaming — so the model continues
  with the new information instead of dropping what it was doing. Commands
  that only read state or change display knobs run immediately; the rest
  queue and run once the turn ends, in order (`/queue` shows/edits the list).
- **Ctrl+B** sends what is typed right now: it cuts a streaming answer short
  but never kills a tool that is already running (the command keeps running
  and the model gets its result with your message in the next round).
- **Esc Esc** stops the turn (the first Esc arms it). The partial answer is
  kept as a cancelled message and anything queued then runs as the next turn.
- **Ctrl+T** thinking, **Ctrl+G** `$EDITOR`, **Ctrl+L** clear the transcript,
  **Ctrl+Q** quit (a second press when busy or with queued items), Tab
  completion and Up/Down history shared with the inline REPL. Tool approvals
  ask in a modal (`y`/`a`/`n`/`A`).

`OLLAMARUN_TUI_FULLSCREEN=1` runs the app on the alternate screen (the
transcript disappears on exit instead of staying in the terminal), and
`OLLAMA_TUI_DISABLE` keeps `--ui tui` on the inline REPL. The inline REPL is
also the fallback whenever Textual is missing or the output is not a terminal.

## Sessions

Every conversation is one atomic JSON file plus `index.json` under
`~/etc/data/ollamarun/sessions/` (prompt history in `~/etc/data/ollamarun/`,
honoring `OLLAMA_NOHISTORY`); this is a session store, not a cache — nothing
is deleted without going through `trash-put`.

- Start/resume: `-c/--continue [ID]`, `--session NAME`, `-r/--resume` picker,
  `--new` to force a fresh session, `--title TEXT`, `--tag TAG` (repeatable),
  `--auto-title` to ask the model for a title after the first turn.
- Inspect: `--list`, `--show ID`, `--search TEXT`, `--export ID`
  (`--export-format md|json|html`), `--delete ID [-y]`, `--reindex` — all with
  the shared output convention: rich table by default, `--json`/`--csv`/
  `--markdown` plus `--out FILE` to redirect.
- `--no-save` keeps a run out of the store entirely; `--json`/`--raw` cover
  scripting.

`/set`, `/show`, `/load`, `/save` (`/api/create`), `/clear`, `/new`, `/bye`
mirror ollama's REPL; `/undo`, `/retry`, `/edit`, `/branch`, `/context`,
`/summarize`, `/image`, `/copy`, `/export`, `/stats`, `/title`, `/tag`,
`/sessions` and the rest are described in [Slash commands](#slash-commands).

## Models and housekeeping

- The model picker (a model-less `ollamarun` run, `/model`) sorts
  by `modified` (newest first), `name`, `size`, `parameters` or `context`;
  `--sort KEY` seeds it and the picker prompt takes `--sort KEY` in place.
  `context` costs one lazy `/api/show` per model, cached per host.
- A model missing from the target machine (local or `--remote`) offers to pull
  it after printing that machine's free disk space (local `disk_usage`,
  remote via a BatchMode ssh `df` probe; degrades to unknown), with live
  per-layer progress. `--pull` pulls without asking, `/pull [MODEL]` does it
  from the REPL, and declining the prompt opens the model picker.
- Unloading: `/model` and `/load --model` unload the current model before
  switching, `/load SESSION` prompts current-vs-last model and unloads when
  switching, and REPL exit unloads (`unload_on_exit` config, `--keep-loaded`
  disables, `--unload` also unloads after one-shots). Embedding models are
  never unloaded.
- Embedding-capability models route to `/api/embed` honoring
  `--dimensions`/`--truncate` instead of chat.

## Context and placement

`--gpu-layers SPEC` / `/gpu-layers` sets how many layers run on the GPU: `N`,
`P%` of the model's layers (`block_count` from `/api/show`), `+N`/`-N`
adjustments (baseline: the current value, else a full offload), `all`, `cpu`
or `auto`. It is stored in the session, re-resolved per model, wins over
`/set parameter num_gpu`, and unloads a resident model so the next request
reloads with the new placement.

`/num-ctx` sizes the window:

- `N` sets it (capped to the model's max with a warning).
- `--max-gpu` picks the largest context that still keeps the model fully on
  the GPU. The model is profiled first: `/api/show`'s GGUF metadata supplies
  the per-layer KV geometry (hybrid attention, sliding windows and their
  per-layer key/value dims, MLA's compressed latent, recurrent `ssm.*` state)
  and the parameter count, `/api/tags` supplies the on-disk size and
  quantization, and the two are cross-checked (`params × bytes/param`; the
  model row in the report names all of it). Free VRAM (nvidia-smi, locally or
  over one ssh probe for `--remote` machines) is then reduced by llama.cpp's
  fit margin (`max_gpu_margin_mb`, 1 GiB by default); the separate
  `max_gpu_overhead_mb` reserve defaults to 0 because the fit margin already
  covers the buffers llama.cpp projects. Resident models' VRAM is credited
  back (it returns when they reload at the new context), and a pinned
  `/gpu-layers` is honoured: the output layer and the topmost blocks are
  counted as the offloaded share, and the non-block weights (token
  embeddings, output layer, vision projector) are estimated rather than
  split evenly. `--probe` then verifies by loading candidates and reading
  each real GPU split back from `/api/ps`, solving the measured
  bytes-per-token line for the capacity, and leaves the model warm at the
  largest 100%-GPU context. Resident and probe measurements calibrate future
  estimates using a conservative per-token slope and fixed overhead; the
  profile formula remains the fallback. The bounded cache is scoped to model
  digest, host, hardware, placement and KV settings, expires after 180 days,
  and is stored at `~/etc/data/ollamarun/ctxfit-calibration.json`.
- `--max-cpu N` fills DRAM instead: it raises the context until the model's
  host RAM reaches N% (1-100) of the DRAM available to it —
  `/proc/meminfo`'s MemAvailable on the Ollama machine (one ssh probe for
  `--remote` hosts) plus the RAM the loaded model already holds (`/api/ps`
  `size - size_vram`, back when it reloads), less a `max_cpu_overhead_mb`
  allowance. The same per-layer KV, sliding-window, recurrent-state and
  weight-split profile decides the placement: some layers may stay on the GPU
  unless `/gpu-layers` is `cpu`; a pinned layer count is honoured and only the
  host side counts. `--probe` solves the measured footprint line the same way
  and never chooses a context above the target. Compatible probe/resident data
  calibrates future CPU estimates as well; automatic GPU placement is checked
  before a cached slope is applied.

CLI forms: `--num-ctx N | max-gpu`, plus `--probe` and `--max-cpu N`. Config
knobs: `max_gpu_margin_mb` (1 GiB), `max_gpu_overhead_mb` (0; the fit margin
already covers llama.cpp's projected buffers), `max_gpu_probes` (10),
`max_cpu_overhead_mb` (256 MiB) and `max_cpu_probes` (10).

## Remote

`--remote NAME` (config `remote`/`remote_hosts`) runs the whole thing on
another machine: inference, model management (load/unload, `/api/ps`) and
`/pull` happen there, while sessions, files, images and the tool harness stay
local — there is no separate remote code path. Names resolve through
`tailscale status --json` (falling back to the MagicDNS name), and any
`host[:port]` works. Short names come from the config's `remote_hosts` map —
the shipped entries are just my own fleet, so replace them with yours. A
short `/api/version` preflight fails fast with the Tailscale node state when
the machine is down, and resuming a session reconnects to the machine it last
ran on.

## OpenAI-compatible API (`ollamarun serve`)

`ollamarun serve` runs a local HTTP API in front of Ollama, so any OpenAI
client can use the local models. The listener is LAN-accessible by default
(`0.0.0.0:11440`); `--bind IP` / `--port N` narrow or change it and are
separate from the upstream `--host` / `--remote`, which pick the Ollama
machine:

```console
$ ollamarun serve [--bind IP] [--port N] [--host HOST | --remote NAME] [--keep-loaded]
```

| Endpoint | What it does |
| --- | --- |
| `GET /v1/models` | installed models as OpenAI model objects |
| `GET /v1/models/{id}` | one model (404 + error envelope when unknown) |
| `POST /v1/embeddings` | one embeddings request (`input` as a string or an array of strings) |
| `POST /v1/chat/completions` | one model turn, streaming (`stream: true`) or not |

> **Trusted-network-only.** The service has no authentication; the bind
> address is its only boundary. Keep it on the LAN/tailnet and never expose
> the port to the public internet.

Model turns are a translation proxy: one request is one `/api/chat` call
(an `n > 1` request is emulated with that many sequential generations), and
one embeddings request is one `/api/embed` call. Client `tools` are passed
to the backend and returned as standard
`tool_calls` with `finish_reason: "tool_calls"` — they are never executed
here, so this endpoint cannot trigger the agent harness (full agent turns
live under the native `/v1/agent/*` endpoints). Streaming
emits OpenAI SSE only: a role chunk, content deltas, the finish chunk, an
optional usage chunk (`stream_options.include_usage`) and `data: [DONE]`;
thinking models surface their reasoning as the non-standard
`delta.reasoning_content`, which normal clients ignore. A dropped streaming
client cancels its own upstream generation.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11440/v1", api_key="unused")
print(client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "hello"}]).choices[0].message.content)
```

Supported fields: `temperature`, `top_p`, `max_tokens` /
`max_completion_tokens` (→ `num_predict`), `stop`, `seed`,
`presence_penalty`, `frequency_penalty`, `response_format` (`json_object` /
`json_schema` → Ollama `format`), `reasoning_effort` (low/medium/high →
Ollama `think`), `tools`, `tool_choice` (`auto`, `none`, or a specific
function), `stream`, `stream_options.include_usage`, and the extension
`system` (`"auto"` = the built-in default prompt, text = replace it, absent
= none; a client-sent system message always wins). `user` and `metadata`
are accepted and ignored (tracking hints); `n` runs that many generations
(1..8, emulated sequentially, with the counters summed — see the limits
note); `parallel_tool_calls` is honored: a `false` returns only the first
tool call the backend produced (the backend cannot be told to emit a single
call, so this is a truncation, not model steering).

**Embeddings.** `POST /v1/embeddings` maps to `/api/embed`:
`{model, input, encoding_format, dimensions, user}` with `input` a
non-empty string or array of non-empty strings (a client's token arrays are
rejected explicitly), `encoding_format` `"float"` (default) or `"base64"`
(the little-endian float32 packing the official SDK sends by default), and
`dimensions` forwarded to the backend for models that support truncated
vectors. The response is the OpenAI list shape — `{object: "list", data:
[{object: "embedding", index, embedding}], model, usage: {prompt_tokens,
total_tokens}}` — with the token counts read from the backend. The model
must advertise the `embedding` capability (`model_does_not_support_embeddings`
otherwise; an unverifiable empty capability list passes), the `serve_models`
allowlist applies, and the resolved-input budget bounds the inputs exactly
like chat content.

Request validation is strict, and every rejection is a structured error
rather than a silent ignore: unknown fields anywhere (top level, tools,
`tool_choice`, `response_format` wrappers), `n` outside 1..8 or non-integer
(each emulated choice is a full generation), `logprobs`/`top_logprobs`,
`logit_bias`, `audio`, non-text `modalities`
and `tool_choice: "required"` are rejected with `400`s; `strict: true` on a
function tool is rejected because the backend does not enforce
tool-argument schemas (`strict: false` is a no-op). `image_url` content
parts accept `data:` URIs or http(s) URLs (fetched with explicit timeouts, at
most 3 redirects and a 10 MiB cap — the SSRF caveat is the locked
trusted-network boundary), are validated against the model's `capabilities`
and refused with `model_does_not_support_images` when vision is absent. A
`{"type": "file", "file_id": "art_…"}` part resolves an uploaded artifact
(see the artifact section under the agent API): text-ish files are inlined
as a `<file name="…">` block, images go through the same vision gate, and
other binaries are described but never pretended to be processed.
`json_schema` enforcement uses Ollama's grammar
(`format`); only the `schema` object is forwarded. Errors use the OpenAI
envelope with proper status codes (400/404/405/408/413/502); upstream
failures surface as stable `model_not_found` / `backend_unreachable` /
`backend_error` messages, oversized bodies and stalled uploads are rejected
before parsing, and responses never carry tracebacks or internal paths —
`--debug` logs requests, upstream detail and tracebacks to
`~/etc/logs/ollamarun.log`.

**Limits.** The JSON body cap is `serve_body_limit` (default 32 MiB), and
it applies to `Content-Length` and chunked bodies alike: a chunked upload
is decoded with the cumulative decoded size checked before each chunk is
read, `Transfer-Encoding` together with `Content-Length` is an ambiguity
error, and malformed framing (bad sizes, missing CRLF, truncation, a
missing terminal chunk, a non-chunked coding) is a structured 400.
Resolving content parts is bounded separately, because a small body can
name many large inputs: `serve_max_messages`, `serve_max_content_parts`,
`serve_max_message_bytes`, `serve_max_text_bytes`, `serve_max_images` and
`serve_max_image_bytes` (cumulative after resolving data URIs, remote
fetches and file inlines; counts fail before any fetch). Embeddings inputs
are charged to the same budget (input count, per-input bytes, total text).
Violations are `413 input_too_large` naming the limit.
`serve_request_timeout` (default 30 s) is a per-operation socket timeout —
a streaming response that keeps writing (heartbeats included) is never cut
off by it — and `serve_sse_ping_interval` (default 15 s) is the SSE
heartbeat.

**Model allowlist.** `serve_models` (default: empty) restricts which
models the API exposes and accepts: names or fnmatch patterns matched
against the installed models after tag normalization (`qwen3` matches
`qwen3:latest`; `qwen3*` matches every tag). The list applies everywhere a
model is selected, listed or used — the model list, lookup, chat
completions, session creation and model edits, agent turns (a stored
session whose model fell out of the list is refused before it runs) and
the fit endpoints — and a model outside it answers exactly like an unknown
one (`404 model_not_found`).

**Compatibility testing.** The surface is pinned by the project's test
suite, which a client can trust as the compatibility contract: a required
round-trip through the official `openai` SDK (models, embeddings in both
encodings, completions with `n > 1`, streaming with usage, typed tool
calls, the 400 mappings, a mid-stream failure) plus the deterministic
request/response matrix and a real-process smoke of `ollamarun serve` —
`python3 test_ollamarun.py`, with `--live` adding a real-backend pass
(models, completions, embeddings, one agent turn). Everything accepted is
listed above, and everything the backend cannot do is rejected with a
structured 400 rather than silently ignored — permanently, for
`logprobs`/`top_logprobs` (Ollama exposes no token probabilities),
`logit_bias` (no backend equivalent), `audio` and non-text modalities
(Ollama has no audio modality), `tool_choice: "required"` (the backend
cannot force a call) and `strict: true` tool schemas (tool-argument schemas
are not enforced; `response_format` schemas are, through Ollama's `format`
grammar).

## Agent API (turns, events, approvals, sessions)

The same `ollamarun serve` process also exposes ollamarun's own endpoints:
instead of one inference call, a client submits a **full agent turn** — the
internal tool loop, approvals and session persistence run server-side — and
follows it over Server-Sent Events.

| Endpoint | What it does |
| --- | --- |
| `POST /v1/agent/turns` | start a full agent turn (`202` + the turn id and its URLs) |
| `GET /v1/agent/turns/{id}` | snapshot: `queued`/`running`/`waiting_approval`/`completed`/`failed`/`cancelled`, seq, result, error (404 unknown, 410 swept) |
| `GET /v1/agent/turns/{id}/events` | SSE turn events (`id: <turn>:<seq>`, typed `event:` names; replay with `?after=N` or `Last-Event-ID`) |
| `POST /v1/agent/turns/{id}/cancel` | request cancellation |
| `POST /v1/agent/approvals/{id}` | resolve a parked tool decision (`run`/`remember`/`rule`/`auto`/`deny`) |
| `POST /v1/files` | upload one file (`{filename, media_type, data_base64}`) as an artifact |
| `GET /v1/artifacts/{id}` | download an artifact's bytes (recorded media type) |
| `GET /v1/artifacts/{id}/meta` | the artifact's `ArtifactRef` |
| `POST /v1/agent/fit` | synchronous context estimate (optional `apply`) |
| `POST` `/v1/agent/fits` | start an asynchronous probe job (202 + warnings) |
| `GET /v1/agent/fits/{id}` / `POST .../cancel` | job status, progress and result / cancel |
| `GET`/`POST /v1/sessions` | list / create sessions |
| `GET`/`PATCH`/`DELETE /v1/sessions/{id}` | inspect, edit (title/model/system/tags/parameters/`num_ctx`/`gpu_layers`) or trash a session; edits and deletes are refused (409 `session_busy`) while a turn or fit runs |

```json
POST /v1/agent/turns
{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "Clean up /tmp/build"}],
  "persist": true,
  "settings": {
    "tools": true, "max_rounds": 8,
    "approval": "interactive",
    "system": null, "parameters": {},
    "cancel_on_disconnect": false
  }
}
```

For an existing session (`session_id`) the messages are the delta; for a new
one they seed the history. Turns persist by default (`persist: false` runs
ephemeral). `settings.parameters` — and the sessions' `parameters` field,
including the PATCH merge — is validated: known Ollama options get their
type and range checked (the same ranges the OpenAI mapper enforces),
unknown names pass through but are bounded to scalars and small lists with
size caps, and non-finite numbers or nested structures are refused with a
400. The event stream carries `turn.started`, `message.delta`,
`tool.call`/`tool.result`, `approval.required`/`approval.resolved`,
`turn.warning` and the terminal `turn.completed`/`turn.error`/
`turn.cancelled`, each with a deterministic `<turn_id>:<seq>` id — a
reconnecting client resumes with `Last-Event-ID`, a truncated buffer is
reported as `stream.gap`, and the turn status endpoint is the recovery path
afterwards. A subscription without `?after=`/`Last-Event-ID` is live-only —
nothing is replayed, so a finished turn yields an empty stream; catching-up
clients pass `after=0` or read the status snapshot instead.

Concrete requests — a new session seeds the history, an existing one gets the
delta (`persist` defaults to true) — and the turn's URLs come back immediately:

```console
$ curl -s http://127.0.0.1:11440/v1/agent/turns \
    -H 'Content-Type: application/json' -d '{
      "session_id": "ses_…",
      "messages": [{"role": "user", "content": "now file it under reports"}],
      "settings": {"approval": "interactive"}
    }'
{"id": "trn_…", "object": "turn", "session_id": "ses_…", "status": "queued",
 "created_at": "…", "status_url": "/v1/agent/turns/trn_…",
 "events_url": "/v1/agent/turns/trn_…/events",
 "cancel_url": "/v1/agent/turns/trn_…/cancel"}

$ curl -s http://127.0.0.1:11440/v1/agent/turns \
    -H 'Content-Type: application/json' -d '{
      "model": "qwen3:8b",
      "messages": [{"role": "user", "content": "Summarize this repo"}],
      "persist": false,
      "settings": {"max_rounds": 4, "system": "One paragraph, no lists.",
                   "num_ctx": "max-gpu", "gpu_layers": "80%"}
    }'
```

Follow it over SSE — each frame is named after its event type and carries a
`<turn_id>:<seq>` id, so a reconnect resumes with `?after=N` or
`Last-Event-ID`:

```console
$ curl -N 'http://127.0.0.1:11440/v1/agent/turns/trn_…/events?after=0'
id: trn_…:1
event: turn.started
data: {"id": "trn_…:1", "seq": 1, "type": "turn.started", "turn_id": "trn_…",
       "session_id": "ses_…", "ts": "…",
       "data": {"model": "qwen3:8b", "session_id": "ses_…"}}
id: trn_…:2
event: message.delta
data: {"id": "trn_…:2", "seq": 2, "type": "message.delta", "turn_id": "trn_…",
       "session_id": "ses_…", "ts": "…",
       "data": {"channel": "content", "text": "The repo is"}}
id: trn_…:3
event: tool.call
data: {"id": "trn_…:3", "seq": 3, "type": "tool.call", "turn_id": "trn_…",
       "session_id": "ses_…", "ts": "…", "data": {"name": "run_shell",
       "args": {"command": "ls -la"}, "detail": "ls -la", "round": 1}}
id: trn_…:4
event: tool.result
data: {"id": "trn_…:4", "seq": 4, "type": "tool.result", "turn_id": "trn_…",
       "session_id": "ses_…", "ts": "…", "data": {"name": "run_shell",
       "ok": true, "text": "total 42\n…", "duration": 0.03, …}}
id: trn_…:5
event: turn.completed
data: {"id": "trn_…:5", "seq": 5, "type": "turn.completed", "turn_id": "trn_…",
       "session_id": "ses_…", "ts": "…",
       "data": {"content": "…", "tool_calls": 1, "rounds": 2, "stats": {…}, …}}
```

The snapshot endpoint gives the same result with no stream (and a finished
turn has no live events, so this — or `?after=0` — is how a late client
catches up):

```console
$ curl -s http://127.0.0.1:11440/v1/agent/turns/trn_…
{"id": "trn_…", "object": "turn", "session_id": "ses_…", "status": "completed",
 "seq": 5, "result": {"status": "completed", "content": "…", "rounds": 2,
 "usage": {"prompt_tokens": 412, "completion_tokens": 96, "total_tokens": 508},
 "artifacts": [], "fit": null}, "error": null}

$ curl -s -X POST http://127.0.0.1:11440/v1/agent/turns/trn_…/cancel
{"id": "trn_…", "object": "turn", "cancel_requested": true, "status": "running"}
```

**Approvals.** `settings.approval` decides who answers:

- `"policy"` (the default) applies the configured tool policy — allowlisted
  and rule-covered calls run, everything else is denied with a note. The
  serve policy is never `auto`; `--yes` remains the only CLI auto path.
- `"interactive"` parks genuinely undecided calls: the turn enters
  `waiting_approval` and emits `approval.required` with the `approval_id`,
  the call, the reason and suggested rules. `POST /v1/agent/approvals/{id}`
  resolves it with `run`, `remember` (this exact call from now on), `rule`
  (a generalized `{"kind", "pattern"}` grant), `auto` (everything for the
  rest of this turn) or `deny`. The first decision wins; duplicates get
  `409`. An unanswered approval times out (config `serve_approval_timeout`,
  default 300 s) as **denied**: the tool result says `not run: approval
  timed out` and the model continues.

The parked event carries everything needed to decide (the client already saw
it on the stream, as `approval.required`):

```json
{"name": "run_shell", "args": {"command": "rm -rf build"},
 "detail": "rm -rf build", "fingerprint": "…", "reason": "catastrophic rm",
 "approval_id": "apr_…",
 "suggestions": [{"kind": "shell", "pattern": "rm -rf build",
                  "base": "", "created": "…"}]}
```

The decision is one POST — the body's `action` picks the outcome:

```console
$ curl -s http://127.0.0.1:11440/v1/agent/approvals/apr_… \
    -H 'Content-Type: application/json' -d '{"action": "run"}'
{"id": "apr_…", "object": "approval", "turn_id": "trn_…", "session_id": "ses_…",
 "name": "run_shell", "status": "resolved", "action": "run", "run": true,
 "note": "confirmed", "resolved_at": "…"}
```

| Request body | Effect |
| --- | --- |
| `{"action": "run"}` | Run this call once |
| `{"action": "remember"}` | Run it, remembering the exact call for the session |
| `{"action": "rule", "rule": {"kind": "shell", "pattern": "make :*"}}` | Run it and store a generalized rule (`call`, `shell`, `path` or `python`) |
| `{"action": "auto"}` | Allow everything for the rest of this turn |
| `{"action": "deny"}` | Refuse it; the model gets a not-run note |

**Disconnects and cancellation.** A dropped event stream is an
unsubscribe — the turn keeps running (its results land in the session)
unless it asked for `"cancel_on_disconnect": true`. `POST .../cancel`
cancels explicitly: it aborts the in-flight generation, kills cancellable
subprocesses, resolves a waiting approval as cancelled and persists the
terminal state. Shutdown is an ordered lifecycle: the listener stops
accepting, new turns and fits are refused (`503 server_shutting_down`),
parked approvals resolve as cancelled, in-flight HTTP handlers drain
within `serve_shutdown_timeout`, and `serve_unload_on_exit` (default true)
unloads only the models this server actually used — embedding models stay
resident and a parallel CLI session's models are never touched
(`--keep-loaded` disables it).

**Tool workspace.** Serve runs its tool context jailed to
`serve_workspace` (default: the directory serve started in, printed in the
banner): relative paths in the file tools resolve under it, resolved
escapes — symlinks included — are refused, and the `run_shell`/`run_python`
working directories follow the same rule. This is an argument and
default-working-directory policy, **not** subprocess confinement: an
approved command can still reach whatever the process user can. The CLI
keeps its previous behavior.

**Feature switches.** `serve_agent: false` runs the process as an
inference server: the native routes (turns, approvals, sessions, fits) are
not registered at all (404), while models, chat completions and the
`/v1/files` + `/v1/artifacts/*` endpoints keep serving — the OpenAI
surface can still send `file` parts. One model policy applies to every
route: see `serve_models` above.

Turn records live in memory — a restart loses live turns (404/410), while
session history stays under `~/etc/data/ollamarun/sessions/`. One active
turn per session (`409 session_busy` with `active_turn_id`; editing or
deleting the session is the same 409 while the turn runs), a global cap
(config `serve_max_active_turns`, `429`), and an optional wall-clock limit
per turn (config `serve_turn_timeout`, `0` = no limit).

**Files and artifacts.** `POST /v1/files` uploads one file as an artifact
(`{filename, media_type, data_base64}`, strict JSON) and returns the full
record:

```json
{"id": "art_…", "media_type": "text/plain", "filename": "report.txt",
 "size": 1842, "sha256": "…", "source": "user", "turn_id": null,
 "session_id": null, "created_at": "…", "url": "/v1/artifacts/art_…"}
```

```console
$ B64=$(base64 -w0 report.txt)
$ curl -s http://127.0.0.1:11440/v1/files \
    -H 'Content-Type: application/json' \
    -d "{\"filename\": \"report.txt\", \"media_type\": \"text/plain\",
         \"data_base64\": \"$B64\"}"
{"id": "art_…", "url": "/v1/artifacts/art_…", …}

$ curl -s -o report.txt http://127.0.0.1:11440/v1/artifacts/art_…   # download
$ curl -s http://127.0.0.1:11440/v1/artifacts/art_…/meta           # the ref
```

Messages reference it as `{"type": "file", "file_id": "art_…"}` — in
agent-turn content arrays and in OpenAI messages alike; a missing or
expired id is a structured `404 artifact_not_found` before inference:

```json
{"role": "user", "content": [
  {"type": "text", "text": "Summarize this report"},
  {"type": "file", "file_id": "art_…"}
]}
```

Artifacts are **immutable snapshots** (the hash and size come from the
stored bytes), live under `~/etc/data/ollamarun/artifacts/` (opaque ids,
index-driven lookup, sanitized display names only, and a resolved-root
containment invariant: every index/manifest path must resolve inside the
store root, so a symlinked directory under it can never redirect a read or
a sweep) and expire after
config `serve_artifact_ttl` (default 7 days; the sweep also reaps stale
reservations and debris at startup). The per-file cap is
`serve_artifact_max_bytes` (default 16 MiB — the base64-in-JSON request
must fit the 32 MiB body cap) and `serve_artifact_quota_bytes` (default
2 GiB) is reserved under a lock so concurrent uploads cannot overshoot a
`507 artifact_quota_exceeded`. Tool outputs publish automatically
(`write_file`/`edit_file`/`copy_file`/`move_file` snapshot their result,
regular files only); scripts publish explicitly through the
approval-gated `publish_artifact` tool. Every publication emits
`artifact.created` with the full record before `tool.result`, which
references the ids; turn results and session metadata list the refs. A
`persist: false` turn still registers and serves its artifacts — retention
is TTL-only and shared with session-deleted conversations (deleting a
session never deletes artifacts early).

**Context and placement.** A session's context size and GPU placement are
first-class fields: `PATCH /v1/sessions/{id}` (and the turn `settings`)
accept `num_ctx` (a positive integer, refused past the model's trained
maximum) and `gpu_layers` (`N`, `P%`, `+N`/`-N`, `all`, `cpu` or `auto`; a
change unloads the resident model, exactly like `/gpu-layers`).

```console
$ curl -s http://127.0.0.1:11440/v1/sessions \
    -H 'Content-Type: application/json' \
    -d '{"model": "qwen3:8b", "title": "build log triage", "tags": ["infra"]}'
{"id": "ses_…", "object": "session", "title": "build log triage",
 "model": "qwen3:8b", "host": "http://127.0.0.1:11434", "tags": ["infra"],
 "messages": 0, "turns": 0, "system": null, "artifacts": [],
 "active_turn_id": null, "created_at": "…", "updated_at": "…"}

$ curl -s -X PATCH http://127.0.0.1:11440/v1/sessions/ses_… \
    -H 'Content-Type: application/json' \
    -d '{"num_ctx": 32768, "gpu_layers": "80%",
         "parameters": {"temperature": 0.4}}'
```

Fitting comes in two speeds. `POST /v1/agent/fit` estimates synchronously
(no model loads) — `{model?, session_id?, mode: "gpu"|"cpu", cpu_percent,
gpu_layers?, apply?}` — and returns the `num_ctx`, the bound and a report
(parameters, weights, per-token KV, machine budget); `apply: true` writes
it to the session. `POST /v1/agent/fits` starts the same fit as an
asynchronous probe job that verifies the boundary by actually loading the
model: `202` with the job id and warnings (other resident models or active
turns make it slower; prior measurements on this host make it faster),
`GET /v1/agent/fits/{id}` for status/progress/result, `POST .../cancel` to
stop it. A job with a session holds it — turns, edits and a second fit get
`409 session_busy` with `active_fit_id` — a global cap
(`serve_max_active_fits`) answers `429`, and completed jobs live until
`serve_fit_ttl`. Turn `settings.num_ctx` also accepts `"max-gpu"` /
`"max-cpu"` (with `settings.cpu_percent`): the turn estimates at start —
never a blocking probe — persists the fitted context and reports it as a
typed `session.fit` event and `result.fit` on the turn status; a failed fit
leaves the context unchanged. `serve_fit` gates the whole surface:
`"probe"` (default), `"estimate"` (no probing) or `"off"`.

```console
# estimate only -- never loads the model; `apply` writes it to the session
$ curl -s http://127.0.0.1:11440/v1/agent/fit \
    -H 'Content-Type: application/json' \
    -d '{"session_id": "ses_…", "mode": "gpu", "apply": true}'
{"object": "fit", "model": "qwen3:8b", "mode": "gpu", "num_ctx": 131072,
 "source": "estimate", "bound": "max_gpu", "gpu_layers": null, "applied": true,
 "report": {"weights_bytes": …, "kv_per_token": …, "free_bytes": …, …}}

# verify by loading: an asynchronous probe job (202), then poll it
$ curl -s http://127.0.0.1:11440/v1/agent/fits \
    -H 'Content-Type: application/json' \
    -d '{"session_id": "ses_…", "mode": "cpu", "cpu_percent": 80}'
{"id": "fit_…", "object": "fit", "status": "queued", "session_id": "ses_…",
 "mode": "cpu", "probe": true, "apply": false, "warnings": [],
 "status_url": "/v1/agent/fits/fit_…",
 "cancel_url": "/v1/agent/fits/fit_…/cancel"}

$ curl -s http://127.0.0.1:11440/v1/agent/fits/fit_…
{"id": "fit_…", "object": "fit", "status": "completed",
 "progress": {"stage": "probe", "mode": "cpu", "ctx": 65536},
 "applied": false, "result": {"model": "qwen3:8b", "mode": "cpu",
 "num_ctx": 65536, "source": "probe", "bound": "max_cpu", …}}
```

**Deployment and auth.** `ollamarun serve` is meant to run on the trusted
network under the user's own service management (the `serve_*` keys cover
the listener, limits and shutdown; the startup banner prints the surface in
effect). There is deliberately no authentication yet — the bind address is
the only boundary, so the port must never reach the public internet. When
authentication lands it wraps these routes without reshaping them: it is a
transport concern, and never a substitute for the tool-approval gate.

## Example clients

Two reference clients under `examples/` double as integration checks — both
print a `PASS`/`SKIP`/`FAIL` line per step and exit non-zero on a failure:

- `examples/openai_client.py` — the official `openai` SDK against the `/v1`
  surface: the model list, a completion, streaming with usage, embeddings
  (the SDK's default base64 encoding), `n = 2` and a tools request. Needs
  the SDK (`pip install --user openai`); pass a chat model:

  ```console
  $ python3 examples/openai_client.py --model qwen3:8b
  ```

- `examples/agent_client.py` — the native agent flow over plain HTTP:
  submit a turn, follow the typed SSE stream, resolve a parked approval,
  read the session summary. `--mode interactive --approve run` exercises
  the approval path; `--session-id` continues an existing session:

  ```console
  $ python3 examples/agent_client.py --prompt "List the files in /tmp"
  ```

Both take `--base-url` for a remote or non-default listener (the OpenAI one
wants the `/v1` suffix).

## Prompts and tool definitions

The model-facing text lives in files, not code:

- `ollamarun_lib/defs/tools/*.json` — one JSON per tool: `name`, the
  `description` and JSON-schema `parameters` the model sees, `read_only` /
  `always_safe` safety flags, `requires` (`search` for `web_search`'s ddgs
  backend, `tts` for the speech tools, `chem` for the inventory tools), the
  model-invented `aliases` that resolve onto it, and the `handler` that runs
  it.
- `ollamarun_lib/defs/prompts/` — `system.md` (the built-in default system
  prompt), `system_tools.md` (the tool, web-search, math-in-code and
  tool-result rules, appended when tools are on), `title.md` and
  `summarize.md` (both take the text to wrap as `{{content}}`).

A same-named file under `~/etc/config_files/ollamarun-defs/` overrides the
shipped one, so prompts and tool wording can be tuned without touching
Python. Bad definition files warn and are skipped instead of breaking the
harness.

Every request carries a system prompt: the session's own (`--system`,
`--system-file`, config `system`, `/system TEXT`) when set, the built-in
default otherwise. `--no-system` / config `system: ""` / `/system off` send
none, `/system clear` returns to the default, and `/system` alone shows what
is in effect. It is sent as a leading system message and never enters the
saved history.

## Usage

```console
$ ollamarun MODEL [PROMPT] [--host HOST | --remote NAME] [--ui MODE]
                  [--pull] [--sort KEY] [--gpu-layers SPEC]
                  [--num-ctx [N | max-gpu | --max-cpu N] [--probe]]
                  [--think [STR]] [--render MODE] [--option K=V]
                  [--image PATH] [--system TEXT | --system-file FILE | --no-system]
                  [-c [ID] | --session NAME | -r] [--new] [--title TEXT] [--tag TAG]
                  [--tools | --no-tools] [-y] [--speak | --no-speak]
                  [--list | --show ID | --search TEXT | --export ID] [--export-format FMT]
                  [--delete ID [-y]] [--reindex] [--out FILE] [--no-save]
                  [--json | --csv | --markdown | --raw] [--verbose]
                  [--show-thinking | --hidethinking] [--no-math] [--no-tps]
                  [--unload | --keep-loaded] [--pager] [--timeout SECONDS] [--debug]
                  # ollama-compatible: --dimensions --format --hidethinking --insecure
                  #   --keepalive --nowordwrap --think --truncate --verbose
                  # Ctrl+T expands/collapses the thinking panel while streaming
```

## Slash commands

Dispatched in the REPL and the TUI; `/help` also lists the shortcuts (`/?`).

| Command | What it does |
| --- | --- |
| `/help, /? [topic]` | Show commands (topic: shortcuts) |
| `/set parameter NAME VALUE \| system TEXT` | Session parameters / system prompt |
| `/show [info\|license\|modelfile\|parameters\|system\|template]` | Model details |
| `/load [MODEL\|SESSION]` | Load a saved session or switch model |
| `/save NAME` | Create an Ollama model from this session |
| `/pull [MODEL]` | Pull a model onto this host (updates if present) |
| `/clear` | Forget the context, keep settings, start a new session |
| `/new` | Start a brand new session |
| `/sessions [filter]` | Browse and switch saved sessions |
| `/title [TEXT\|--auto]` | Show or set the session title |
| `/tag [list\|add TAG…\|rm TAG]` | Session tags |
| `/model, /models [NAME\|--list\|--sort KEY]` | Switch model, list or sort the installed models |
| `/ps` | Show resident models (ollama ps) |
| `/num-ctx [N \| --max-gpu [--probe] \| --max-cpu N [--probe]]` | Context window: N, the largest that keeps the GPU, or N% of DRAM |
| `/gpu-layers [N\|N%\|+N\|-N\|all\|cpu\|auto]` | GPU layers for this model |
| `/params` | Show session parameters |
| `/tools [on\|off\|policy MODE\|enable\|disable NAME…\|NAME on\|off\|NAME help\|NAME call ARGS\|allow\|rules\|help]` | Tool harness, per-tool control (aliases resolve), generated help pages and direct calls |
| `/tts [on\|off\|voice NAME\|lang CODE\|tools on\|off\|voices]` | Auto-speak via tts-remote |
| `/think [on\|off\|low\|medium\|high]` | Thinking mode |
| `/thinking [show\|collapse\|hide\|toggle\|full]` | Thinking display (Ctrl+T while streaming) |
| `/system [TEXT\|clear\|off]` | Show or set the system prompt (built-in default when unset) |
| `/undo` | Drop the last exchange |
| `/retry` | Regenerate the last response |
| `/edit` | Edit the last message in `$EDITOR` and resend |
| `/branch [TITLE]` | Fork this conversation into a new session |
| `/context` | Context-window usage |
| `/summarize` | Compact older messages into a summary |
| `/image PATH… \| clear` | Attach images to the next message |
| `/render [markdown\|plain\|raw]` | Response rendering mode |
| `/math [on\|off]` | LaTeX→Unicode math conversion |
| `/pager [on\|off]` | Page long transcripts |
| `/tps [on\|off]` | Tokens/s footer while streaming |
| `/copy [N]` | Copy the last response (or its Nth code block) |
| `/export [--format md\|json\|html] [FILE]` | Export this session |
| `/stats` | Session and last-generation stats |
| `/queue [list\|clear\|rm N]` | Messages and commands typed while busy |
| `/bye, /exit` | Leave |

Auto-speak: `--speak` / config `tts_auto` / `/tts on` read every final answer
aloud through `tts-remote say --markdown` (fire-and-forget; the `speaking`
line ends with the same `X → Y` as the tool's result box).

## Config, files and logs

| Path | What |
| --- | --- |
| `~/etc/config_files/ollamarun.json` | Config (created from defaults on first run); `--config FILE` overrides |
| `~/etc/config_files/ollamarun-defs/{tools,prompts}/` | Per-user tool/prompt overrides |
| `~/etc/data/ollamarun/sessions/`, `index.json`, `history` | Session store, index and prompt history |
| `~/etc/logs/ollamarun.log` | Debug log (`--debug`) |

Selected config keys: `render`, `math`, `thinking`, `ui`, `show_stats`,
`tps_footer`, `max_render_chars`, `pager`, `auto_title`, `auto_compact`,
`image_storage`, `max_file_kb`, `options`, `unload_on_exit`, `gpu_layers`,
`tools*`, `tool_colors`, `tts*`, `chem_*`, `remote*`, `max_gpu_*`,
`max_cpu_*`, `timeout`, plus the `serve_*` matrix below. See
`DEFAULT_CONFIG` in `ollamarun_lib/constants.py` for the full set.

`ollamarun serve` validates every `serve_*` key at startup: an invalid
value names itself and exits 2, an omitted key uses the default, and a `0`
means exactly what the row says.

| Key | Default | Meaning / zero |
| --- | --- | --- |
| `serve_bind`, `serve_port` | `0.0.0.0`, 11440 | listener; port 0 = ephemeral |
| `serve_agent` | `true` | native agent routes; `false` = inference + files only |
| `serve_models` | `[]` | allowlist names/globs; empty = every model |
| `serve_workspace` | startup cwd | tool path-jail root |
| `serve_body_limit` | 32 MiB | request body cap (chunked included) |
| `serve_request_timeout` | 30 s | per-operation socket timeout |
| `serve_sse_ping_interval` | 15 s | SSE heartbeat |
| `serve_shutdown_timeout` | 10 s | bounds each shutdown stage |
| `serve_unload_on_exit` | `true` | unload the used models on exit |
| `serve_max_messages` | 16384 | messages per request |
| `serve_max_content_parts` | 4096 | content parts per request |
| `serve_max_message_bytes` | 8 MiB | one message's UTF-8 bytes |
| `serve_max_text_bytes` | 32 MiB | total resolved text per request |
| `serve_max_images` | 64 | images per request |
| `serve_max_image_bytes` | 64 MiB | cumulative decoded image bytes |
| `serve_max_active_turns` | 4 | concurrent turns (429 past it) |
| `serve_turn_event_buffer` | 10000 | per-turn event ring |
| `serve_turn_event_ttl` | 86400 s | completed-turn retention; 0 = keep |
| `serve_approval_timeout` | 300 s | parked approval deadline; 0 = wait forever |
| `serve_turn_timeout` | 0 | wall-clock turn cap; 0 = no cap |
| `serve_artifact_root` | `~/etc/data/ollamarun/artifacts/` | store root |
| `serve_artifact_max_bytes` | 16 MiB | per-file cap; 0 = unlimited |
| `serve_artifact_quota_bytes` | 2 GiB | store quota; 0 = unlimited |
| `serve_artifact_ttl` | 7 days | retention; 0 = keep forever |
| `serve_fit` | `probe` | `off` / `estimate` / `probe` |
| `serve_max_active_fits` | 2 | concurrent fits (429 past it) |
| `serve_fit_ttl` | 86400 s | completed-fit retention; 0 = keep |

## Layout

- `ollamarun` — thin executable entry point (`#! /bin/python3.10`); calls
  into `ollamarun_lib/`.
- `ollamarun_lib/` — the implementation, one module per pipeline stage
  (math rendering, HTTP client, sessions, REPL, TUI, CLI, ...) — see
  CLAUDE.md's Architecture section for what lives where.
- `ollamarun_lib/defs/` — the model-facing tool definitions and prompts.
- `test_ollamarun.py` — test suite. Run with `python3 test_ollamarun.py`.

`CLAUDE.md` and `AGENTS.md` (mirror copies) carry the working guidance for
agents editing this repo.

## Development

This is a personal project, but the internals are documented for anyone
reading along: `CLAUDE.md` and `AGENTS.md` (mirror copies) describe the module
layout, the conventions (Python 3.10, `rich`, argparse in `main()`) and where
to change what.

Run the tests with `python3 test_ollamarun.py`. The suite doubles as the
compatibility contract for both API surfaces — it round-trips the official
`openai` SDK and exercises a real `ollamarun serve` process against a stub
backend — and `--live` adds a pass against a real Ollama server.

There is no contribution process; issues are welcome, but the roadmap is
whatever I am learning next — currently the web UI.

## Acknowledgements

This project exists mostly because of other people's work:

- **[Ollama](https://ollama.com)** — the local server every request goes
  through. The whole harness is an unaffiliated wrapper around its HTTP API,
  and a very happy user of it.
- **The OpenAI API** — the `ollamarun serve` `/v1` surface reimplements
  OpenAI's HTTP schema (chat completions, embeddings, SSE streaming, tool
  calls) so unmodified OpenAI clients work against local models.
  "OpenAI-compatible" is descriptive only — no affiliation with, or
  endorsement by, OpenAI is implied.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — its KV-cache
  geometry and fit arithmetic are what the context/placement fitting is
  modelled on, and it performs the actual inference behind Ollama.
- **The Python ecosystem** — [rich](https://github.com/Textualize/rich)
  (rendering, tables, progress),
  [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit)
  (the inline REPL), [Textual](https://github.com/Textualize/textual) (the
  TUI), [pylatexenc](https://github.com/phfaist/pylatexenc) (inline-math
  fallback), [ddgs](https://github.com/deedy5/ddgs) (web search),
  [pypdf](https://github.com/py-pdf/pypdf) (PDF text extraction),
  [pyperclip](https://github.com/asweigart/pyperclip) (clipboard) and
  [requests](https://github.com/psf/requests) (HTTP) — plus the official
  [openai](https://github.com/openai/openai-python) SDK, which the test suite
  round-trips as the compatibility contract.
- **The system tools it leans on** — [Tailscale](https://tailscale.com)
  (`--remote`), [trash-cli](https://github.com/andreafrancia/trash-cli)
  (`trash-put` deletes), [poppler](https://poppler.freedesktop.org)
  (`pdftotext`) and NVIDIA's `nvidia-smi` (GPU probes).

## License

[Apache-2.0](LICENSE) — a personal project provided as is, with no warranty.
