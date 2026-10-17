# ollamarun

A terminal agent harness for Ollama: run models, give them tools, keep the
conversation. It is a drop-in superset of `ollama run` — every ollama 0.34 run
flag is accepted — and adds an approval-gated tool-calling loop (22 built-in
tools), persistent sessions, live markdown + 2D Unicode math rendering, an
optional Textual TUI, context/GPU placement fitting, and inference on any
Tailscale machine.

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
- **Two front ends.** The prompt_toolkit inline REPL (default) or `--ui tui`:
  transcript pane, status bar, and an input box that stays usable while the
  model works.
- **Any machine on the tailnet.** `--remote NAME` runs all inference and model
  management on `pi`, `kiddo`, `g3`, `optiplex7070` (alias `optiplex`) or
  `guri` while sessions, files and tools stay local.
- **Context and placement fitting.** `/num-ctx --max-gpu` finds the largest
  window that stays fully on the GPU, `/num-ctx --max-cpu N` fills DRAM to N%,
  `--probe` verifies by loading, and `/gpu-layers` pins layer counts.
- **Model-facing text is data.** Tool definitions and prompts live as
  JSON/Markdown under `ollamarun_lib/defs/`, overridable per user without
  touching Python.

## The tool harness

With `--tools` (the default), the request carries the tool definitions and the
model can call them through Ollama function calling. Calls and results are
kept in the session (so `/show`/`--show` replay them), a finished loop is
separated from the committed answer by a centered `response` divider, and
every call/result panel is colour-coded — one fixed colour per tool, a random
palette colour (remembered for the run) for anything unmapped, with config
`tool_colors` overrides. The result box wears the call's colour, prints
ok/failed green/red in its title, and the `speak` tool's box names where the
voice is generated and where it will be heard (`optiplex → g3`, `(cached)` for
a phrase served from disk). Invented tool names (`search`, `curl`, `mkdir`,
...) and call-id-prefixed ones (`call_fetch_url`) resolve onto the real
tools through per-tool alias lists.

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
`tts-remote` is available, and the chem tools when `chemInventoryAPI` resolves
(config `chem_inventory` names a different path, `chem_tools: false` hides
them). A tool's `requires` capability gates it the same way for user-added
definitions.

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
  the GPU: KV bytes per token come from `/api/show`'s GGUF metadata (hybrid
  per-layer attention included), the model size from `/api/tags`, and free
  VRAM from nvidia-smi — locally or over one ssh probe for `--remote`
  machines — minus llama.cpp's fit margin and an overhead allowance. Resident
  models' VRAM is credited back (it returns when they reload at the new
  context), and a pinned `/gpu-layers` is honoured. `--probe` then verifies
  by loading candidates and reading each real GPU split back from `/api/ps`,
  solving the measured bytes-per-token line for the capacity, and leaves the
  model warm at the largest 100%-GPU context.
- `--max-cpu N` fills DRAM instead: it raises the context until the model's
  host RAM reaches N% (1-100) of the DRAM available to it —
  `/proc/meminfo`'s MemAvailable on the Ollama machine (one ssh probe for
  `--remote` hosts) plus the RAM the loaded model already holds (`/api/ps`
  `size - size_vram`, back when it reloads), less a `max_cpu_overhead_mb`
  allowance. Some layers may stay on the GPU unless `/gpu-layers` is `cpu`; a
  pinned layer count is honoured and only the host side counts. `--probe`
  solves the measured footprint line the same way and never chooses a context
  above the target.

CLI forms: `--num-ctx N | max-gpu`, plus `--probe` and `--max-cpu N`. Config
knobs: `max_gpu_margin_mb`, `max_gpu_overhead_mb`, `max_gpu_probes`,
`max_cpu_overhead_mb`, `max_cpu_probes`.

## Remote

`--remote NAME` (config `remote`/`remote_hosts`) runs the whole thing on
another machine: inference, model management (load/unload, `/api/ps`) and
`/pull` happen there, while sessions, files, images and the tool harness stay
local — there is no separate remote code path. Names resolve through
`tailscale status --json` (falling back to the MagicDNS name), and any
`host[:port]` works. A short `/api/version` preflight fails fast with the
tailscale node state when the machine is down, and resuming a session
reconnects to the machine it last ran on.

## OpenAI-compatible API (`ollamarun serve`)

`ollamarun serve` runs a local HTTP API in front of Ollama, so any OpenAI
client can use the local models. The listener is LAN-accessible by default
(`0.0.0.0:11440`); `--bind IP` / `--port N` narrow or change it and are
separate from the upstream `--host` / `--remote`, which pick the Ollama
machine:

```console
$ ollamarun serve [--bind IP] [--port N] [--host HOST | --remote NAME]
```

| Endpoint | What it does |
| --- | --- |
| `GET /v1/models` | installed models as OpenAI model objects |
| `GET /v1/models/{id}` | one model (404 + error envelope when unknown) |
| `POST /v1/chat/completions` | one model turn, streaming (`stream: true`) or not |

> **Trusted-network-only.** The service has no authentication; the bind
> address is its only boundary. Keep it on the LAN/tailnet and never expose
> the port to the public internet.

Model turns are a translation proxy: one request is one `/api/chat` call.
Client `tools` are passed to the backend and returned as standard
`tool_calls` with `finish_reason: "tool_calls"` — they are never executed
here, so this endpoint cannot trigger the agent harness (full agent turns
arrive with the native `/v1/agent/*` endpoints in a later phase). Streaming
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
are accepted and ignored (tracking hints).

Request validation is strict, and every rejection is a structured error
rather than a silent ignore: unknown fields anywhere (top level, tools,
`tool_choice`, `response_format` wrappers), `n` other than the integer `1`,
`logprobs`/`top_logprobs`, `logit_bias`, `audio`, non-text `modalities`
and `tool_choice: "required"` are rejected with `400`s; `strict: true` on a
function tool is rejected because the backend does not enforce
tool-argument schemas (`strict: false` is a no-op). `image_url` content
parts must be `data:` URIs of at most 10 MiB, are validated against the
model's `capabilities` and refused with `model_does_not_support_images`
when vision is absent. `json_schema` enforcement uses Ollama's grammar
(`format`); only the `schema` object is forwarded. Errors use the OpenAI
envelope with proper status codes (400/404/405/408/413/502); upstream
failures surface as stable `model_not_found` / `backend_unreachable` /
`backend_error` messages, oversized bodies and stalled uploads are rejected
before parsing, and responses never carry tracebacks or internal paths —
`--debug` logs requests, upstream detail and tracebacks to
`~/etc/logs/ollamarun.log`.

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
| `GET`/`POST /v1/sessions` | list / create sessions |
| `GET`/`PATCH`/`DELETE /v1/sessions/{id}` | inspect, edit (title/model/system/tags) or trash a session; edits and deletes are refused (409 `session_busy`) while a turn runs |

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
ephemeral). The event stream carries `turn.started`, `message.delta`,
`tool.call`/`tool.result`, `approval.required`/`approval.resolved`,
`turn.warning` and the terminal `turn.completed`/`turn.error`/
`turn.cancelled`, each with a deterministic `<turn_id>:<seq>` id — a
reconnecting client resumes with `Last-Event-ID`, a truncated buffer is
reported as `stream.gap`, and the turn status endpoint is the recovery path
afterwards.

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

**Disconnects and cancellation.** A dropped event stream is an
unsubscribe — the turn keeps running (its results land in the session)
unless it asked for `"cancel_on_disconnect": true`. `POST .../cancel`
cancels explicitly: it aborts the in-flight generation, kills cancellable
subprocesses, resolves a waiting approval as cancelled and persists the
terminal state; a shutdown does the same for every active turn.

Turn records live in memory — a restart loses live turns (404/410), while
session history stays under `~/etc/data/ollamarun/sessions/`. One active
turn per session (`409 session_busy` with `active_turn_id`; editing or
deleting the session is the same 409 while the turn runs), a global cap
(config `serve_max_active_turns`, `429`), and an optional wall-clock limit
per turn (config `serve_turn_timeout`, `0` = no limit).

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
`max_cpu_*`, `serve_bind`, `serve_port`, `serve_approval_timeout`,
`serve_turn_timeout`, `timeout`, plus the turn-manager keys
(`serve_max_active_turns`, `serve_turn_event_buffer`, `serve_turn_event_ttl`)
for the agent endpoints. See `DEFAULT_CONFIG` in
`ollamarun_lib/constants.py` for the full set.

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

## Usage as a submodule

This repo is checked out as a git submodule at `ollamarun/` inside
[`vinibretas/binscripts`](https://github.com/vinibretas/binscripts) (`~/bin`
on the machine it runs on). `~/bin/ollamarun` (this submodule's directory) is
added to `PATH` alongside `~/bin` itself so the `ollamarun` command resolves
straight to the script here.

See `~/bin/AGENTS.md` in binscripts for the surrounding tool conventions
(shebang pin, `rich` usage, argparse structure, config/log/data locations).
