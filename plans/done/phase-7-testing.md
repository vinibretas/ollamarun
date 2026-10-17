# Phase 7 — Testing and compatibility

The API layer (Phases 1–6 and 10) shipped its own tests with every phase; this
phase is the roll-up: one explicit coverage matrix, the audit's gap tests, the
two missing gates (a pyflakes pass and a *required* SDK test), a real-process
smoke check, and the suite split so the canonical command stays workable. It
changes no product behavior.

Status: implemented and landed at v1.57. The contracts below are what
shipped; the audit's findings and the deviations are recorded under "Recorded
implementation decisions".

## Baseline: what already exists (audited at v1.56)

- `test_ollamarun.py` is the single canonical suite (~13.4k lines, ~150
  `test_*` functions), run as `python3 test_ollamarun.py`; `--live` adds the
  real-backend smoke (`test_serve_live`). Every `test_*` table in the Phases
  2–6/10 plans has a landed function and the full suite is green at v1.56.
- `test_openai_sdk` (the official `openai` SDK round-trip) currently **skips**
  when the package is absent; it is installed here (2.29.0) and already
  covers models, completion, streaming + usage, typed tool calls, the 400
  mappings and a mid-stream failure.
- pyflakes 3.4 output over the package + entry script is only the deliberate
  re-export hubs (`constants.py`, `__init__.py`) plus **five real findings**:
  `ui.py` unused `dataclasses.field`/`typing.Callable`, `repl.py` two
  placeholder-less f-strings and the deliberate `_prompt_approval` re-export.
  The test file has five more (unused locals, `textual` import, and a genuine
  `undefined name 'Any'`).
- `/bin/python3.10` on this box is really 3.12.3 and no fleet host runs
  `serve`, so the `feature_version=(3, 10)` AST guard
  (`test_python310_syntax`) stays the 3.10 enforcement; there is no genuine
  3.10 interpreter to smoke against (recorded, not tested).

## Coverage matrix (requirement → coverage → level)

Levels: **unit** (function-level), **HTTP** (in-process server + fake
backend), **process** (the real `ollamarun serve` binary), **live**
(real Ollama, `--live` only).

| Phase-7 requirement | Covered by | Level |
|---|---|---|
| Normal inference | `test_serve_http`, `test_agent_turn_lifecycle`, `test_turn_manager_headless_turn`, `test_openai_sdk`, `test_serve_live` | HTTP/live |
| Multi-step tools | `test_tool_loop`, `test_tool_loop_separation`, `test_quiet_tools_real_turn`, `test_agent_policy_mode`, `test_agent_approvals` | unit/HTTP |
| Model vs agent isolation | `test_serve_http` (tools passthrough), `test_agent_policy_mode` (execution) — **no side-by-side pin** → §2 | HTTP |
| Tool failure (denial) | `test_unattended_turn_denial`, `test_tool_loop`, `test_agent_policy_mode` (deny) | unit/HTTP |
| Tool failure (error outcome) | `test_tool_execution` (unit only) → §2 | HTTP |
| Model failure | `test_turn_manager_synthesized_terminal` (crash), `test_serve_http` (502s), `test_openai_sdk` (mid-stream); **no end-to-end agent-turn failure** → §2 | HTTP |
| Approval required / timeout / duplicate | `test_agent_approvals`, `test_agent_approval_edge`, `test_serve_approval_timeout_zero`, `test_approval_runtime_contract` | unit/HTTP |
| Session persistence / recovery | `test_agent_turn_lifecycle`, `test_agent_turn_expiry`, `test_artifact_session_persistence`, `test_serve_shutdown` (on-disk contents) | HTTP |
| Session after cancellation | shutdown path asserted (`test_serve_shutdown`); **user-cancel path not** → §2 | HTTP |
| Cancellation | `test_turn_manager_cancel`, `test_cancel_reaches_production_tools`, `test_cancel_stops_turn`, `test_cancellation_destructive_guard` | unit/HTTP |
| Interrupted streams / disconnect | `test_serve_disconnect`, `test_agent_disconnect_policy`, `test_stream_cancel`, `test_subscriber_overflow` | unit/HTTP |
| Events: determinism, seq, lifecycle | `test_turn_event_envelope`, `test_turn_manager_lifecycle`, `test_agent_turn_lifecycle` | unit/HTTP |
| Events: replay, subscriber disconnect, truncation | `test_agent_turn_lifecycle` (replay), `test_turn_buffer_truncation`, `test_turn_subscriber_overflow`; **no HTTP-level `stream.gap`** → §2 | unit/HTTP |
| Lifecycle transitions | `test_agent_turn_lifecycle`, `test_agent_turn_controls`, `test_turn_manager_lifecycle` | HTTP |
| Shutdown while waiting / in flight | `test_serve_shutdown` (parked approval, on-disk session), `test_agent_approval_edge` | HTTP |
| HTTP schemas, SSE framing, errors | `test_serve_http`, `test_serve_sockets`, `test_openai_mapping`, `test_chunked_bodies` | HTTP |
| Error status/body, body limits | malformed matrix in `test_serve_http`/`test_openai_mapping`, `test_input_limits` | HTTP |
| No-auth behavior (locked contract) | implicit everywhere; **no explicit pin** → §2 | HTTP |
| Concurrency and session isolation | `test_serve_cache_concurrency`, `test_turn_manager_session_busy`, `test_turn_manager_max_active` | unit/HTTP |
| Static path traversal | `test_artifact_fetch_api` (traversal-shaped URLs), `test_workspace_jail`, `test_artifact_root_containment` | HTTP |
| Artifacts: upload/fetch/ids/provenance/TTL/quota | the nine `test_artifact_*` functions | unit/HTTP |
| Compatibility: official SDK | `test_openai_sdk` — currently skip-if-absent → §3 | HTTP |
| CLI regression, headless no-stdout | the whole suite; `test_headless_renderer` ("writes nothing to stdout"), `test_turn_manager_headless_turn` | unit |

## Workstreams

### 1. Coverage matrix first (this section)

Every phase-7 requirement is mapped above to test(s) and level before any test
is added; the five gaps it found are the only new coverage.

### 2. Gap tests (extend existing owners, no duplicates)

- **Model vs agent isolation** — new `test_turn_mode_isolation`: the same
  scripted `tool_calls` on both surfaces; `/v1/chat/completions` returns them
  with exactly one inference call and executes nothing, while the
  `/v1/agent/turns` twin runs an allowlisted tool through the loop, makes the
  second round and records the tool message in the session (the locked
  "model turns and agent turns stay distinguishable" decision).
- **Agent-turn backend failure** — extend `test_agent_turn_lifecycle`:
  scripted `{"error": "boom"}` → status `failed`, non-empty `error`, terminal
  `turn.error`, the persisted session keeps the user message and no assistant
  answer.
- **Tool error mid-turn** — extend `test_agent_policy_mode`: an allowlisted
  `ls` on a missing path returns `tool.result ok: false`, the model continues
  and the session records the failed result.
- **HTTP `stream.gap`** — extend `test_turn_buffer_truncation`: a small
  `serve_turn_event_buffer` runs a real turn, and `/events?after=0` starts
  with the `stream.gap` SSE event before the surviving seq range.
- **No-auth pin** — extend `test_serve_http`: requests with and without an
  `Authorization` header behave identically (200s, no `WWW-Authenticate`,
  no auth language in errors) — the locked trusted-network contract.
- **Session after user cancellation** — extend `test_agent_turn_controls`:
  after cancelling a mid-stream turn, the stored session keeps the user
  message and the cancelled (empty) assistant message.

### 3. Required SDK gate

`test_openai_sdk` **fails** (never skips) when `openai` is missing and prints
the SDK version it exercised; `openai` becomes a required test dependency
(and stays a test-only one — the runtime never imports it). Coverage is
already complete: models, typed completion, streaming + usage, tool calls,
400 mappings, mid-stream `APIError`.

### 4. Real-process smoke

`test_serve_process` (deterministic, every run): launches the real entry point
(`ollamarun serve --bind 127.0.0.1 --port 0`, temp config with a temp
artifact root/workspace) against a stub Ollama upstream
(`BaseHTTPRequestHandler`, the `test_artifact_remote_fetch` pattern), reads
the bound address from the startup banner, exercises `/v1/models`, a
non-stream completion and a streaming completion, sends **SIGTERM during a
slow stream**, and asserts clean exit 0, no traceback, the drain note and the
tracked-model unload call — the one lifecycle path only a real process can
show.

The live smoke (`--live`, `test_serve_live`) gains one full agent turn
(create → poll → events → session) so the real-model path covers the agent
loop, not just chat; it is run once at closure with model + result recorded.

### 5. Suite split

`test_ollamarun_serve.py` takes the API-era tests (the runtime contract
guards, turn manager, server, OpenAI surface, agent API, artifacts, fits) and
their helpers (`_collect`, `_ServeClient`, `_start_api`, `_stop_api`, `_http`,
`_api_error`, `_sse_events`, `_raw_exchange`, `_sized_chat_body`,
`_smallest_chat_model`, `test_serve_live`, `_start_agent_api`, `_poll_turn`,
`_turn_request`, `_headless_events`). The shared harness (`_plain`, `_check`,
`_FakeClient`) stays in `test_ollamarun.py` and is imported — no third module,
no fixture extraction beyond what the move needs. The canonical command stays
`python3 test_ollamarun.py` (it runs both files and prints one verdict);
`python3 test_ollamarun_serve.py` also runs its half standalone.

### 6. Pyflakes gate

`test_pyflakes_clean` runs pyflakes as a subprocess over
`ollamarun_lib/*.py`, the `ollamarun` entry script and both test files. The
only tolerated findings are the re-export hubs' two message classes
(`constants.py`: "imported but unused"; `__init__.py`: the star-import
notes); everything else fails the suite, and the five real findings above are
fixed as part of this. Missing pyflakes fails, like the SDK gate.

### 7. Python 3.10 (recorded, not tested)

`/bin/python3.10` is 3.12.3 locally and no fleet host runs `serve`; the
`feature_version=(3, 10)` AST guard remains the enforcement. Recorded here so
the Phase-1b item closes explicitly.

## Delivery order

1. This plan.
2. Split the suite (§5) — mechanical, full suite green before and after.
3. Pyflakes gate + the nit fixes (§6).
4. Required SDK gate (§3).
5. Gap tests (§2).
6. Real-process smoke + live-smoke extension, run and record (§4).
7. Docs (README testing note, AGENTS/CLAUDE test notes), VERSION 1.57 +
   changelog, matrix folded into the closure; move this plan to `done/` and
   flip the index row.

## Acceptance

- `python3 test_ollamarun.py` runs everything (both halves), green, and
  `--live` still adds the real-backend smoke.
- Every Phase-7 requirement has a named test or a recorded reason it cannot
  be automated (the 3.10 item).
- The official `openai` SDK test is required, and its run is confirmed in the
  closure.
- One real-process lifecycle check and one live local-model smoke have been
  completed and recorded.
- The CLI and every product behavior are unchanged; each step lands with the
  suite green.

## Handoffs

- **Phase 8**: this matrix is the "tested compatibility" source for the docs
  pass; the split file and the gates are the documented test command.
- **Phase 9**: `/v1/embeddings`, `n > 1` and `parallel_tool_calls` land in
  `test_ollamarun_serve.py` under the same gates.

## Recorded implementation decisions

- **The split covers more than the contiguous block.** The original file
  defined the Phase-10 API tests (`test_session_context_fields`,
  `test_fit_estimate_api`, `test_fit_job_api`, `test_turn_start_fit`) early,
  next to the ctxfit CLI tests, while they use the serve helpers that live
  with the API tests; they moved to `test_ollamarun_serve.py` too, so every
  API-surface test shares one home. The move is mechanical (no assertion
  changed), and the canonical command is unchanged: `python3
  test_ollamarun.py` imports the serve half and prints one verdict.
- **The shared harness stays in the main file** (`_plain`, `_check`,
  `_FakeClient`, `CTX_SHOW`, `_fit_check`) and the serve half imports it — no
  third module. The canonical entry registers itself under its import name
  (`sys.modules.setdefault`) before importing the serve half, so the 400 KB
  main file is parsed once; `python3 test_ollamarun_serve.py` also runs
  standalone.
- **The pyflakes gate runs in-process** (`pyflakes.api.checkPath` into a
  captured reporter) rather than as a subprocess, and fails — like the SDK
  gate — when the package is missing. The tolerated findings are exactly the
  deliberate re-export sites: `constants.py`'s "imported but unused" hub
  names, **everything** in `__init__.py` (the star-import hub flags its own
  re-exports and its two compatibility names), and `repl._prompt_approval`'s
  "imported but unused". The audit found more hub noise than the plan
  assumed (the `.tui.*`/`.run.*`/... star lines each report both classes),
  which is why `__init__.py` is exempt as a file.
- **The five real findings were fixed**: `ui.py`'s unused `field`/`Callable`,
  `repl.py`'s two placeholder-less f-strings, the dead `iter()` in
  `test_stream_cancel`, the undefined `Any` in the TUI answers annotation
  (plus the missing `typing` import) and two more dead locals that moved into
  the serve half. The Textual probe became
  `importlib.util.find_spec("textual")`; the TUI itself is scheduled for a
  later refactor, so nothing else there was touched. `repl.py`'s
  compatibility re-export stays as production code and is exempted by the
  gate's documented policy instead of being contorted to satisfy a linter.
- **A backend failure is `content: ""`, not `None`.** The new lifecycle
  assertion pins the actual contract: failed status, the backend message on
  `error` and on `turn.error`, and a persisted session with the user turn
  only. A cancelled mid-stream turn is the opposite shape: partial assistant
  text with `cancelled: true`.
- **The real-process smoke uses a stub upstream** (loopback
  `ThreadingHTTPServer` answering `/api/tags`, `/api/ps`, `/api/show`,
  `/api/chat` — streamed and not — and `/api/generate` for the unload), an
  ephemeral `--port 0` and a temp config for workspace/artifact root. The
  bound address is parsed from the banner; SIGTERM lands mid-stream and the
  test proves the drain let the SSE body finish `[DONE]`, the exit is 0 with
  no traceback, the tracked model is unloaded (`keep_alive: 0`) and the port
  is released.
- **The live smoke enables tools only when the chosen model advertises
  them.** Live observation while landing this: Ollama answers a hard 400
  (`<model> does not support tools`) when tools are sent to a model without
  the capability, and the runtime's capability check is warning-only (CLI and
  API alike) — the turn then fails with the sanitized backend error after the
  warning. That is pre-existing, consistent behavior outside this phase's
  scope; it is recorded here rather than changed. The live agent turn
  therefore passes `settings.tools` per capability, and the live tool
  round-trip stays behind `OLLAMARUN_TEST_MODEL`.
- **Python 3.10 stays recorded, not tested**: `/bin/python3.10` is 3.12.3 on
  this box and no fleet host runs `serve`; the `feature_version=(3, 10)` AST
  guard is the enforcement.

## Closure

Closed at v1.57 with the full deterministic suite green through the canonical
command (`python3 test_ollamarun.py`, ~45 s) and the standalone serve half
(`python3 test_ollamarun_serve.py`), including the new gates and tests:
`test_turn_mode_isolation`, the extended `test_agent_turn_lifecycle`,
`test_agent_policy_mode`, `test_turn_buffer_truncation`, `test_serve_http`,
`test_agent_turn_controls`, `test_serve_process`, `test_pyflakes_clean` and
the required `test_openai_sdk` (it printed the exercised SDK version,
2.29.0, under this run).

Recorded live run (2026-10-17, `python3 test_ollamarun_serve.py --live`):
backend `100.102.94.44:11434` (the `OLLAMA_HOST` tailnet endpoint), model
`smollm:135m` (smallest local chat model, no tools capability) — the model
list, a completion with usage, streaming `[DONE]` and one full agent turn
(completed, `tools=false`, session persisted) all passed; the SDK
round-trip and the deterministic groups ran green in the same command. The
`OLLAMARUN_TEST_MODEL`-gated live tool pass-through was not exercised in
this run (no tool-capable model was selected for it).
