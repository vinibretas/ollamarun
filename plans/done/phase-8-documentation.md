# Phase 8 — Documentation and examples

The closing phase: the API-layer docs are already current (every phase
updated its sections), so this lands the missing example clients and the
last README gaps — deployment and the deferred auth hooks — then closes the
plan index.

Status: implemented and landed at v1.59; the phase index is all-done.

## Baseline (audited at v1.58)

- README already carries the full "OpenAI-compatible API" and "Agent API"
  sections (endpoints, fields, base URL, SDK snippet, streaming/tool
  behavior, limits, allowlist, artifacts, context control, the recorded
  compatibility closures and the `serve_*` config matrix).
- AGENTS/CLAUDE carry the architecture entries, the `serve` command, the
  config keys and the test notes; they are identical.
- `__init__.py` re-exports every API module and the `build_parser()` epilog
  mentions `serve`.
- Missing: the two example clients, an examples pointer in the README, the
  deployment/deferred-auth note in the Agent API section, and a regression
  that keeps the examples honest.

## 1. `examples/openai_client.py`

A reference OpenAI client that doubles as an integration check: it uses the
official SDK against `ollamarun serve` and prints a PASS/SKIP/FAIL line per
step, exiting non-zero on a failure.

- `--base-url` (default `http://127.0.0.1:11440/v1`), `--api-key` (default
  `unused`), `--model` (default: the first listed), `--timeout`.
- Steps: the model list; a non-streaming completion (content + usage); a
  streaming completion (deltas + `stream_options.include_usage`); embeddings
  through the SDK's default base64 encoding; `n = 2` (two indexed choices);
  and a `tools` request reported as a pass either way (a tool-capable model
  answers with `tool_calls`; others get an informative note, and the backend
  rejecting tools outright prints SKIP with the reason).
- The official SDK is a test/maintainer dependency, not a runtime one: a
  missing `openai` import prints the install hint and exits 2.

## 2. `examples/agent_client.py`

A reference native-agent client over `requests` (the repo's HTTP dependency)
and `rich`: it submits a turn, follows the typed SSE stream and resolves
approvals.

- `--base-url` (default `http://127.0.0.1:11440`), `--model`, `--prompt`,
  `--session-id` (continue an existing session), `--mode policy|interactive`
  (maps to `settings.approval`), `--approve run|remember|rule|auto|deny`
  (what to send for a parked approval in interactive mode), `--json` (raw
  event lines instead of the pretty transcript), `--timeout`.
- The stream loop prints `message.delta` content (and thinking dimmed),
  tool calls/results, warnings and approvals; on `approval.required` in
  interactive mode it POSTs the chosen decision; it stops at the terminal
  event, re-reads the status snapshot and prints the summary (status,
  rounds, usage, final content), exiting 0 only for `completed`.

## 3. README

- An **Example clients** paragraph under the API sections: what each script
  does, how to run it, and that they double as integration checks (a note
  pointing at `examples/`).
- The Agent API section gains **Deployment and auth**: run `serve` under the
  user's service manager on the trusted network (the CLI's own guidance
  stays the reference); authentication is deferred by design — when it
  lands it wraps the existing routes without reshaping them, and it is
  never a substitute for tool approval (the locked contract).

## 4. AGENTS.md / CLAUDE.md (identical)

- Layout/commands: the `examples/` directory and how the two clients run.
- Testing notes: `test_examples` runs both clients against the fake stack.

## 5. Regression: `test_examples`

Runs both scripts as subprocesses against the in-process fake backend
(`_start_api` for the OpenAI client, `_start_agent_api` for the agent
client) and asserts exit 0 plus the expected markers — so the reference
clients can never rot silently.

## Delivery order

1. This plan.
2. The two example clients.
3. `test_examples` + the runner wiring.
4. README (examples + deployment/auth), AGENTS/CLAUDE notes.
5. VERSION 1.59 + changelog; plan closure; move to `done/`; flip the index
   row (the phase index is then all-done).

## Acceptance

- Both examples run against a real `ollamarun serve` (documented commands)
  and against the fake stack in the suite.
- The README covers deployment and the deferred auth contract; every API
  surface has a documented example path.
- The full suite (both halves, the gates, the SDK test) stays green; no
  product behavior changes.

## Recorded implementation decisions

- **The examples degrade honestly.** The OpenAI client treats an
  embedding-only model's embeddings refusal and a model without tool
  support as SKIP with the reason (the SDK example can then run against any
  chat model); everything else is PASS/FAIL. The agent client prints
  information, not verdicts, for a model that answers without a tool call.
- **`test_examples` runs the scripts as subprocesses** against the in-process
  fakes (`_start_api` for the SDK client, `_start_agent_api` for the agent
  client) and asserts exit 0 plus the step markers — the reference clients
  are part of the acceptance gate, not just docs. The pyflakes gate's target
  list now includes `examples/*.py`.
- **The agent client uses `requests` + `rich`** (the repo's own runtime
  dependencies) and parses the SSE `data:` envelopes directly, including
  `?after=0` replay and the `approval.required` → `POST
  /v1/agent/approvals/{id}` round; `--json` prints the raw envelopes for
  debugging a stream.
- **The README's deployment/auth wording restates the locked decision**: the
  bind address is the boundary, no auth is deliberate, and future
  authentication wraps the routes without reshaping them and never replaces
  tool approval.

## Closure

Closed at v1.59 with the full deterministic suite green through the
canonical command (`python3 test_ollamarun.py`, ~48 s) and the new
`test_examples` group covering both clients. Live run recorded (2026-10-17):
a real `ollamarun serve` (temp config, ephemeral port, upstream
`100.102.94.44:11434`) with `qwen2.5:0.5b` — `examples/openai_client.py`
green (model list, completion, streaming, `n = 2`, the tools request; the
model actually asked for `get_weather`; embeddings SKIP as a chat model),
`examples/agent_client.py` green with a real `run_shell echo phase8` round
(status completed, rounds 1) and the server exiting 0 on SIGTERM. With this
phase the API-layer index has no pending rows.
