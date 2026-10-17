OllamaRun API Layer — index

Evolve ollamarun from a CLI-oriented agent harness into a reusable agent
backend: an OpenAI-compatible HTTP API plus native endpoints for
agent-specific functionality, all on one agent runtime. The per-phase plans
in `plans/` are the source of truth; this file is only the itemized index.

Design principles (locked)

- OpenAI-compatible at heart. Familiar request/response schemas, endpoints,
  streaming semantics and error formats; the same style for extensions.
- Extend without breaking compatibility. Agent-specific features never
  require nonstandard fields in ordinary OpenAI-compatible requests.
- Reuse the existing runtime. No duplicated agent loop, tool
  implementations, approval logic or session handling in HTTP layers.
- Keep the backend lean. Existing dependencies and the standard library;
  no heavyweight web framework without a concrete need.
- Structured events internally. Responses, tools, approvals, artifacts,
  errors and cancellation have explicit representations.
- Secure by default. The API is LAN-accessible without authentication
  (locked): default `0.0.0.0:11440`, `--bind` to narrow it, trusted-network
  only, never exposed to the public internet. Tool policy is preserved
  exactly — network access is never authorization.

Phase index

| Phase | Status | Plan | Delivers |
|---|---|---|---|
| 1 | done | `plans/done/ollamarunPhase1bPlan.md` (closure table) | Runtime inspection and isolation: `AgentRuntime`, typed event vocabulary, terminal/safety contracts, headless turn |
| 1b | done | `plans/done/ollamarunPhase1bPlan.md` | Rails: runtime-owned tool-support check, live `quiet_tools`, full cancellation coverage, cross-process store lock, `@file` fix, 3.10 syntax guard |
| 2 | done | `plans/done/ollamarunPhase2Plan.md` | Event envelopes (`TurnEvent`) and the turn manager: workers, replay/`stream.gap`, retention, cancellation, `SessionBusy`/`TooManyTurns`/`TurnExpired` |
| 3 | done | `plans/done/phase-3-openai-api.md` | OpenAI-compatible surface: `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/chat/completions` (stream + non-stream), security baseline, `ollamarun serve` |
| 4 | done | `plans/done/phase-4-agent-api.md` | Native agent endpoints: turns, events, cancellation, approvals, sessions, turn status/recovery |
| 5 | done | `plans/done/phase-5-artifacts.md` | File/image input and artifact output with provenance, storage and retrieval |
| 6 | pending | `plans/phase-6-security-operations.md` | Operational hardening: caps, timeouts, path jails, signal handling, `serve_*` policy keys, chunked request decoding |
| 7 | pending | `plans/phase-7-testing.md` | Runtime/lifecycle/HTTP/artifact/compatibility test coverage; OpenAI SDK round-trip |
| 8 | pending | `plans/phase-8-documentation.md` | Compatibility + agent API documentation and example clients |
| 9 | pending | `plans/phase-9-compatibility.md` | Compatibility completion: `/v1/embeddings`, `n>1`, `parallel_tool_calls`, remote image URLs (Phase 5 handoff) and the recorded closures for backend-impossible features |

Locked decisions (cross-phase)

- LAN-accessible without authentication. `0.0.0.0:11440` by default,
  `--bind IP [--port N]` to change it; the API listener stays separate from
  Ollama's `--host`. The service is trusted-network-only and must never be
  exposed directly to the public internet. Authentication, when added later,
  wraps the routes without reshaping them and is never a substitute for tool
  approval.
- Two streaming contracts, one runtime. `/v1/chat/completions` speaks OpenAI
  SSE only; the typed `turn.*` event stream (seq, replay, `stream.gap`)
  belongs to the native agent endpoints and never leaks onto the OpenAI wire.
- Model turns and agent turns stay distinguishable. A model turn is one
  inference call: client `tools` pass through and come back as
  `tool_calls`, never executed. An agent turn runs the internal tool loop.
- Disconnect policy. A model-only completion may cancel upstream generation
  on disconnect; a native agent turn keeps running — a disconnect is an
  unsubscribe and cancellation stays an explicit operation.
- Turns are in-memory; the session JSON is the durable record. A server
  restart destroys live turn records; expired/missing turns are recovered
  from the session store. `GET /v1/agent/turns/{id}` answers 404 (never
  existed) and 410 (deliberately swept) distinctly.
- One active turn per session inside the process; the initial contract is
  single-writer (CLI and API never mutate one persistent session at the same
  time). Cross-process session locking is a later upgrade.
- Approvals are explicit state: every pending approval has its own id tied to
  one turn and one tool operation, the first resolution wins, a timeout
  defaults to denial, and cancellation/shutdown resolve waiting approvals as
  cancelled.

Definition of done

- ollamarun runs as a reusable backend without terminal interaction; the CLI
  keeps using the same runtime.
- Standard OpenAI clients do chat completions (streaming included) against a
  custom base URL; unsupported features fail explicitly and are documented.
- Native endpoints expose full agent turns, sessions, approvals and
  artifacts with the lifecycle properties above.
- Tool policies and security controls stay enforced independently of the
  client; automated compatibility and regression tests cover the critical
  paths; a separate web UI can be built entirely as a client.

Out of scope for this stage: building the web UI, redesigning the reasoning
loop, distributed orchestration, replacing existing tool implementations.
