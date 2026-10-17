OllamaRun API Layer — Implementation Plan

Objective

Evolve ollamarun from a CLI-oriented agent harness into a reusable agent backend with an OpenAI-compatible HTTP API and native extensions for agent-specific functionality.

The API will serve as the foundation for future clients, including the planned web UI and other lightweight tools in the local computing fleet.

The goal is one agent runtime, multiple interfaces. The CLI, HTTP API, and future web UI must share the same underlying inference, tool execution, session management, and security logic.

Design Principles

1. OpenAI-compatible at heart. Implement familiar request/response schemas, endpoint conventions, streaming semantics, and error formats wherever applicable. Follow the established API style even for custom extensions.
2. Extend without breaking compatibility. Agent-specific features must not require nonstandard fields in ordinary OpenAI-compatible requests.
3. Reuse the existing runtime. Do not duplicate the agent loop, tool implementations, approval logic, or session handling in the HTTP layer.
4. Keep the backend lean. Prefer the existing dependencies and standard-library facilities when practical. Do not introduce a heavyweight web framework or orchestration system without a concrete need.
5. Structured events internally. Responses, tool activity, approval requests, artifacts, errors, and cancellation must have explicit representations.
6. Secure by default. HTTP access must preserve the CLI’s existing tool permissions and approval policies. Binding the API to localhost is the default.

Phase 1 — Inspect and Isolate the Runtime

Before implementing HTTP endpoints, identify and document the boundaries between the CLI and the agent runtime.

Tasks

* Map the existing execution flow from user input through inference, tool calls, approval checks, session updates, and final response.
* Identify dependencies on terminal input/output, prompt-toolkit, Textual, and other CLI-specific components.
* Extract or formalize a reusable runtime entry point that accepts a structured request and produces a structured result.
* Ensure session state, configuration, cancellation, and policy enforcement can operate without an interactive terminal.
* Preserve existing CLI behavior.

Avoid a large refactor of unrelated components. Extract only the boundaries required to support another interface.

Acceptance criteria

* The existing CLI continues to work.
* The agent can execute a turn without requiring a terminal-specific interaction.
* Core agent logic can be invoked programmatically.
* Approval-required operations cannot silently bypass their existing safeguards.

Phase 2 — Establish the Internal API and Event Model

Define the internal representation before mapping it to HTTP.

Core objects

* Request: messages, model selection, generation parameters, optional session identifier, and execution settings.
* Message: role, content parts, tool calls, and tool results.
* Agent event: typed event with structured payload and associated turn/session identifiers.
* Artifact: identifier, media type, filename, storage reference, and metadata.
* Turn result: final response, execution status, usage data when available, and relevant artifact references.

Event types

* turn.started
* message.delta
* tool.call
* tool.result
* approval.required
* artifact.created
* turn.error
* turn.completed
* turn.cancelled

These are proposed internal event names, not claims of OpenAI-standard event compatibility.

Events must have deterministic serialization and a well-defined lifecycle. Text deltas must remain distinguishable from status updates, tool calls, and artifact notifications.

Acceptance criteria

* The CLI and HTTP adapter can consume the same runtime events.
* Event payloads do not depend on terminal formatting.
* Model-generated content is kept separate from runtime metadata.
* The event model supports future multimodal output without changing the core execution model.

Phase 3 — Implement the OpenAI-Compatible HTTP Surface

Implement the smallest useful compatible API first.

Initial endpoints

Endpoint	Purpose
GET /v1/models	List available or configured models
POST /v1/chat/completions	Run a model turn using familiar OpenAI-compatible schemas

Security baseline

The API executes tools and modifies files, so the baseline ships
with the first HTTP phase rather than at hardening time:

* LAN access without authentication is the locked contract: the
  default listener is 0.0.0.0:11440 (all interfaces), configurable
  with --bind, and the API listener stays separate from Ollama's
  --host. The service is trusted-network-only and must never be
  exposed directly to the public internet; the startup banner and
  the docs say so.
* Reject oversized request bodies before parsing them.
* Apply request and upstream timeouts, and propagate cancellation
  to inference when the client cancels or disconnects.
* Return sanitized, structured errors: no internal tracebacks, no
  filesystem paths.
* Preserve tool policies exactly: network access never changes the
  configured execution policy, and authentication (when it is added
  later) is never a substitute for tool approval.

Support the common request fields used by existing clients, with explicit handling of unsupported parameters rather than silently ignoring them.

For non-streaming requests, return a compatible completion structure. For streaming requests, use Server-Sent Events (SSE) and emit the expected chunk structure and termination marker.

Compatibility should be tested against actual SDK behavior, not just against hand-written examples.

Tool calling

Support standard tool-call request and response structures, including arguments encoded as JSON.

Maintain the distinction between:

* A model turn, where the backend returns the model’s response or proposed tool calls.
* An agent turn, where ollamarun executes its internal tool loop and returns the resulting response.

Do not make these two execution modes indistinguishable. A client using the standard completion endpoint should not unexpectedly receive privileged agent execution merely because it submitted tool definitions.

Two streaming contracts, one runtime

The compatible endpoints speak OpenAI SSE only: role and content
deltas, finish reasons and the terminal marker. The typed event
stream (`turn.started`, `tool.result`, `turn.completed`, …) with
per-turn sequence numbers, replay and gap notifications belongs to
the native agent endpoints (Phase 4) and must not leak into the
OpenAI wire format.

Disconnect policy differs by contract:

* A model-only completion has no recoverable job behind it, so
  cancelling upstream generation on client disconnect is
  acceptable; document and test it.
* A native agent turn keeps running by default. A disconnect is an
  unsubscribe; cancellation stays an explicit operation.

Acceptance criteria

* Standard clients can discover models and submit chat-completion requests.
* Streaming and non-streaming responses behave consistently.
* Tool calls follow the expected message structure.
* Errors use structured responses with appropriate HTTP status codes.
* Unsupported features are documented and handled explicitly.
* The OpenAI SSE contract and the native agent event protocol are kept
  separate; no native envelopes appear in chat-completion streams.
* The security baseline above is in place.

Phase 4 — Add Native Agent Endpoints

Expose capabilities that go beyond ordinary inference without distorting the compatible endpoints.

Proposed endpoints:

Endpoint	Purpose
POST /v1/agent/turns	Start a complete agent turn, including the internal tool loop
GET /v1/agent/turns/{id}	Retrieve one turn's snapshot: status, session, timestamps, sequence, result, error
GET /v1/agent/turns/{id}/events	Subscribe to turn events over SSE
POST /v1/agent/turns/{id}/cancel	Request cancellation
GET /v1/sessions	List persistent sessions
GET /v1/sessions/{id}	Retrieve session metadata and history
DELETE /v1/sessions/{id}	Delete a session, subject to persistence semantics
POST /v1/sessions	Create a session
POST /v1/agent/approvals/{id}	Approve or reject a pending tool operation

Adjust routes to match the existing session model where appropriate. Avoid adding endpoints that duplicate existing functionality without a clear interface benefit.

Agent events should expose tool progress, approval requirements, execution errors, cancellation, and completion in a structured format. The backend must not depend on a client remaining connected for an agent turn to remain coherent unless that behavior is explicitly designed and documented.

Turn status and recovery

`GET /v1/agent/turns/{id}` is the source of truth whenever a client is
not (or no longer) subscribed:

* 200 with the snapshot: lifecycle status, session, timestamps, latest
  event sequence, terminal result and error when available.
* 404 for a turn that never existed.
* 410 (expired) for a turn whose retained record was deliberately swept,
  distinct from 404; session history remains the durable record.
* After a `stream.gap`, clients re-read this endpoint instead of
  inferring state from incomplete events.

Approval lifecycle

* Every pending approval has its own ID, tied to one turn and one exact
  tool operation.
* A turn enters `waiting_approval` only when a decision is genuinely
  pending; the runtime's approver callback parks on that decision.
* Approval and rejection are explicit transitions. A duplicate, expired
  or already-resolved decision must not execute the operation a second
  time (the first resolution wins; later posts are rejected).
* A timeout defaults to denial, never implicit approval.
* The worker waits for a decision without holding the manager's
  registry lock.
* Pending approvals count toward the global active-turn limit, like any
  running turn.
* Cancellation and shutdown while waiting resolve the approval as
  cancelled and stop the worker.

Durability

Turn records and event buffers are in-memory: a server restart destroys
a live turn record even though session history persists. Clients must
not treat turns as durable jobs; an expired or missing turn is recovered
from the session store.

Session exclusivity

The turn manager enforces one active turn per session inside its own
process. It does not prevent a CLI process from concurrently modifying
the same persistent session, so the initial contract is single-writer:
a persistent session must not be actively modified by the CLI and the
API at the same time. Shared cross-process session locking is a later
upgrade; distributed coordination is out of scope.

Disconnect policy

A native agent turn keeps running when its event subscriber
disconnects; cancellation remains an explicit request. Clients recover
through `GET /v1/agent/turns/{id}` and event replay.

Acceptance criteria

* Clients can run full agent turns without implementing their own tool loop.
* Clients can observe progress and distinguish execution from generation.
* Pending approvals have explicit states and cannot be confused with successful execution.
* Disconnected clients do not leave uncontrolled or orphaned execution behind.
* The turn status endpoint distinguishes retained, expired and unknown
  turns, and supports recovery after a replay gap.
* Approval timeout, duplicate decisions, cancellation while waiting and
  shutdown while waiting are covered by tests.
* A disconnected subscriber leaves the turn running and an explicit
  cancel stops it.

Phase 5 — Add File, Image, and Artifact Handling

Design multimodal support around distinct input and output paths rather than a generic binary blob.

Input

* Text messages.
* Image inputs through compatible content parts where supported.
* File attachments through a documented upload or reference mechanism.
* Validated artifact references from previous operations.

Output

* Text and Markdown.
* Images returned or referenced by models.
* Images generated by executing code.
* Other files created by tools.
* Structured metadata describing each artifact.

Model-generated images and images created by scripts are different provenance classes. Preserve that distinction in artifact metadata and event types.

Use artifact identifiers and metadata in agent responses rather than embedding arbitrary binary data in every message. Choose storage locations, retention, size limits, and cleanup policies explicitly.

Do not imply that a model supports image input or generation just because the HTTP API accepts the corresponding request structure. Validate capabilities against the selected backend model.

Acceptance criteria

* Clients can send supported image inputs and access returned artifacts.
* Generated files and images have stable references and media types.
* File paths, sizes, formats, and access permissions are validated.
* Artifact retrieval does not expose arbitrary filesystem paths.

Phase 6 — Security and Operational Controls

Treat the API as a remote control surface for an agent capable of executing code and modifying files. The service is LAN-accessible without authentication (locked): the bind address and the network are the boundary, and the documentation must state trusted-network-only.

Tasks

* Keep the default listener LAN-accessible (0.0.0.0:11440) with --bind
  to narrow or change it; keep the API listener configuration separate
  from Ollama's --host.
* Do not implement authentication in this stage; keep the route
  structure and config keys easy to wrap with it later.
* Preserve tool-level permissions and approval policies; unauthenticated
  access must not silently change the configured tool execution policy.
* Validate request sizes, message lengths, file sizes, and parameter ranges.
* Add request timeouts, cancellation propagation, and controlled concurrency.
* Prevent clients from accessing arbitrary files through uploaded paths or artifact identifiers.
* Keep message content and sensitive runtime data out of logs by default.
* Return structured, sanitized errors.

Authentication, when it is added later, is not a replacement for tool
approval: a valid API client must not automatically gain permission to
execute every available tool without the intended policy checks.

Acceptance criteria

* The API is LAN-accessible by default and documented as trusted-network-only.
* Existing tool safeguards remain effective over HTTP unchanged.
* Invalid or oversized requests fail safely.
* Cancellation and failures leave sessions and runtime resources in a consistent state.

Phase 7 — Testing and Compatibility Validation

Build tests around the runtime and the API boundary.

Runtime tests

* Normal inference.
* Multi-step tool execution.
* Tool failures and model failures.
* Approval-required operations.
* Session persistence and recovery.
* Cancellation and interrupted streams.

Lifecycle tests

* Turn state transitions end to end (created → running →
  waiting_approval → completed/failed/cancelled), not only individual
  methods.
* Replay, `stream.gap` recovery and the turn status endpoint after a gap.
* Client disconnect and reconnect policies differ for model-only and
  native agent turns, and both are exercised.
* Approval timeout, duplicate resolution and shutdown while waiting.

HTTP tests

* Request and response schemas.
* Streaming chunk format and stream termination.
* Tool-call serialization.
* Error status codes and bodies.
* No-auth behavior in trusted-network mode (loopback + LAN bind;
  requests need no credentials).
* File and artifact access controls.
* Concurrent requests and session isolation.

Compatibility tests

Use a standard OpenAI SDK against the API with a custom base URL. Confirm model discovery, ordinary chat completions, streaming, and tool-call responses with representative clients.

Record unsupported SDK features and edge cases rather than claiming full compatibility prematurely.

Acceptance criteria

* Automated tests cover the critical execution and security paths.
* The compatible API works with at least one existing OpenAI SDK client.
* The CLI continues to pass its existing tests.
* Streaming behavior is verified with both normal completion and failures.

Phase 8 — Documentation and Client Integration

Document the API as two related surfaces.

Compatibility documentation

* Supported endpoints and fields.
* Base URL and model identifiers.
* SDK configuration examples.
* Streaming and tool-calling behavior.
* Explicit list of compatibility gaps.

Agent API documentation

* Full-turn execution.
* Event schemas and lifecycle.
* Sessions and persistence.
* Approval workflows.
* Attachments and artifact retrieval.
* Configuration and deployment (the trusted-network boundary;
  authentication is deferred, with its hooks documented).

Add a minimal example client that demonstrates standard completions and a second example that executes an agent turn. These become practical integration tests and reference implementations.

Recommended Implementation Order

1. Inspect the existing runtime and isolate the programmatic entry point.
2. Define internal request, result, event, and artifact schemas.
3. Implement GET /v1/models.
4. Implement non-streaming POST /v1/chat/completions.
5. Add SSE streaming and cancellation behavior.
6. Add standard tool-call serialization.
7. Implement complete agent turns and approval events.
8. Add session-management endpoints.
9. Add file, image, and artifact handling.
10. Harden access controls and operational behavior.
11. Test against real OpenAI-compatible SDK clients.
12. Document the API and prepare for the web UI.

This sequence is intentionally incremental. Every stage should leave the existing CLI usable and the API in a testable state.

Definition of Done

The API stage is complete when:

* ollamarun runs as a reusable backend without requiring terminal interaction.
* Existing CLI functionality continues to use the same agent runtime.
* Standard OpenAI-compatible clients can perform ordinary chat completions, including streaming.
* Native endpoints expose full agent execution, sessions, approvals, and artifacts.
* Tool policies and security controls remain enforced independently of the client.
* The API has automated compatibility and regression tests.
* A separate web UI can be implemented entirely as a client of this backend, without reimplementing the agent loop.

Out of scope for this stage: building the complete web UI, redesigning the agent’s reasoning loop, introducing distributed orchestration, or replacing existing tool implementations. The deliverable is a stable, reusable API foundation on which those features can be built.
