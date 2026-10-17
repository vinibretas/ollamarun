# Phase 3 — OpenAI-compatible HTTP surface

Only what you listed, implemented honestly.

- The model-turn/agent-turn distinction is explicit: submitting `tools`
  here never triggers privileged agent execution. Full agent turns live
  only under `/v1/agent/turns`.

**Endpoints:** `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/chat/completions` (stream + non-stream). Add `POST /v1/embeddings` only if it falls out cheaply; it’s not required.

**Behavior:**

- Runs `mode="model"` through the runtime. One inference call. No harness tools, no approvals, no session writes.
- Client `tools` are passed to Ollama and returned as standard `tool_calls`; `finish_reason: "tool_calls"`. This is the “model turn” your plan distinguishes.
- Stateless by default; clients resend history. No implicit session creation.
- Map `temperature`, `top_p`, `max_tokens`/`max_completion_tokens`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `response_format`, `tools`, `tool_choice`, `stream_options.include_usage`.
- Thinking deltas in non-standard `delta.reasoning_content`; normal clients ignore it.
- SSE: role chunk, content deltas, `finish_reason`, usage chunk when requested, `data: [DONE]`; periodic `: ping`; chunked encoding; flush per event.
- Errors in the OpenAI envelope with correct status codes. **Explicitly reject** `n > 1`, `logprobs`, `logit_bias`, `audio`, non-text modalities, and unknown unsupported fields rather than silently ignoring them. Document `response_format` schema support as Ollama-dependent.
- Validate vision capability before accepting image parts; return `model_does_not_support_images` if absent.

**Security baseline (ships with this phase):**

- LAN access without authentication is the locked contract: default
  listener `0.0.0.0:11440` (all interfaces), `--bind` to narrow or
  change it; the API listener stays separate from Ollama’s `--host`.
  The service is trusted-network-only and must never be exposed
  directly to the public internet; the startup banner and the docs
  say so.
- Cap request bodies before parsing; request and upstream timeouts
  with cancellation propagated to inference on cancel/disconnect;
  sanitized structured errors (no tracebacks, no filesystem paths);
  tool policies preserved exactly — network access never changes the
  configured execution policy.

**Two streaming contracts:** these endpoints speak OpenAI SSE only
(role and content deltas, finish reason, terminal marker). The typed
turn stream (`turn.started`, `tool.result`, …) with per-turn seq,
replay and gap notices belongs to the native agent endpoints and
never leaks into chat-completion streams. Disconnect policy: a
model-only completion may cancel upstream generation on disconnect;
a native agent turn keeps running — a disconnect is an unsubscribe
and cancellation stays explicit.

**Acceptance:** official OpenAI SDK works against a custom base URL for models, chat, streaming, and tool-call responses; hand-written `curl` matches; unsupported features fail explicitly.

## Recorded implementation decisions

- **Model turns live in the server, not in `AgentRuntime`.** The OpenAI layer
  builds the Ollama payload and drives one `client.chat()`/`chat_once()`
  call; the runtime stays agent-turn-only. This closes Phase 1b item 12 by
  deviation: a stateless model turn has no session, tool loop, approvals or
  subscribers, and passthrough history (assistant `tool_calls`,
  `role: tool` + `tool_call_id`) must not go through the harness message
  pipeline (`Session.message_payloads` filters tool history by harness
  rules). `TurnSettings.mode`/`TurnRequest.mode` were removed as dead
  fields (and `mode` dropped from `turn.started`). Chunk accumulation is
  shared (`renderer.consume_chunk`/`stats_from_final`), so no inference
  logic is duplicated.
- **Request extension: top-level `system`.** Absent or `""` sends no system
  prompt (the caller's messages pass through untouched); `"auto"` injects
  the current built-in default (`default_system_prompt(with_tools=False)`,
  since model turns have no harness tools); any other text becomes the
  leading system message, and a client-sent system message still wins —
  the same precedence `build_payload` gives the CLI.
- **Modules and command.** `ollamarun_lib/server.py` (generic HTTP
  plumbing: router, request/response model, body caps, sanitized error
  envelopes, SSE writer with `: ping` heartbeat) and
  `ollamarun_lib/openai_api.py` (the `/v1` mapping). `ollamarun serve
  --bind 0.0.0.0 --port 11440 [--host HOST|--remote NAME]` is dispatched
  from `cli.main()`; `--host`/`--remote` name the upstream Ollama while
  `--bind`/`--port` name the API listener. Config keys `serve_bind` /
  `serve_port` (defaults `0.0.0.0` / `11440`).
- **Per-request `OllamaClient`.** `abort_stream()` has one active-stream
  slot, so each HTTP request gets its own client instance; a disconnecting
  SSE client cancels only its own upstream generation. Model-list and
  capability lookups use short-TTL caches (5s / 60s) on the API object.
- **Vision.** Image content parts must be `data:` URIs in this phase
  (remote URLs land in Phase 5 with the size cap and SSRF note). Before
  accepting images the server checks the model's `capabilities` and answers
  `model_does_not_support_images` when `vision` is absent (an empty/missing
  capability list passes through — older backends cannot be verified).
- **`tool_choice`.** `"auto"` passes tools, `"none"` drops them, the object
  form passes only the named function (Ollama cannot force a call);
  `"required"` is rejected explicitly as unsupported. `reasoning_effort`
  maps low/medium/high onto Ollama's `think`; anything else is rejected.
- **`/v1/embeddings`** is not implemented (the phase marked it optional).
