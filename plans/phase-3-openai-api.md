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
