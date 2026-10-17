# Phase 9 — Compatibility completion

Phase 3 shipped a strict OpenAI-compatible core: unsupported requests fail
with structured errors instead of being guessed at. This phase closes the
remaining surface so the API never carries a standing list of rejected
features. Every item is either implemented here or closed with a recorded
rationale — nothing is left as a documented gap.

Status: implemented and landed at v1.58. The contracts below are what
shipped; the deviations and the recorded closures are under "Recorded
implementation decisions".

## Baseline (audited at v1.57)

- The `/v1` surface is `openai_api.py`: `GET /v1/models`,
  `GET /v1/models/{id}`, `POST /v1/chat/completions`; strict validation with
  exact envelopes; `n` other than the integer 1, `logprobs`/`top_logprobs`,
  `logit_bias`, `audio`, non-text `modalities` and `tool_choice: "required"`
  rejected; `parallel_tool_calls` type-checked and then ignored (the README
  says "until the compatibility phase honors it"); `strict: true` tool
  schemas rejected.
- `OllamaClient.embed(model, inputs, *, truncate, dimensions, options)`
  already wraps `/api/embed` (the CLI's embedding path uses it), and
  `run_embed` maps its result; `/v1/models` already lists embedding models.
- Remote `image_url` values already work: `resolve_content_parts` fetches
  http(s) through `fetch_remote_media` (redirect/timeout/size caps) on both
  surfaces; the README documents it. This phase verifies the OpenAI surface
  end to end and drops any stale data-URI-only wording.
- **The official SDK sends `encoding_format: "base64"` by default** when the
  caller does not specify one (openai-python 2.x) and decodes the base64
  client-side, so the embeddings endpoint must support both `"float"` and
  `"base64"` to round-trip under the SDK.

## 1. `POST /v1/embeddings`

The OpenAI embeddings schema over `/api/embed`; stateless like chat
completions.

- Request: `{model, input, encoding_format, dimensions, user}` — strict field
  set, unknown fields rejected like the chat mapper. `input` is a non-empty
  string or a non-empty list of non-empty strings (OpenAI's token-array
  forms are rejected explicitly with `unsupported_value`). `encoding_format`
  is `"float"` (default) or `"base64"`. `dimensions` is a positive integer,
  forwarded to the backend (the backend decides whether the model supports
  it); `user` is accepted and ignored, matching the chat mapper.
- Response: `{object: "list", data: [{object: "embedding", index,
  embedding}], model, usage: {prompt_tokens, total_tokens}}`. `base64`
  encodes the float32 vector little-endian (`struct.pack("<…f")`), exactly
  what the SDK decodes; usage comes from the backend's `prompt_eval_count`
  (no completion tokens — embeddings do not generate).
- Gates, in order: the `serve_models` allowlist (`404 model_not_found`,
  identical to an unknown model), then the capability check — a model whose
  advertised capabilities lack `embedding` is refused with
  `400 model_does_not_support_embeddings` (an empty capability list passes,
  the same "older backends cannot be verified" rule as vision). The model is
  noted on the serve tracker like a chat completion.
- The inputs ride the shared `InputBudget` (count against
  `serve_max_messages`, per-input bytes against `serve_max_message_bytes` and
  the total against `serve_max_text_bytes`) so embeddings cannot bypass the
  Phase 6 aggregate caps.
- Upstream failures map through the existing `translate_error` /
  `backend_error` path, so no raw backend text reaches the client.

## 2. `n > 1`

Emulated with N sequential upstream generations (`n` is a positive integer,
default 1, capped at `MAX_N = 8` — each choice is a full generation on a
local backend, and the cap keeps one request bounded; the documented bound is
part of the contract).

- Non-streaming: `choices` indexed `0..n-1` with per-choice messages and
  finish reasons; `usage` is the sum across the generations (documented as
  an emulation rather than one backend call).
- Streaming: the role chunk, the deltas, the tool-call chunk and the finish
  chunk of each choice carry that choice's `index`; choices finish one after
  another (sequential emulation) and the terminal `[DONE]` follows all of
  them. `stream_options.include_usage` emits one usage chunk with the
  aggregated counters and empty `choices`.
- A mid-stream upstream failure keeps the existing shape: one error event,
  then `[DONE]`, and the remaining choices are abandoned.

## 3. `parallel_tool_calls: false`

The backend cannot be instructed to emit a single call, so the flag is
honored by truncation: when `parallel_tool_calls` is `false`, only the first
returned tool call is forwarded (non-streaming and streaming alike); `true`
or absent passes every call through untouched. The README says exactly that
(a truncation, not model steering).

## 4. Remote image URLs (verification)

Phase 5 owns the rules; this phase pins the OpenAI surface: an `image_url`
content part with an http(s) URL fetches through `fetch_remote_media` (the
stub-server test), a data URI still works, and a non-http(s) scheme is the
structured `unsupported_value` 400. Any "data-URI only" wording is gone from
the README.

## 5. Recorded closures

These are deliberate contracts, not gaps: the backend cannot provide the
capability, so the endpoint rejects it explicitly and the decision lives
here rather than only in the README.

| Feature | Decision |
|---|---|
| `logprobs` / `top_logprobs` | Unsupported permanently: Ollama exposes no token probabilities; a structured 400 `unsupported_parameter` is the contract. |
| `logit_bias` | Unsupported permanently: no backend equivalent. |
| `audio` / non-text modalities | Unsupported permanently: Ollama has no audio modality; images are the only non-text input. |
| `tool_choice: "required"` | Unsupported permanently: the backend cannot force a call, and a retry loop would burn tokens without a guarantee. The object form (a specific function) remains the supported way to constrain choice. |
| `strict: true` tool schemas | Unsupported permanently: the backend does not enforce tool-argument schemas; `strict: false` is a no-op. `response_format` schemas stay enforced through Ollama's `format` grammar. |

## Tests

| Test | Proves |
|---|---|
| `test_embeddings_api` (new) | The HTTP matrix: string/list input, float and base64 vectors (decoded and compared), usage, `dimensions` forwarding, `user` ignored, unknown fields, empty/bad inputs, bad `encoding_format`/`dimensions`, the capability gate, the `serve_models` gate, empty capabilities pass, upstream failure mapping, the input budget |
| `test_openai_mapping` (extend) | `n` validation (absent/1/2..8 accepted; 0,−1,`True`,`1.0`,`"2"`,9 rejected), `parallel_tool_calls` mapping, embeddings request parsing (unit level) |
| `test_serve_http` (extend) | Non-streaming `n=3` (three choices, three upstream calls, summed usage) and streaming `n=2` (per-choice indexes, finish per choice, one aggregated usage chunk, `[DONE]`); `parallel_tool_calls: false` truncating two returned calls to one in both modes |
| `test_artifact_remote_fetch` (extend) | The OpenAI surface accepts an http(s) `image_url`, fetches it through the stub server and sends the bytes upstream as a data URI |
| `test_openai_sdk` (extend) | The SDK's embeddings round-trip (its default base64 encoding) plus explicit `encoding_format="float"`, and an `n=2` completion |
| `test_serve_live` (extend) | Live embeddings against the first embedding-capable model (skipped with a note when none is installed) |

## Delivery order

1. This plan.
2. Embeddings: mapper, response shaping, handler, route; `test_embeddings_api`.
3. `n > 1` and `parallel_tool_calls`: mapper, response/stream shaping;
   extend `test_openai_mapping` and `test_serve_http`.
4. Remote-image verification and its OpenAI-surface test.
5. SDK and live additions; run the live smoke.
6. README (endpoints, supported fields, validation, closures), AGENTS/CLAUDE,
   VERSION 1.58 + changelog, plan closure; move this plan to `done/` and flip
   the index row.

## Acceptance

- `/v1/embeddings` round-trips against a real embedding model and matches the
  OpenAI schema under the official SDK (which defaults to base64).
- `n > 1` returns n choices in both modes; `parallel_tool_calls: false`
  truncates to one call; remote image URLs work through the Phase 5 rules.
- The README's validation section describes only accepted/rejected
  behavior; this phase owns every non-supported item.
- The full suite (both halves, the pyflakes gate and the required SDK test)
  is green; the CLI and the agent surface are untouched.

## Handoffs

- **Phase 8**: the README sections this phase updates are the ones the docs
  pass finalizes; `examples/openai_client.py` should exercise embeddings and
  `n > 1` once written.

## Recorded implementation decisions

- **The official SDK's base64 default shaped the endpoint.** openai-python
  2.x sends `encoding_format: "base64"` whenever the caller does not specify
  one and decodes it client-side, so the endpoint implements both `"float"`
  and `"base64"` (little-endian float32, `struct.pack("<…f")`); the SDK
  round-trip in `test_openai_sdk` pins both.
- **`MAX_N = 8` is a module constant**, not a `serve_*` config key: it is a
  structural bound of the emulation (each choice is a full sequential
  generation), in the same spirit as `MAX_IMAGE_BYTES`, and it is documented
  in the README. A request above it is `400 unsupported_value`.
- **Generations run sequentially, not concurrently.** Parallel fan-out would
  multiply load on a local backend for a feature whose per-choice sampling is
  stochastic anyway; the documented consequence is latency ≈ n × one
  generation. Streaming therefore emits complete per-choice blocks (role,
  deltas, finish) with the choice's `index`, then one aggregated usage chunk,
  then `[DONE]`.
- **`parallel_tool_calls: false` truncates rather than steers.** The backend
  has no "emit one call" instruction, so the flag keeps the first returned
  call (`_forwarded_tool_calls`, non-streaming and streaming); the README
  says exactly that so the semantics are not oversold.
- **Embeddings gates mirror the vision rules**: an advertised capability list
  without `embedding` is `400 model_does_not_support_embeddings`; an empty
  list passes (older backends cannot be verified); the model is noted on the
  serve tracker (unload skips embedding-capability models by house rule).
- **`dimensions` is forwarded, not emulated.** The backend rejects it for
  models without truncated-vector support; the API does not slice vectors
  itself. `truncate` keeps the client default (true).
- **Embeddings inputs ride the shared `InputBudget`** (count against
  `serve_max_messages`, per-input bytes against `serve_max_message_bytes`,
  the total against `serve_max_text_bytes`), so the Phase 6 aggregate caps
  cannot be bypassed through the new endpoint.
- **The SDK test's rejection case moved to `n = 9`.** The old matrix used
  `n = 2` as the rejected field; it is now the positive emulation case
  (indexes/usage asserted), and the `> 8` bound is the 400 contract. The
  SDK's fake advertises the embedding capability so its embeddings checks
  can run.
- **The remote-image verification is a surface test, not new code.** The
  OpenAI mapper already resolved http(s) `image_url` values through
  `fetch_remote_media`; `test_artifact_remote_fetch` now pins that path
  through the stub server, and the stub's deliberate-timeout write no longer
  prints a broken-pipe traceback.
- **`usage_from` stays** (agent_api's `turn_body` shares it); the new
  `aggregate_usage` sums per-generation stats and is used by both chat
  response shapes.

## Closure

Closed at v1.58 with the full deterministic suite green through the canonical
command (`python3 test_ollamarun.py`, ~47 s), the pyflakes gate and the
required SDK round-trip (openai 2.29.0) included, and these new/extended
regression groups: `test_embeddings_api`, `test_openai_mapping` (n bounds and
the three-state `parallel_tool_calls`), `test_serve_http` (n = 3 non-stream,
n = 2 streaming, streaming truncation), `test_artifact_remote_fetch` (the
OpenAI-surface fetch), `test_openai_sdk` (embeddings in both encodings, n = 2)
and `test_serve_live` (a real embedding model).

Recorded live run (2026-10-17, `python3 test_ollamarun_serve.py --live`):
backend `100.102.94.44:11434`, chat model `smollm:135m`, embeddings
`nomic-embed-text:latest` (768-dim vectors for both inputs, usage counted) —
all green, including the agent turn. `OLLAMARUN_TEST_EMBED_MODEL` overrides
the embedding-model pick and the check prints a note and skips when none is
installed.
