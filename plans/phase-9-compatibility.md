# Phase 9 — Compatibility completion

Phase 3 ships a strict OpenAI-compatible core: unsupported requests fail
with structured errors instead of being guessed at. This phase closes the
remaining surface so the API never carries a standing list of rejected
features. Every item is either implemented here or closed with a recorded
rationale — nothing is left as a documented gap.

## Implement

- **`POST /v1/embeddings`** — the OpenAI embeddings schema over
  `OllamaClient.embed()`: `{model, input, encoding_format, dimensions,
  user}` with `input` as a string or a list of strings; response
  `{object: "list", data: [{object: "embedding", index, embedding}],
  model, usage: {prompt_tokens, total_tokens}}`. `/v1/models` already lists
  embedding models; the endpoint validates the model advertises the
  `embedding` capability before accepting, and honors `dimensions` only
  when the backend supports it.
- **`n > 1`** — N upstream generations with choices indexed `0..n-1` and
  usage aggregated across the calls (the accounting is documented, because
  it is an emulation rather than one backend request). Streaming
  multiplexes per-choice deltas by `index` and finishes each choice
  independently before the terminal marker. Non-streaming and streaming
  are both covered.
- **`parallel_tool_calls: false`** — only the first returned tool call is
  forwarded, so a client that asked for sequential execution never
  receives parallel `tool_calls`; `true` or absent passes through
  untouched.
- **Remote `image_url` values (http/https)** — owned by Phase 5 (size cap,
  redirect and SSRF rules); this phase verifies the OpenAI surface uses
  those rules and drops the data-URI-only restriction.

## Close (recorded decisions)

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

## Acceptance

- `/v1/embeddings` round-trips against a real embedding model and matches
  the OpenAI schema under the official SDK.
- `n > 1` returns n choices in both modes; `parallel_tool_calls: false`
  truncates to one call; remote image URLs work through the Phase 5 rules.
- The README's validation section describes only accepted/rejected
  behavior; this phase owns every non-supported item.
