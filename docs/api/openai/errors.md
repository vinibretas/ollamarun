# Errors

Request validation is strict, and every rejection is a structured error
rather than a silent ignore: unknown fields anywhere (top level, tools,
`tool_choice`, `response_format` wrappers), `n` outside 1..8 or non-integer
(each emulated choice is a full generation), `logprobs`/`top_logprobs`,
`logit_bias`, `audio`, non-text `modalities` and `tool_choice: "required"` are
rejected with `400`s; `strict: true` on a function tool is rejected because
the backend does not enforce tool-argument schemas (`strict: false` is a
no-op).

`json_schema` enforcement uses Ollama's grammar (`format`); only the `schema`
object is forwarded.

Errors use the OpenAI envelope with proper status codes (400/404/405/408/413/502);
upstream failures surface as stable `model_not_found` / `backend_unreachable`
/ `backend_error` messages, oversized bodies and stalled uploads are rejected
before parsing, and responses never carry tracebacks or internal paths —
`--debug` logs requests, upstream detail and tracebacks to
`~/etc/logs/ollamarun.log`.

Everything the backend cannot do is rejected permanently:

- `logprobs`/`top_logprobs` — Ollama exposes no token probabilities.
- `logit_bias` — no backend equivalent.
- `audio` and non-text modalities — Ollama has no audio modality.
- `tool_choice: "required"` — the backend cannot force a call.
- `strict: true` tool schemas — tool-argument schemas are not enforced
  (`response_format` schemas are, through Ollama's `format` grammar).
