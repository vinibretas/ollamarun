# Errors

Request validation is strict: every rejection is a structured error with the
OpenAI envelope, never a silent ignore. The captures below are verbatim
responses.

## Unknown and unsupported fields

```console
$ … -d '{"model": "llama3.2:3b", "messages": […], "bogus": 1}'
{"error": {"message": "unknown field: bogus", "type": "invalid_request_error",
 "param": "bogus", "code": "unknown_field"}}                       # HTTP 400
```

Unknown fields are rejected everywhere they can appear — top level, inside
tools, `tool_choice`, `response_format` wrappers.

```console
$ … -d '{…, "n": 9}'
{"error": {"message": "n is limited to 8 emulated generations", "type":
 "invalid_request_error", "param": "n", "code": "unsupported_value"}}

$ … -d '{…, "logprobs": true}'
{"error": {"message": "logprobs is not supported", "type":
 "invalid_request_error", "param": "logprobs", "code": "unsupported_parameter"}}

$ … -d '{…, "tool_choice": "required"}'
{"error": {"message": "tool_choice must be \"auto\", \"none\" or a specific function
 (the backend cannot force a call)", "type": "invalid_request_error",
 "param": "tool_choice", "code": "unsupported_value"}}

$ … -d '{…, "tools": [{…, "strict": true}]}'
{"error": {"message": "strict tool schemas are not supported (the backend does not
 enforce tool-argument schemas)", "type": "invalid_request_error",
 "param": "tools[0].function.strict", "code": "unsupported_parameter"}}
```

Also rejected: `logit_bias`, `audio`, non-text `modalities`, malformed
message shapes, and non-integer `n`.

## Unknown models and upstream failures

```json
{"error": {"message": "model 'nope:latest' not found",
 "type": "invalid_request_error", "param": "model", "code": "model_not_found"}}
```

`model_not_found` (404) is also what a model outside the
[allowlist](../security.md#model-allowlist) answers. The server distinguishes:

| Code | Means |
| --- | --- |
| `model_not_found` | Unknown model or outside the allowlist. |
| `backend_unreachable` | The Ollama machine could not be reached (502). |
| `backend_error` | The backend failed the request (502), sanitized — raw detail goes to the `--debug` log. |
| `model_does_not_support_images` / `…embeddings` | Capability gates. |
| `input_too_large` | A resolved-input limit (413). |
| `not_found` | Unknown endpoint (404) — including agent routes when `serve_agent: false`. |

## What is never in a response

Status codes are used properly (400/404/405/408/413/502), and responses
never carry tracebacks or internal paths. Oversized bodies and stalled
uploads are rejected before parsing; malformed chunked framing is a
structured 400. `--debug` logs requests, upstream detail and tracebacks to
`~/etc/logs/ollamarun.log` for the operator — not the client.

The same envelope shape is used by the [agent API](../agent/turns.md); its
typed errors (`session_busy` 409, `rate_limit_error` 429, expired turns 410)
are listed on [Turns](../agent/turns.md#records-and-limits).
