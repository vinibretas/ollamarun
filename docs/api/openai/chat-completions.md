# Chat completions

`POST /v1/chat/completions` is one model turn, streaming or not. It is a
translation proxy: one request becomes one `/api/chat` call upstream, and
client `tools` pass through without ever being executed (that is the
[native agent API](../agent/turns.md)'s job).

```console
$ curl -s http://127.0.0.1:11440/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "llama3.2:3b",
         "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
         "temperature": 0}'
```

```json
{
  "id": "chatcmpl-192eef6fe6594a949b5a814a",
  "object": "chat.completion",
  "created": 1792340145,
  "model": "llama3.2:3b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "pong"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 30, "completion_tokens": 2, "total_tokens": 32}
}
```

## Supported fields

| Field | Notes |
| --- | --- |
| `model` | Required; must be installed and inside the allowlist. |
| `messages` | OpenAI roles; assistant `tool_calls` and `role: tool` history convert back to the Ollama shape. |
| `temperature`, `top_p`, `stop`, `seed`, `presence_penalty`, `frequency_penalty` | Forwarded as Ollama options. |
| `max_tokens` / `max_completion_tokens` | → `num_predict` (either name works). |
| `response_format` | `{"type":"json_object"}` or `json_schema` → Ollama's `format` grammar (only the `schema` object is forwarded). |
| `reasoning_effort` | `low`/`medium`/`high` → Ollama `think`; thinking comes back as `reasoning_content` (see [Streaming](streaming.md)). |
| `tools`, `tool_choice` | Pass-through; `auto`, `none` or a specific function. `required` is rejected — see [Tool calls](tools.md). |
| `stream`, `stream_options.include_usage` | See [Streaming](streaming.md). |
| `n` | 1..8, emulated sequentially — see [n > 1 choices](n-choices.md). |
| `parallel_tool_calls` | `false` truncates the returned calls to the first (the backend cannot be told). |
| `user`, `metadata` | Accepted and ignored (tracking hints). |
| `system` *(extension)* | `"auto"` = the built-in default prompt, text = replace it; absent = none. A client-sent system message always wins. |

Everything else — unknown fields anywhere, `logprobs`, `logit_bias`,
`audio`, non-text modalities — is a structured `400`, never a silent
ignore. The full matrix is in [Errors](errors.md).

## Python

The official SDK works unmodified:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11440/v1", api_key="unused")
reply = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "hello"}],
)
print(reply.choices[0].message.content)
```

- [Streaming](streaming.md) — the SSE frame sequence.
- [Tool calls](tools.md) — pass-through semantics and `parallel_tool_calls`.
- [Vision](vision.md) — `image_url` and `file` parts.
- [Errors](errors.md) — every rejection and its envelope.
