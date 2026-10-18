# Chat completions

`POST /v1/chat/completions` is one model turn, streaming (`stream: true`) or
not. Model turns are a translation proxy: one request is one `/api/chat` call
(an `n > 1` request is emulated with that many sequential generations).

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11440/v1", api_key="unused")
print(client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "hello"}]).choices[0].message.content)
```

Supported fields: `temperature`, `top_p`, `max_tokens` /
`max_completion_tokens` (→ `num_predict`), `stop`, `seed`,
`presence_penalty`, `frequency_penalty`, `response_format` (`json_object` /
`json_schema` → Ollama `format`), `reasoning_effort` (low/medium/high →
Ollama `think`), `tools`, `tool_choice` (`auto`, `none`, or a specific
function), `stream`, `stream_options.include_usage`, and the extension
`system` (`"auto"` = the built-in default prompt, text = replace it, absent
= none; a client-sent system message always wins). `user` and `metadata`
are accepted and ignored (tracking hints).

- [Streaming](streaming.md) — the SSE chunk shapes.
- [Tool calls](tools.md) — pass-through and `parallel_tool_calls`.
- [n > 1 choices](n-choices.md) — emulated fan-out.
- [Vision](vision.md) — `image_url` parts.
- [Errors](errors.md) — strict validation and the error envelope.
