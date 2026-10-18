# Tool calls

Client `tools` are passed to the backend and returned as standard
`tool_calls` with `finish_reason: "tool_calls"`. They are **never executed
here** — this endpoint cannot trigger the agent harness, sandboxes or
approvals; full agent turns live under the native
[`/v1/agent/turns`](../agent/turns.md) endpoint.

A real request/response (llama3.2:3b asked for the weather with one function
declared):

```console
$ curl -s http://127.0.0.1:11440/v1/chat/completions \
    -H 'Content-Type: application/json' -d '{
      "model": "llama3.2:3b",
      "messages": [{"role": "user", "content": "What is the weather in Porto Alegre? Use the tool."}],
      "tools": [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}],
      "temperature": 0}'
```

```json
{
  "id": "chatcmpl-2ecb878a39e04dc3aa0d3d31",
  "object": "chat.completion",
  "model": "llama3.2:3b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_elpr4nhe",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Porto Alegre\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {"prompt_tokens": 163, "completion_tokens": 17, "total_tokens": 180}
}
```

Note `arguments` is a JSON **string** (OpenAI's shape), and `content` is
`null` when the model only calls tools. Sending the `tool` results back works
like any OpenAI client: append the assistant message with its `tool_calls`,
then one `{"role": "tool", "tool_call_id": …, "content": …}` per call — the
mapper converts the history to Ollama's shape.

## Semantics worth knowing

| Detail | Behavior |
| --- | --- |
| `tool_choice` | `auto`, `none`, or a specific function are forwarded. `"required"` is rejected (`400`) — the backend cannot force a call. |
| `parallel_tool_calls: false` | Truncates the response to the **first** tool call. The backend cannot be told to emit only one, so this is a filter, not steering. |
| `strict: true` on a function | Rejected (`400`): tool-argument schemas are not enforced by the backend (`strict: false` is a no-op). `response_format` schemas *are* enforced, via Ollama's grammar. |
| Streaming | Tool calls arrive as one indexed array chunk on `delta.tool_calls`, then `finish_reason: "tool_calls"`. |
| Aliases | None here — the function names are yours; the server never maps them onto its own tool registry. |

For a client that wants the harness — real execution, approvals, sessions —
use `POST /v1/agent/turns` with `"settings": {"tools": true}`.
