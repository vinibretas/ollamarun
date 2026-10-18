# Streaming

With `"stream": true` the response is OpenAI SSE: one `data:` line per
chunk, `data: [DONE]` as the terminator. A real capture (trimmed):

```console
$ curl -sN http://127.0.0.1:11440/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "llama3.2:3b",
         "messages": [{"role": "user", "content": "Count from 1 to 3, one number per line."}],
         "temperature": 0, "max_tokens": 24, "stream": true,
         "stream_options": {"include_usage": true}}'
data: {"id": "chatcmpl-d2183d18…", "object": "chat.completion.chunk", "created": 1792340150,
       "model": "llama3.2:3b",
       "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}

data: {"id": "chatcmpl-d2183d18…", …, "choices": [{"index": 0, "delta": {"content": "1"}, "finish_reason": null}]}
…
data: {"id": "chatcmpl-d2183d18…", …, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: {"id": "chatcmpl-d2183d18…", …, "choices": [],
       "usage": {"prompt_tokens": 38, "completion_tokens": 6, "total_tokens": 44}}

data: [DONE]
```

The order, every time:

1. a **role chunk** (`delta.role` set, empty content),
2. **content deltas** (one per chunk; for an `n > 1` request the choice
   `index` varies),
3. the **finish chunk** (`delta` empty, `finish_reason` set),
4. the **usage chunk** — only with `"stream_options": {"include_usage":
   true}`; `choices` is `[]`, exactly as OpenAI does it,
5. `data: [DONE]`.

## Thinking models, tool calls

- Reasoning models surface their thinking as the non-standard
  `delta.reasoning_content` (also on the non-streaming `message`), which
  normal clients ignore — verified live with `minicpm-v4.6:1b`.
- A tool-call reply streams as one indexed array chunk on
  `delta.tool_calls` with `finish_reason: "tool_calls"` — see
  [Tool calls](tools.md).

## Disconnects

A dropped streaming client cancels its own upstream generation. Each request
builds its own `OllamaClient`, so a client vanishing only affects its own
stream, never another request's. Heartbeats and the
[request timeout](../security.md#limits) never cut a stream that is still
producing.

The native agent surface has its own typed event stream with ids and replay
— see [Event stream](../agent/events.md); this page is the OpenAI surface
only.
