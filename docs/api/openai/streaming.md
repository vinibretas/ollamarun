# Streaming

Streaming emits OpenAI SSE only: a role chunk, content deltas, the finish
chunk, an optional usage chunk (`stream_options.include_usage`) and
`data: [DONE]`; thinking models surface their reasoning as the non-standard
`delta.reasoning_content`, which normal clients ignore. A dropped streaming
client cancels its own upstream generation.

The native agent surface has its own typed event stream — see
[Event stream](../agent/events.md).
