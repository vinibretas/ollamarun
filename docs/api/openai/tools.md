# Tool calls

Client `tools` are passed to the backend and returned as standard
`tool_calls` with `finish_reason: "tool_calls"` — they are never executed
here, so this endpoint cannot trigger the agent harness. Full agent turns
live under the native [`/v1/agent/*` endpoints](../agent/turns.md).

- `tool_choice` accepts `auto`, `none` or a specific function;
  `tool_choice: "required"` is rejected (the backend cannot force a call).
- `parallel_tool_calls` is honored: a `false` returns only the first tool call
  the backend produced (the backend cannot be told to emit a single call, so
  this is a truncation, not model steering).
- `strict: true` on a function tool is rejected because the backend does not
  enforce tool-argument schemas (`strict: false` is a no-op). See
  [Errors](errors.md).
