# n > 1 choices

`n` runs that many generations, 1..8, emulated **sequentially** — each
choice is a full `/api/chat` call — with the usage counters summed:

```console
$ curl -s http://127.0.0.1:11440/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Name one color."}],
         "n": 2, "temperature": 0.9, "max_tokens": 8}'
```

```json
{
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "Blue."}, "finish_reason": "stop"},
    {"index": 1, "message": {"role": "assistant", "content": "Blue."}, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 58, "completion_tokens": 6, "total_tokens": 64}
}
```

(That run happened to sample the same answer twice; the choices are
independent samples.)

- The choices carry `index` 0..n-1 in request order.
- `usage` is the sum over the emulated choices.
- On a stream, chunk deltas carry the per-choice `index`; the usage chunk
  reports the summed counters and closes the stream as always.
- `n` outside 1..8 or a non-integer is rejected before any generation:

  ```console
  $ … -d '{"model": "llama3.2:3b", "messages": […], "n": 9}'
  {"error": {"message": "n is limited to 8 emulated generations",
   "type": "invalid_request_error", "param": "n", "code": "unsupported_value"}}
  ```

Because each emulated choice is a full generation, a large `n` costs
proportionally — on a local GPU that is real seconds, not a rounding error.
