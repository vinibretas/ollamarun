# Footer

Two lines of live feedback sit around an answer: the tok/s + context footer
while it streams, and the `⏱` timings line under it when it finishes.

## Live tok/s and context

```text
~20.0 tok/s · (avg 20.9) · 48 tok · ctx 12.345k/32.768k (38.123%)
```

| Segment | Meaning |
| --- | --- |
| `~20.0 tok/s` | The rate over a 5-second sliding window (`TpsEstimator`), so it reacts quickly to speed changes. |
| `(avg 20.9)` | The cumulative average for the response. |
| `48 tok` | Tokens streamed so far this response (thinking counts). |
| `ctx 12.345k/32.768k (38.123%)` | Context the next request will carry over the effective window. |

Counting rules: values read raw below 1k, 3-decimal `k` above. The ctx part
appears from the second round on, when there is a base to count from — it
adds the current round's chunks to the last request's totals.

The **denominator is the window the model is actually loaded with**, not the
model's architectural maximum: a session `num_ctx` wins, then the loaded
runner's `/api/ps` window, then the model default. Until the runner reports,
the counter is marked provisional. (That resolution order is the same one
[`/context`](../context/overview.md) and the 80%/85% warnings use.)

`--no-tps` hides the footer for a run; `/tps [on|off]` toggles it
mid-session; config `tps_footer: false` turns it off permanently. The
deprecated [TUI](../../tui/index.md) shows the same numbers in its status bar.

## Timings

When a response ends on a terminal, a dim line prints under it:

```text
⏱  load 3 ms · prompt 87 tok (1030 tok/s) · eval 2 tok (21.3 tok/s) · total 184 ms
```

| Segment | Meaning |
| --- | --- |
| `load` | Time Ollama spent loading the model (0 when warm). |
| `prompt` | Prompt tokens and their processing rate; `cached` shows reused KV. |
| `eval` | Generated tokens and their rate. |
| `total` | The whole request. |

Visibility rules:

- On a terminal it is always shown — `--verbose` is obsolete there and only
  matters off a terminal.
- Piped/redirected output omits it **unless `--verbose`**:

  ```console
  $ ollamarun --no-tools llama3.2:3b "Reply with exactly: pong"
  llama3.2:3b ›
  pong
  $ ollamarun --no-tools --verbose llama3.2:3b "Reply with exactly: pong"
  llama3.2:3b ›
  pong
  ⏱  load 3 ms · prompt 87 tok (1030 tok/s) · eval 2 tok (21.3 tok/s) · total 184 ms
  ```

- `--raw` and `--json` never carry it.
- Config `show_stats: false` turns it off everywhere.

The same line is what `--show` prints at the end of a transcript replay (with
the session's totals instead of one request's).
