# Parameters

Everything a request sends to Ollama beyond the messages: the model options
(`temperature`, `num_predict`, `top_p`, …), the system prompt, the thinking
flag, keep-alive and the response format. They live on the session and are
visible in one table.

## `/params`

```text
llama3.2:3b › /params
              session parameters
╭─────────────┬──────────────────────────────╮
│ parameter   │ value                        │
├─────────────┼──────────────────────────────┤
│ temperature │ 0.2                          │
│ num_predict │ 64                           │
│ think       │ None                         │
│ keep_alive  │ 2h                           │
│ gpu layers  │ auto (ollama decides)        │
│ format      │                              │
│ system      │ built-in default (258 chars) │
╰─────────────┴──────────────────────────────╯
```

Model options appear first (only the ones set), then the session-level knobs:
`think`, `keep_alive`, `gpu layers`, `format`, `system`.

## Setting them

| Where | How |
| --- | --- |
| REPL | `/set parameter NAME VALUE` — values are coerced (`0.2` → float, `true` → bool, JSON passes through). |
| REPL | `/set parameter NAME` with no value clears it. |
| REPL | `/set system TEXT` — the system prompt (see [Prompts](../../config/prompts.md)). |
| Launch | `--option NAME=VALUE` (repeatable) or `--options-json '{"temperature": 0.7}'`. |
| Launch | `--system TEXT`, `--system-file FILE`, `--no-system`. |

```text
llama3.2:3b › /set parameter temperature 0.2
set temperature = 0.2
llama3.2:3b › /set parameter num_predict 64
set num_predict = 64
```

Options are passed through to Ollama's `/api/chat` `options` object as-is —
the same names the API and `ollama run --option` use. A few interact with
dedicated commands and settings:

- `num_gpu` — [`/gpu-layers`](../context/gpu-layers.md) wins over it in the
  payload.
- `num_ctx` — [`/num-ctx`](../context/num-ctx.md) is the managed path;
  setting the option directly works but skips the cap/warning logic.
- `think` — prefer [`/think`](../rendering/thinking.md) or `--think`, which
  also drive the display side.
- `keep_alive` — prefer `--keepalive`/config `keepalive`; it is stored as
  the session's `keepalive` and shown as `keep_alive` in the table.

## What a request carries

`/params` is a view of the session; the wire payload is assembled per round
by `build_payload()`: model, messages (history + the effective system
message), `stream`, `options`, `think`, `keep_alive`, `format` — and the
tool schemas when tools are on. Everything in the table above is part of it,
except `gpu layers`, which resolves into `options.num_gpu` per the
[/gpu-layers spec](../context/gpu-layers.md).
