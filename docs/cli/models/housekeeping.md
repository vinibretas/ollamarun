# Load, unload and residency

Ollama loads a model on the first request and keeps it for a while
(`keep_alive`); the harness manages that lifecycle so you are not guessing
what is resident. `/ps` is always the source of truth:

```text
llama3.2:3b › /ps
                                ollama ps
╭─────────────────┬───────────┬──────────┬────────┬─────────────────────╮
│ model           │ size      │ VRAM     │ quant  │ until               │
├─────────────────┼───────────┼──────────┼────────┼─────────────────────┤
│ llama3.2:3b     │ 3.2 GiB   │ 2.7 GiB  │ Q4_K_M │ 2026-10-18 12:51:04 │
╰─────────────────┴───────────┴──────────┴────────┴─────────────────────╯
```

## Loading

The session's model starts loading in the background the moment the REPL
opens, and a one-time `ollama ps` table prints when it becomes resident —
so the first message does not wait for a cold load you could have seen
coming. One-shots load on demand; either way the [timings footer](../rendering/footer.md)
reports the load component per request.

## Switching

`/model NAME` and `/load --model NAME` **unload the current model first**,
then the new one loads lazily:

```text
llama3.2:3b › /model qwen3:0.6b
unloaded llama3.2:3b
qwen3:0.6b › 
```

`/load SESSION` is smarter: the saved session carries its own model, so when
it differs from the current one the harness asks which to run (*current* or
*last*, defaulting to the session's) and unloads on the switch. Resuming a
session also reconnects to the machine it last ran on.

## Unloading

| Situation | Behavior |
| --- | --- |
| Leaving the REPL (`/bye`, Ctrl+D) | Unloads the session model when config `unload_on_exit` is true (the default). |
| `--keep-loaded` | Never unloads on exit. |
| `--unload` | Also unloads after a one-shot run. |
| Embedding models | **Never** unloaded — the harness leaves them resident on purpose. |
| `ollamarun serve` shutdown | Unloads only the models that server used, unless `--keep-loaded` (see [API lifecycle(../../api/index.md#lifecycle)). |

The exit unload is visible and verifiable:

```text
llama3.2:3b › /bye
unloaded llama3.2:3b
session 20261018-124236-e402 saved · ~/etc/data/ollamarun/sessions
```

(`--keep-loaded` leaves both the model and that line out.)

## Keep-alive

`--keepalive DURATION` (preferred: config `keepalive`) sets how long Ollama
keeps the model after a request — `5m`, `2h`, `0` to unload right after —
and it is stored per session, so the `until` column in `/ps` reflects it
(`/params` shows the current value under `keep_alive`).

The unload happens through `keep_alive: 0` requests against `/api/ps`
residency checks, never by killing anything — a model unloaded here behaves
exactly like one that expired on its own.
