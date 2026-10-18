# API overview

`ollamarun serve` puts an HTTP API in front of Ollama from the same process
that runs the CLI. Two surfaces share the listener:

- the **[OpenAI API](openai/models.md)** — `GET /v1/models`,
  `POST /v1/chat/completions`, `POST /v1/embeddings`: model turns only, a
  stateless translation proxy any OpenAI client can talk to;
- the **[native agent API](agent/turns.md)** — full agent turns with the
  tool loop, approvals, sessions, artifacts and context fits.

```console
$ ollamarun serve [--bind IP] [--port N] [--host HOST | --remote NAME] [--keep-loaded]
```

Starting it prints what is in force — the real banner from a
`--port 0` run:

```text
ollamarun serve · 2.0 · OpenAI-compatible + agent API
  listening http://0.0.0.0:38573 (all interfaces)
  upstream  http://127.0.0.1:11434
  workspace /home/you/code/ollamarun
  models /v1/models · chat /v1/chat/completions · agent turns /v1/agent/turns ·
sessions /v1/sessions
trusted-network-only: no authentication — never expose this port to the internet
Ctrl-C stops the server (active turns are cancelled; loaded models are kept)
```

`--bind`/`--port` choose the listener (port 0 picks an ephemeral one);
`--host`/`--remote` pick the Ollama machine; the two are independent. The
workspace line is the [tool jail](security.md#tool-workspace).

!!! warning "Trusted-network-only"

    The service has no authentication, and the bind address is its only
    boundary. Keep it on the LAN/tailnet and never expose the port to the
    public internet.

## Lifecycle

**Disconnects.** A dropped event stream is an *unsubscribe*, not a
cancellation — the turn keeps running and its results land in the session.
Both sides of that were checked live:

```text
--- default: turn_ef28e7b2a20d
  client dropped the stream: ConnectionError
  status after disconnect: completed (checked 8s)
--- cancel_on_disconnect: turn_9ac3c55775b0
  client dropped the stream: ConnectionError
  status after disconnect: cancelled (checked 2s)
```

A turn that asks for `"cancel_on_disconnect": true` cancels when its stream
drops; otherwise it finishes. `POST /v1/agent/turns/{id}/cancel` is the
explicit path — it aborts the in-flight generation, kills cancellable
subprocesses, resolves a parked approval as cancelled and persists the
terminal state.

**Shutdown** is ordered: the listener stops accepting, new turns and fits are
refused (`503 server_shutting_down`), parked approvals resolve as cancelled,
in-flight HTTP handlers drain within `serve_shutdown_timeout`, and
`serve_unload_on_exit` (default true) unloads only the models this server
actually used — embedding models stay resident and a parallel CLI session's
models are never touched (`--keep-loaded` disables it).

## Everything the API can be

| Knob | Effect |
| --- | --- |
| `serve_agent: false` | Inference + files only; the native routes are not registered (`404`). The banner says `agent routes disabled`. |
| `serve_models: ["qwen3*"]` | An allowlist; a model outside it answers exactly like an unknown one (`404 model_not_found`) everywhere — list, lookup, chat, sessions, turns, fits. |
| `serve_workspace: DIR` | The tool path jail (default: the directory serve started in). |
| `serve_fit: off\|estimate\|probe` | Gates the fit surface. |

Both switches were exercised live: with `serve_agent: false` and
`serve_models: ["qwen3*"]`, the model list showed only the qwen3 entries,
`llama3.2:3b` answered `404 model_not_found`, `POST /v1/agent/turns` was
`404 unknown endpoint`, and `POST /v1/files` still answered `201`. Details
on [Security and limits](security.md).

## Compatibility testing

The surface is pinned by the project's test suite, which a client can trust
as the contract: a required round-trip through the official `openai` SDK
(models, embeddings in both encodings, completions with `n > 1`, streaming
with usage, typed tool calls, the 400 mappings, a mid-stream failure), the
deterministic request/response matrix, and a real-process smoke of
`ollamarun serve` — `python3 test_ollamarun.py`, with `--live` adding a
real-backend pass.

Next pages:

- [Security and limits](security.md) — the `serve_*` boundaries.
- [OpenAI API](openai/models.md) — models, chat, streaming, tools, errors.
- [Agent API](agent/turns.md) — full turns, events, approvals, sessions,
  artifacts, fits.
- [Example clients](example-clients.md) — two runnable references.
- [Config file](../config/file.md#serve-keys) — every `serve_*` key.
