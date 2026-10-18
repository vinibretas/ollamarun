# API overview

`ollamarun serve` runs a local HTTP API in front of Ollama, so any OpenAI
client can use the local models. The listener is LAN-accessible by default
(`0.0.0.0:11440`); `--bind IP` / `--port N` narrow or change it and are
separate from the upstream `--host` / `--remote`, which pick the Ollama
machine:

```console
$ ollamarun serve [--bind IP] [--port N] [--host HOST | --remote NAME] [--keep-loaded]
```

!!! warning "Trusted-network-only"

    The service has no authentication, and the bind address is its only
    boundary. Keep it on the LAN/tailnet and never expose the port to the
    public internet.

Two surfaces share the process:

- **[OpenAI API](openai/models.md)** — `GET /v1/models`, `POST
  /v1/chat/completions` and `POST /v1/embeddings`: model turns only, a
  stateless translation proxy.
- **[Agent API](agent/turns.md)** — full agent turns with the internal tool
  loop, approvals, sessions, files/artifacts and context fits.

## Lifecycle

**Disconnects and cancellation.** A dropped event stream is an unsubscribe —
the turn keeps running (its results land in the session) unless it asked for
`"cancel_on_disconnect": true`. `POST .../cancel` cancels explicitly: it
aborts the in-flight generation, kills cancellable subprocesses, resolves a
waiting approval as cancelled and persists the terminal state.

Shutdown is an ordered lifecycle: the listener stops accepting, new turns and
fits are refused (`503 server_shutting_down`), parked approvals resolve as
cancelled, in-flight HTTP handlers drain within `serve_shutdown_timeout`, and
`serve_unload_on_exit` (default true) unloads only the models this server
actually used — embedding models stay resident and a parallel CLI session's
models are never touched (`--keep-loaded` disables it).

## Compatibility testing

The surface is pinned by the project's test suite, which a client can trust as
the compatibility contract: a required round-trip through the official
`openai` SDK (models, embeddings in both encodings, completions with `n > 1`,
streaming with usage, typed tool calls, the 400 mappings, a mid-stream
failure) plus the deterministic request/response matrix and a real-process
smoke of `ollamarun serve` — `python3 test_ollamarun.py`, with `--live` adding
a real-backend pass (models, completions, embeddings, one agent turn).

The next pages:

- [Security and limits](security.md) — the `serve_*` boundaries.
- [Example clients](example-clients.md) — two runnable reference clients.
- [Config file](../config/file.md) — the full `serve_*` key matrix.
