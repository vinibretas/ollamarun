# Quickstart

Needs Linux, Python 3.10+ and a reachable Ollama server — everything else is
covered in [Installation](installation.md). This page is the shortest path
from zero to a working run, a REPL and the API layer; every command below was
executed as shown.

## One-shot

```console
$ ollamarun --no-tools llama3.2:3b "Say hello in one short sentence."
llama3.2:3b ›
Hello.
```

With a `PROMPT` argument the run is a single generation that exits when the
answer is done. Drop the prompt and it is an interactive session instead:

```console
$ ollamarun llama3.2:3b              # the inline REPL
$ ollamarun                          # model picker first, then the REPL
```

Piped stdin works as the prompt too, so it slots into scripts:

```console
$ printf 'What is 2+2? Answer with the number only.\n' | ollamarun --no-tools llama3.2:3b
llama3.2:3b ›
4
```

Useful one-shot switches: `--raw` writes only the answer text, `--json` writes
the final Ollama response payload (with usage and timings), `--no-save` keeps
the run out of the session store, and `-y` removes tool-approval prompts when
the model is allowed to act on its own. The full surface is in
[Usage and flags](../cli/usage.md).

## The REPL

A model-less run opens the picker; a run with a model goes straight to the
inline prompt:

```text
╭─ ollamarun ──────────────────────────────────────────────────────────╮
│ model   llama3.2:3b                                                  │
│ session 20261018-105614-b08e · new session                           │
│ host    http://127.0.0.1:11434                                       │
│ render  markdown · thinking collapsed · 95 local models              │
│ /help for commands · Ctrl+D to exit · Ctrl+T thinking                │
╰──────────────────────────────────────────────────────────────────────╯
llama3.2:3b › 
```

Everything interactive lives behind slash commands — `/model`, `/params`,
`/tools`, `/undo`, `/branch`, `/export`, `/bye` — and a `@path` mention
inlines a file into the next prompt. Start with [The REPL](../cli/repl.md),
[Slash commands](../cli/commands.md) and
[@file mentions](../cli/file-mentions.md).

Sessions persist as you go: leave the REPL and come back with `-c` (the most
recent session) or `--session NAME`; browse everything with `--list`. See
[Sessions](../cli/sessions/storage.md).

## The API layer

`ollamarun serve` faces both the OpenAI API and the native agent API over one
listener. Start it, then talk to it with any OpenAI client or `curl`:

```console
$ ollamarun serve --port 0 --keep-loaded
ollamarun serve · 2.0 · OpenAI-compatible + agent API
  listening http://0.0.0.0:44611 (all interfaces)
  upstream  http://127.0.0.1:11434
  workspace /home/you/code/ollamarun
  models /v1/models · chat /v1/chat/completions · agent turns /v1/agent/turns ·
sessions /v1/sessions
trusted-network-only: no authentication — never expose this port to the internet
Ctrl-C stops the server (active turns are cancelled; loaded models are kept)
```

```console
$ curl -s http://127.0.0.1:44611/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Reply with exactly: pong"}]}'
```

```json
{
  "id": "chatcmpl-44c986f679d84aa498b3ed86",
  "object": "chat.completion",
  "created": 1792331897,
  "model": "llama3.2:3b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "pong"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 30, "completion_tokens": 2, "total_tokens": 32}
}
```

Two runnable reference clients double as integration checks — the official
`openai` SDK one and a native agent-flow one:

```console
$ python3 examples/openai_client.py --model llama3.2:3b
$ python3 examples/agent_client.py --prompt "List the files in /tmp"
```

- [API overview](../api/index.md) — both surfaces, the listener and the
  lifecycle.
- [Example clients](../api/example-clients.md) — what each client exercises.
- [OpenAI API](../api/openai/chat-completions.md) and
  [Agent API](../api/agent/turns.md) — the full reference.

!!! warning "Trusted-network-only"

    The service has no authentication and the bind address is its only
    boundary. Keep it on the LAN/tailnet; never expose the port to the
    public internet.

## Where next

- [CLI overview](../cli/index.md) for everything the terminal harness does.
- [Tools](../cli/tools/index.md) for the approval-gated tool loop.
- [Config file](../config/file.md) for the keys behind the defaults.
