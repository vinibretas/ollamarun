# Quickstart

One-shot, or straight into the interactive REPL:

```console
$ ./ollamarun qwen3:8b "explain this repo"    # one-shot
$ ./ollamarun qwen3:8b                        # interactive REPL
```

Serve the API layer and drive it from the reference clients:

```console
$ ./ollamarun serve                           # OpenAI + agent API on :11440
$ python3 examples/openai_client.py --model qwen3:8b
$ python3 examples/agent_client.py --prompt "List the files in /tmp"
```

Where to go next:

- [Usage and flags](../cli/usage.md) for the full CLI surface.
- [The REPL](../cli/repl.md) and [slash commands](../cli/commands.md) for
  interactive use.
- [API overview](../api/index.md) for `ollamarun serve`.
- [Config file](../config/file.md) for everything configurable.
