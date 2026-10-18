# Remote inference

`--remote NAME` retargets every Ollama call at another machine on your
tailnet: inference, model management (`/api/ps`, load/unload) and `/pull`
happen there, while **sessions, files, images and the tool harness stay
local**. There is no separate remote code path — the same client, the same
commands, a different base URL.

```console
$ ollamarun --remote kiddo --no-tools llama3.2:3b "Reply with exactly: pong"
llama3.2:3b ›
pong
```

## Names and resolution

| Short name | Tailscale node |
| --- | --- |
| `pi` | `vinipi` |
| `kiddo` | `kiddo` |
| `g3` | `g3` |
| `optiplex7070`, `optiplex` | `optiplex7070` |
| `guri` | `guri` |

Anything else is treated as a hostname or `host[:port]`, and `remote_hosts`
in the config maps your own names to machines. Resolution reads
`tailscale status --json` and uses the peer's tailnet IP (falling back to the
MagicDNS name), so `--remote tower` works for any node — not only the short
list above. `remote_ssh_user` in the config names a non-default ssh user for
the probes below.

## The preflight

Before starting, the harness probes `/api/version` with a short (4 s)
timeout and fails fast with what it knows:

```console
$ ollamarun --remote tower --no-save llama3.2:3b "hi"
cannot reach remote tower at http://100.105.136.73:11434:
HTTPConnectionPool(host='100.105.136.73', port=11434): Max retries exceeded
with url: /api/version …
$ echo $?
1
```

The same failure happens on a live node with the wrong port
(`--remote kiddo:9`) in about a second. For the known short names, when
tailscale itself reports the node offline, the message appends the state —
`— tailscale reports kiddo offline` — so a powered-off machine is not
confused with a down Ollama.

## Probes on the remote machine

When something needs a machine fact — free disk before a pull, free VRAM for
a [context fit](context/max-gpu.md), available DRAM for
[max-cpu](context/max-cpu.md) — a `--remote` session gets it over one
BatchMode ssh probe (reading `OLLAMA_MODELS` from the ollama unit for the
disk path). The `system_status` tool reads the same numbers, and
`target="local"` flips to the machine you are sitting at:

```text
llama3.2:3b › /tools system_status call
│ kiddo (ollama at http://100.107.146.36:11434)          │
│ RAM: 7.8 GiB free of 11.4 GiB                          │
│ GPU: no nvidia-smi data                                │
│ disk (models): 784.3 GiB free of 915.3 GiB at /        │

llama3.2:3b › /tools system_status call target=local
│ this machine                                           │
│ RAM: 16.9 GiB free of 31.2 GiB                         │
│ GPU 0: 729.0 MiB free of 4.0 GiB                       │
│ disk (models): 92.9 GiB free of 467.3 GiB at /usr/share/ollama/.ollama/models │
```

A probe that cannot reach the machine degrades to `unknown` rather than
failing the command that needed it.

## Sticky sessions

Sessions store the machine they last ran on, and resuming reconnects there:

```console
$ ollamarun --remote kiddo --no-tools --session remote-demo llama3.2:3b \
    "Remember the word: anchor. Reply OK."
OK
$ ollamarun --no-tools -c remote-demo "What word did I ask you to remember?"
resuming session on kiddo (http://100.107.146.36:11434)
The word "anchor"
```

Passing `--host`/`--remote` on the resume overrides the stored target.

## What stays local

Everything that is not inference: sessions (one store, wherever you run
from), `@file` inlining, images (sent with the request), the tool harness
and its results, exports. `ollamarun serve --remote NAME` applies to the API
layer the same way — the listener is still local, the upstream is the
remote.
