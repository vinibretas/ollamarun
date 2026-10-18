# Pulling models

When the model you asked for is not installed on the target machine, the run
does not fail with a bare error. It reports where it looked, probes the free
disk space, and offers to pull — with live per-layer progress.

## The missing-model flow

```text
model nonexistent-model:99b is not installed on 100.102.94.44:11434
free disk space on 100.102.94.44:11434: unknown
pull nonexistent-model:99b? [y/n] (n):
```

- **y** pulls it (`/api/pull`, one progress bar per layer) and the run
  continues.
- **n** (the default) opens the [model picker](picker.md) to choose
  something that *is* installed.
- `--pull` skips the question and pulls without asking — the shape for
  scripts and one-shots.
- `/pull [MODEL]` does the same from inside the REPL.

The disk line is the machine that will store the model: a local run reports
its own numbers (`free disk space on this machine: 62.7 GiB of 467.3 GiB
(/usr/share/ollama/.ollama/models)`), and a remote target is probed with a
BatchMode ssh `df` — reading `OLLAMA_MODELS` from the ollama unit — which
degrades to `unknown` when ssh cannot answer.

## `--pull`

```console
$ ollamarun --no-tools --pull qwen2:0.5b "Reply with exactly: pong"
model qwen2:0.5b is not installed on 127.0.0.1:11434
free disk space on this machine: 92.9 GiB of 467.3 GiB (/usr/share/ollama/.ollama/models)
pulling qwen2:0.5b ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 352.2 MB / 352.2 MB  94 MB/s  0:00:04
…
```

Each layer reports its own bytes and rate (Ollama's progress events, one
rich bar per layer), then the model load and the answer follow. The check
above ran end to end in ~14 s on a LAN, and the model was removed again with
the API afterwards.

## `/pull` — pull or update

`/pull` works on installed models too: it checks the manifest and
re-verifies every layer, downloading only what changed.

```text
llama3.2:3b › /pull llama3.2:3b
free disk space on http://100.102.94.44:11434: unknown
llama3.2:3b is installed — checking for updates
pulling llama3.2:3b · pulling manifest  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0/? bytes
  dde5aa3fc5ff  ━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 GB  0:00:00
  966de95ca8a6  ━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 kB  0:00:00
…
```

With no argument it pulls the session's model.

## Where models come from

Pulls go to the Ollama server the session talks to — locally or on the
[`--remote`](../remote.md) machine; there is no separate remote path. The
model store is the server's (`/usr/share/ollama/.ollama/models` by default,
`OLLAMA_MODELS` overrides). Sessions store only the *name*, so a model
pulled here is usable from every front end and the API at once.

- [Housekeeping](housekeeping.md) — what happens to a pulled model when
  you switch away or exit.
- [Models API(../../api/openai/models.md) — the same list over HTTP.
