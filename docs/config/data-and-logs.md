# Data and logs

Everything the harness persists lives under three roots:

| Path | What |
| --- | --- |
| `~/etc/config_files/ollamarun.json` | The [config](file.md). |
| `~/etc/config_files/ollamarun-defs/{tools,prompts}/` | Per-user overrides for the model-facing text. |
| `~/etc/data/ollamarun/sessions/` | One JSON per conversation. |
| `~/etc/data/ollamarun/index.json` (+ `.lock`) | The session index and its flock. |
| `~/etc/data/ollamarun/history` | Prompt input history (off with `OLLAMA_NOHISTORY`). |
| `~/etc/data/ollamarun/artifacts/` | The serve [artifact store](../api/agent/artifacts.md). |
| `~/etc/data/ollamarun/ctxfit-calibration.json` | Measured [context-fit samples](../cli/context/probes.md#calibration). |
| `~/.cache/ollamarun/session-hosts/` | [Background session](../cli/sessions/background.md) sockets, metadata and start locks (`0700`, same user). |
| `~/etc/logs/ollamarun.log` | The debug log (`--debug`). |
| `~/etc/logs/ollamarun-session-host.log` | Background host stdout/stderr. |

## The session store

One atomic JSON file per conversation; the index is a cache over them and
`--reindex` rebuilds it. Deletes always go through `trash-put`, so a
mistaken delete is recoverable — see
[Storage](../cli/sessions/storage.md).

## The debug log

`--debug` turns on request-level logging to `~/etc/logs/ollamarun.log` for
both the CLI and `ollamarun serve` (which also sends its tracebacks there —
clients never see them). A one-shot with `--debug` leaves a compact trail:

```text
2026-10-18 13:50:56,105 DEBUG ollamarun 2.0 starting
2026-10-18 13:50:56,108 DEBUG Starting new HTTP connection (1): 100.107.146.36:11434
2026-10-18 13:50:56,114 DEBUG http://100.107.146.36:11434 "POST /api/show HTTP/1.1" 200 None
2026-10-18 13:50:56,310 DEBUG http://100.107.146.36:11434 "POST /api/chat HTTP/1.1" 200 None
```

The log is append-only and grows; trim or rotate it yourself if it matters
(nothing else reads it).

## Config creation

The config file is created from the defaults on first run; the session store
and log directory are created on demand. Removing the config file just
recreates it with defaults — sessions and artifacts are untouched (they live
under `~/etc/data/`).

## What is not stored

- Nothing is sent anywhere except your Ollama server(s): the harness is
  local-first, and `--remote` only retargets *which* machine receives the
  inference.
- Prompts and images live in the session only per your `image_storage` /
  `--no-save` choices; there is no telemetry.
