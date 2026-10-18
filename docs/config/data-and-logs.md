# Data and logs

| Path | What |
| --- | --- |
| `~/etc/data/ollamarun/sessions/`, `index.json`, `history` | Session store, index and prompt history |
| `~/etc/logs/ollamarun.log` | Debug log (`--debug`) |

The store is documented in [Storage](../cli/sessions/storage.md); the debug
log receives requests, upstream detail and tracebacks for `ollamarun serve`
too. Artifacts live under `~/etc/data/ollamarun/artifacts/` — see
[Artifacts](../api/agent/artifacts.md). Context-fit calibration is kept at
`~/etc/data/ollamarun/ctxfit-calibration.json` — see
[Probes and calibration](../cli/context/probes.md).
