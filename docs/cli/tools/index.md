# Tools

With `--tools` (the default), the request carries the 23 built-in tool
definitions (some capability-gated) and the model can call them through Ollama
function calling.

A few tools are wired to my own side projects: `speak`/`list_voices` drive
`tts-remote` and the `chem_*` lookups go through `chemInventoryAPI`. They are
capability-gated — they only appear when the CLI resolves — and mostly serve
as examples of the extension mechanism: adding a tool (personal or general)
is one JSON file under `ollamarun_lib/defs/tools/` plus a handler.

- [The tool loop](tool-loop.md) — how calls, results and replay look.
- [Reference](reference.md) — every built-in tool.
- [Approvals and policy](approvals.md) — what runs unattended and what asks.
- [Definitions](definitions.md) — the JSON format and per-user overrides.
