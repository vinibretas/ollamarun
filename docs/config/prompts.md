# Prompts

The model-facing text lives in files, not code:

- `ollamarun_lib/defs/prompts/` — `system.md` (the built-in default system
  prompt), `system_tools.md` (the tool, web-search, math-in-code and
  tool-result rules, appended when tools are on), `title.md` and
  `summarize.md` (both take the text to wrap as `{{content}}`).

A same-named file under `~/etc/config_files/ollamarun-defs/prompts/` overrides
the shipped one, so prompts can be tuned without touching Python.

Every request carries a system prompt: the session's own (`--system`,
`--system-file`, config `system`, `/system TEXT`) when set, the built-in
default otherwise. `--no-system` / config `system: ""` / `/system off` send
none, `/system clear` returns to the default, and `/system` alone shows what
is in effect. It is sent as a leading system message and never enters the
saved history.

Tool wording is the other half of the model-facing text — see
[Tool definitions](../cli/tools/definitions.md).
