# Tool definitions

The model-facing text lives in files, not code. Every tool is one JSON file
under `ollamarun_lib/defs/tools/`, carrying:

- `name`, the `description` and JSON-schema `parameters` the model sees;
- `read_only` / `always_safe` safety flags (they feed the
  [approval policy](approvals.md));
- `requires` (`search` for `web_search`'s ddgs backend, `tts` for the speech
  tools, `chem` for the inventory tools);
- the model-invented `aliases` that resolve onto it;
- the `handler` that runs it.

A same-named file under `~/etc/config_files/ollamarun-defs/tools/` overrides
the shipped one, so tool wording can be tuned without touching Python. Bad
definition files warn and are skipped instead of breaking the harness.

Adding a tool is one JSON file plus a handler — see
[Contributing](../../development/contributing.md#where-to-change-what).
Prompts are the other half of the model-facing text; see
[Prompts](../../config/prompts.md).
