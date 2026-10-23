# Tool definitions

What the model sees of a tool — its name, description and JSON-schema
parameters — is data, not code. One JSON file per tool under
`ollamarun_lib/defs/tools/`, and a same-named file in
`~/etc/config_files/ollamarun-defs/tools/` overrides it.

## The format

`read_file.json`, shipped verbatim:

```json
{
  "name": "read_file",
  "description": "Read a text file (optionally a line range) from the local machine.",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "description": "file path"},
      "offset": {"type": "integer", "description": "first line (0-based)"},
      "limit": {"type": "integer", "description": "how many lines"}
    },
    "required": ["path"]
  },
  "handler": "read_file",
  "read_only": true
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | The tool's name; must match `[A-Za-z_][A-Za-z0-9_]*`. |
| `description` | yes | Model-facing text — say *what it does* and *when to use it*; this is the model's only documentation. |
| `parameters` | yes | JSON schema with `type: "object"`, `properties`, optional `required`. |
| `handler` | yes | The handler name, looked up in `tools/registry.py`'s `HANDLERS`. |
| `read_only` | no | `true` = never asks; the calls are read-only by construction. |
| `always_safe` | no | `true` = never asks even though it does something (used by `speak`). |
| `requires` | no | A capability gate; the tool only appears when it resolves. |
| `aliases` | no | Invented names that resolve onto this tool. |

`parameters` may be an empty object (`"properties": {}, "required": []`);
the schema is what `build_tool_schemas()` hands to Ollama, so valid
JSON-schema shapes work as they do anywhere.

## Capability gates

| `requires` | The tool appears when | Uninstalled tool's help says |
| --- | --- | --- |
| `search` | `ddgs` is importable | `the ddgs package (pip install ddgs)` |
| `tts` | a `tts-remote` command resolves | `the tts-remote command` |
| `chem` | a `chemInventoryAPI` command resolves | `the chemInventoryAPI command (config: chem_inventory)` |
| `artifacts` | a serve artifact store is injected | `the serve artifact store (ollamarun serve)` |

`/tools NAME help` still works for an uninstalled tool — it shows the
definition and what it needs.

## Aliases

`aliases` maps the names models invent onto the real tool: `search` →
`web_search`, `curl` → `fetch_url`, `mkdir` → `make_dir`, `recall` →
`search_sessions`, and so on. Call-id-prefixed names (`call_fetch_url`) are
also resolved by stripping the prefix and retrying. Aliases show up in
`/tools NAME help` and complete in the REPL.

## Overrides

Put a same-named file in `~/etc/config_files/ollamarun-defs/tools/` and it
replaces the shipped one (same `name`, whole definition). That is how you
reword a tool for your own models, tighten a schema or add aliases without
touching Python. Worked example — an override whose description starts with
`CUSTOM OVERRIDE:`:

```console
$ mkdir -p ~/etc/config_files/ollamarun-defs/tools
$ cat > ~/etc/config_files/ollamarun-defs/tools/read_file.json <<'EOF'
{
  "name": "read_file",
  "description": "CUSTOM OVERRIDE: read a text file. Prefer this over run_shell cat.",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "description": "file path"},
      "offset": {"type": "integer", "description": "first line (0-based)"},
      "limit": {"type": "integer", "description": "how many lines"}
    },
    "required": ["path"]
  },
  "handler": "read_file",
  "read_only": true
}
EOF
```

Next session, the change is live:

```text
llama3.2:3b › /tools read_file help
read_file — on
CUSTOM OVERRIDE: read a text file. Prefer this over run_shell cat.
model calls it with read_file(path, offset?, limit?)
…
```

A broken file never breaks the harness — it warns and is skipped, and the
shipped definition stays in effect:

```text
ignoring tool definition ~/etc/config_files/ollamarun-defs/tools/broken.json:
bad JSON: Expecting property name enclosed in double quotes: line 1 column 3 (char 2)
```

The loader reads the shipped directory first, then the override directory, so
a later layer wins per file stem (`load_tool_definitions()`); duplicate tool
names inside one load are also skipped with a warning.

## A truly new tool

Overriding only changes what exists: the `handler` must still name a module
`ollamarun_lib/tools/` knows. Adding a *new* capability is therefore two
steps — a JSON file plus a `tools/NAME.py` module exposing `run(args, ctx)`
and its line in `tools/registry.py`'s `HANDLERS` — and the loader picks the
file up automatically.
The handlers page in
[Contributing](../../development/contributing.md#where-to-change-what) is the short
version.

## The sibling: prompts

Tool wording is model-facing text; the system/title/summarize prompts are
the other half of it, with the same override-directory mechanism — see
[Prompts](../../config/prompts.md).
