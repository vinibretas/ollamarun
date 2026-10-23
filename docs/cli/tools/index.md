# Tools

With tools on — the default — every chat request carries the built-in tool
definitions, and the model can call them through Ollama function calling. The
harness is the difference between a chat that *says* it listed your files and
one that actually did: calls are executed locally behind an approval gate,
their results go back to the model as `tool` messages, and the loop repeats
until the model answers without asking for anything else.

- [The tool loop](tool-loop.md) — what happens between a call and the answer.
- [Approvals and policy](approvals.md) — the gate and its modes.
- [Definitions](definitions.md) — the JSON the model sees, and how to change it.
- [Reference](reference.md) — every tool with its parameters.

## What the model gets

The request carries one schema per **enabled** tool, built from the
definition files: name, description and JSON-schema parameters. Which tools
are offered depends on the host:

| Condition | Tools |
| --- | --- |
| Always | `run_shell`, `run_python`, `read_file`, `read_pdf`, `write_file`, `edit_file`, `list_dir`, `make_dir`, `move_file`, `copy_file`, `delete_file`, `fetch_url`, `search_sessions`, `list_models`, `system_status`, `context_status`, `session_info` |
| The `ddgs` package resolves | `web_search` |
| A `tts-remote` CLI resolves | `speak`, `list_voices` |
| A `chemInventoryAPI` CLI resolves (config `chem_tools: true`) | `chem_search`, `chem_container`, `chem_locations` |
| `ollamarun serve` runs an artifact store | `publish_artifact` |

Each tool also has **aliases** for names models tend to invent — `/tools
search off` disables `web_search`, and a call to `curl` or `mkdir` resolves
onto the real tool before the registry is searched. `/tools NAME help` lists
a tool's aliases.

With a response format set (`--format`), the harness steps aside entirely:
the request carries no schemas **and** the system prompt drops the tool
rules, so the model is never told about tools it cannot call. `/tools` shows
each tool as `off` and says why, and `session_info` reports tools off.

## The four ways a call is handled

1. **Auto** — read-only tools (`read_file`, `list_dir`, …) and the
   `speak` tool never ask.
2. **Safe list** — allowlisted shell commands (`ls`, `cat`, `grep`, `rg`,
   read-only git, …) and provably harmless Python run unattended.
3. **Rules and remembered calls** — a generalized rule or an exact remembered
   call from this session runs without asking.
4. **Ask** — everything else asks in the REPL; in an unattended run it is
   denied with a note, unless `-y/--yes` or `/tools policy auto` is in
   effect.

Details and the prompt itself: [Approvals and policy](approvals.md).

## Seeing the state

`/tools` is the dashboard — state, approval class and the policy line:

```text
llama3.2:3b › /tools
                          tools
╭─────────────────┬───────┬─────────────────────────────╮
│ tool            │ state │ approval                    │
├─────────────────┼───────┼─────────────────────────────┤
│ context_status  │ on    │ auto (read-only)            │
│ copy_file       │ on    │ rules, else asks            │
│ delete_file     │ on    │ rules, else asks            │
│ edit_file       │ on    │ rules, else asks            │
│ fetch_url       │ on    │ auto (read-only)            │
│ list_dir        │ on    │ auto (read-only)            │
│ list_models     │ on    │ auto (read-only)            │
│ make_dir        │ on    │ rules, else asks            │
│ move_file       │ on    │ rules, else asks            │
│ read_file       │ on    │ auto (read-only)            │
│ read_pdf        │ on    │ auto (read-only)            │
│ run_python      │ on    │ safe list, rules, else asks │
│ run_shell       │ on    │ safe list, rules, else asks │
│ search_sessions │ on    │ auto (read-only)            │
│ session_info    │ on    │ auto (read-only)            │
│ system_status   │ on    │ auto (read-only)            │
│ web_search      │ on    │ auto (read-only)            │
│ write_file      │ on    │ rules, else asks            │
╰─────────────────┴───────┴─────────────────────────────╯
policy allowlist · rules 0 (/tools rules) · max rounds 8 · timeout 120s · output cap 4000 chars
```

The same table appears whether the tools are gated by capability or disabled
by you (`disabled`), and `/tools help` explains every form:

```text
╭──────────────────────────────┬──────────────────────────────────────────────────────╮
│ form                         │ what it does                                         │
├──────────────────────────────┼──────────────────────────────────────────────────────┤
│ (none) · list · status       │ table of tools with state and approval mode          │
│ on | off                     │ enable or disable the whole tool harness             │
│ policy auto|allowlist|ask    │ global approval policy                               │
│ enable NAME… | disable NAME… │ toggle several tools at once                         │
│ NAME on | off                │ toggle one tool, e.g. `/tools web_search off`        │
│ NAME help                    │ description, parameters (required/optional), aliases │
│ NAME call ARGS               │ run it yourself: call {"query": "x"} · call query=x  │
│ help [NAME]                  │ this reference, or one tool's page                   │
│ allow                        │ safe shell command allowlist                         │
│ rules [add|rm]               │ stored approval rules                                │
╰──────────────────────────────┴──────────────────────────────────────────────────────╯
```

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `tools` | `true` | Offer tools at all. |
| `tools_policy` | `allowlist` | `allowlist`, `ask` or `auto` (see approvals). |
| `tools_shell_allow` | `[]` | Extra commands for the shell allowlist. |
| `tools_timeout` | 120 | Per-tool subprocess timeout, seconds. |
| `tools_max_output` | 4000 | Chars of tool output that go back to the model. |
| `tools_max_rounds` | 8 | Tool rounds per turn before the loop stops. |
| `tool_colors` | `{}` | Per-tool panel colours, e.g. `{"run_shell": "green"}`. |

`--tools`/`--no-tools` override for a run; `/tools on|off` for a session, and
`/tools NAME on|off` per tool. Disabled tools are not sent to the model at
all, so it does not waste a round calling something that cannot run.

## The record

Calls, results and approvals are stored on the session like any other
message, so `--show` replays them the way they looked live, and exports
include them. A tool loop that ends in an answer is separated from it by a
centered `response` divider; a round that is still working keeps the plain
rule.

## Where the pieces live

- **Definitions** (what the model sees): `ollamarun_lib/defs/tools/*.json`,
  overridable per user — [Definitions](definitions.md).
- **Approval policy**: the allowlist, rules engine and `dangerous_command`
  floor — [Approvals and policy](approvals.md).
- **The loop**: `runtime.py` drives rounds; `repl/core.py` renders them —
  [Architecture](../../development/architecture.md).
