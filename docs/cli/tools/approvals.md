# Approvals and policy

Read-only tools run unattended, as do allowlisted shell commands
(`ls`/`cat`/`grep`/`rg`, read-only `git`, ... — no substitutions or
redirections) and provably-harmless Python. Everything else lists numbered
choices — run once, always allow that exact call, or always allow a suggested
generalized rule (`shell: mkdir -p :*`, `path: ~/proj/**`,
`python: imports os`) stored for the session — and you type one. Unattended
runs deny with a note unless `-y/--yes` or `/tools policy auto` is in effect.
`dangerous_command()` is the floor no rule ever covers: substitutions,
redirections, privilege/wrapper/interpreter commands, disk-level tools,
catastrophic `rm` and pipes into an interpreter always prompt again.

- `/tools` — harness-wide `on|off`, `policy MODE`, `enable`/`disable NAME…`,
  `allow`, `rules`, plus per-tool control with `/tools NAME on|off|help|call
  ARGS` (NAME also takes the invented aliases, e.g. `/tools search off`).
  `help` is generated from the tool's definition JSON — description, every
  parameter with its type and whether it is required or optional, aliases,
  approval behaviour, and what an uninstalled tool needs — and `call` runs
  the tool yourself through the same approval gate and result panel as a
  model call (`call {"query": "x"}`, `call query=x max_results=3`, or a bare
  value when a single parameter is required); a manual call never touches the
  conversation. `/tools help` (and `/help tools`) is the reference.
- `--tools` / `--no-tools` for a single run; config `tools`, `tools_policy`,
  `tools_shell_allow`, `tools_timeout`, `tools_max_output`, `tools_max_rounds`
  (default 8), `tool_colors`

The API surface has its own approval model — a turn parks undecided calls and
a client resolves them. See
[Agent API approvals](../../api/agent/approvals.md).
