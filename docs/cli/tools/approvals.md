# Approvals and policy

Nothing the model asks for runs just because it asked. Every call is
classified, and only the clearly safe classes go unattended.

## What runs unattended

| Class | Examples | Why |
| --- | --- | --- |
| Read-only tools | `read_file`, `list_dir`, `fetch_url`, `web_search`, `search_sessions`, `list_models`, `system_status`, `context_status`, `session_info` | `read_only: true` in the definition; nothing is modified. |
| `speak` | — | `always_safe: true` (speaking changes nothing). |
| Safe-listed shell | `ls`, `cat`, `grep`, `rg`, `find`, `wc`, `file`, `stat`, `tree`, `du`, `df`, `sort`, `cut`, `tr`, `diff`, `echo`, `date`, `which`, `whoami`, `uname`, read-only `git` | Every segment of the pipeline is on the list, with no substitutions, redirections or grouping. |
| Provably harmless Python | `print(6*7)` | Only safe modules/nodes; no filesystem, network, imports of `os`/`sys`, calls like `open`/`eval`. |
| Rule-covered calls | a remembered exact call, a `shell:`/`path:`/`python:` rule | You granted it earlier this session. |
| Policy `auto`, or `-y/--yes` | — | Everything runs; the sharpest tool in the box. |

Everything else **asks** in the REPL and is **denied with a note** in an
unattended run.

## The ask

The prompt lists numbered choices — run once, remember exactly this call,
one line per suggested generalized rule, auto-allow, deny:

```text
write_file needs confirmation: write_file modifies state
  1/y  run once
  2/a  always allow exactly this call
  3    always allow path: /tmp/script-check.txt (this session)
  4/A  auto-allow every later call  [CAREFUL]
  5/n  deny
choose (1): 1
```

Type a number or the letter; Enter takes option 1. Choosing a rule prints a
`allowed by rule …` note on later calls it covers.

## Rules

A rule is a session-scoped grant, stored in the session under `tool_rules`
(and `tool_allow` for exact remembered calls). Four kinds:

| Kind | Covers | Label |
| --- | --- | --- |
| `call` | exactly one call, by fingerprint | `exactly this call` |
| `shell` | one command's literal prefix; an implicit `:*` tail | `shell: mkdir -p :*` |
| `path` | one tool's path argument, by glob relative to the directory it ran in | `path: ~/proj/**` |
| `python` | a `run_python` snippet importing/using only the listed module roots | `python: imports math, statistics` |

`/tools rules` lists them; `/tools rules add KIND PATTERN` stores one by
hand and `/tools rules rm N` removes it. Suggestions are built from the call
being approved (`suggest_rules` in `tools/rules.py`), so the number of choices
varies — a compound shell command, for example, only ever offers the
exact-call grant.

## The floor no rule covers

`dangerous_command()` is the set of things a `shell` rule may never soften;
they prompt again every time, even under a stored rule:

- variable or backtick substitution (`$…`, `` `…` ``),
- output/input redirection (`>`, `<`) and grouping/brace expansion (`()`, `{}`),
- pipes into an interpreter (`curl … | sh`, `… | python3`),
- privilege/wrapper commands (`sudo`, `doas`, `su`, `env`, `xargs`, `nice`,
  `nohup`, `timeout`, …),
- interpreters themselves (`sh`, `bash`, `python`, `node`, …),
- disk/power/network level tools (`dd`, `mkfs`, `mount`, `shutdown`,
  `iptables`, `nc`, …),
- a recursive `rm` aimed at a whole root (`/`, `~`, `*`, `.`, …).

The same floor keeps unsafe Python out: `open`, `exec`, `eval`, `os`, `sys`,
`subprocess`, `socket`, network/OS modules and friends always prompt.

## Shell details

`shell_safety()` splits a command line on `|`, `||`, `&&`, `;`, `&` and
newlines and checks **every** segment: each must be on the allowlist and free
of the floor above. `git` is allowed only for read-only subcommands
(`status`, `log`, `diff`, `show`, …; `git config` only through `--get`
/`--list`-style flags). `tools_shell_allow` in the config adds commands of
your own to the allowlist (remember: best-effort last).

Python's version splits a snippet into imports, calls and attribute access;
its allowlist of modules (math, json, re, statistics, …) is what the
`safe list` covers. Everything else asks.

## Unattended runs

A non-interactive run (piped output, no terminal) cannot ask, so an
otherwise-askable call is denied as a result:

```text
╭─ tool · write_file · failed ─────────────────────────────────────────────────╮
│ $ /tmp/unattended-demo.txt (5 chars)                                         │
│ tool not run: not auto-approved (write_file modifies state) and this run is  │
│ unattended                                                                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```

The model sees that and reacts. `-y/--yes` makes everything run (verified:
the same call writes the file and the answer arrives); `/tools policy auto`
does the same for a session. `--tools` / `--no-tools` switch the harness off
entirely for a run.

## Policy modes and configuration

| Key / command | Effect |
| --- | --- |
| `/tools policy auto\|allowlist\|ask` | Global mode. `ask` asks for *everything* not always-safe; `allowlist` is the default; `auto` approves everything. |
| `--yes` / `-y` | Per-run auto approval (also silences `--delete`). |
| `tools_policy`, `tools_shell_allow`, `tools_timeout`, `tools_max_output`, `tools_max_rounds` | Config defaults for the above. See [Tools](index.md#configuration). |
| `/tools NAME on\|off`, `/tools enable\|disable NAME…` | Take tools out of the model's menu. |

## Over the API

The [agent API](../../api/agent/approvals.md) runs the same engine, but its
policy is never `auto`: sessions submit with `approval: "interactive"` to
park undecided calls for a client decision, or `approval: "policy"` to let
the allowlist/rules decide and deny the rest with a note.
