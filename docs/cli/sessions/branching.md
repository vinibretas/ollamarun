# Branching

`/branch [TITLE]` forks the conversation into a new session, leaving the
current one untouched — the safe way to explore a different direction
without losing the path you were on.

```text
llama3.2:3b › /branch side quest
branched 20261018-122252-de60
```

The current session is saved first, then the branch is created as a full
copy and becomes the active session. Everything conversational and
behavioral comes along:

- all messages so far (the branch starts with the same history),
- `system`, `parameters`, `think`, `keepalive`, `format`, `gpu_layers`,
- tags, stats, the approval `tool_allow` fingerprints and `tool_rules`,
- `tools_disabled` and the tts settings.

What it does **not** copy: the title (unless you pass one, it becomes
`<original title> (branch)`) and the id — the branch is an independent
session from the moment it is created, so future turns on either side never
interfere.

Typical uses:

- **Try an alternative.** Branch, then `/retry` or `/edit` on the branch;
  the original stays exactly as it was.
- **Fork at a decision point.** Get a file to a working state, branch, and
  keep one line of work "clean" while you experiment.
- **Split a long session** into per-topic children before the context gets
  tight.

Going back is ordinary session work: `/sessions` (or `-c`/`--session`/
`--show` from the CLI) lists both, and the original is one pick away. The
prompt column in `/sessions` shows the title, so `(branch)` children are easy
to spot.

See [Search and export](search-export.md) for finding a parent again by
content, and [Save and load](save-load.md) for resuming options.
