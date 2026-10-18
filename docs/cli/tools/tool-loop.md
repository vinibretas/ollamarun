# The tool loop

A turn with tools is a loop of **rounds**. Each round is one `/api/chat`
request; between rounds the harness runs whatever the model asked for and
feeds the results back.

## One round, step by step

1. **Request out.** The payload carries the enabled tool schemas
   (`build_chat_payload` in `client.py`), so the model can choose to call
   one instead of answering.
2. **Call in.** When the reply contains `tool_calls`, each one is printed as
   a coloured call panel with its arguments (pretty-printed JSON).
3. **Decision.** The call goes through the [approval gate](approvals.md):
   auto, safe-listed, rule-covered, or asking you — a denied call is a
   normal result, not an error.
4. **Execution.** The handler runs (`bash -lc` for `run_shell`, a short
   Python program for `run_python`, direct file I/O for the file tools).
5. **Result out.** The result panel prints below the call panel, wearing the
   same colour, with `ok`/`failed` and the duration in its title.
6. **Next round.** The call and its result are appended to the session as an
   `assistant` message with `tool_calls` plus one `tool` message per result,
   and the model is asked again — now with the results in context.
7. **Done.** A reply with no tool calls ends the turn. After a tool loop the
   final answer is separated from the working rounds by a centered
   `response` divider.

Rounds are capped by `tools_max_rounds` (default 8). Past the cap the loop
warns and appends `not executed: tool-call limit reached` results so the
model still knows why nothing ran.

## A real loop

Here is a genuine (if over-eager) round — the model was told "say hello" and
decided a shell command was the way to do it:

```text
llama3.2:3b ›
╭─ tool · run_shell ───────────────────────────────────────────────────────────╮
│ {                                                                            │
│   "command": "echo Hello!"                                                   │
│ }                                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯
running run_shell…
╭─ tool · run_shell · ok · 0.00s ──────────────────────────────────────────────╮
│ $ echo Hello!                                                                │
│ exit 0                                                                       │
│ Hello!                                                                       │
╰──────────────────────────────────────────────────────────────────────────────╯
llama3.2:3b ›
How can I help you today?
```

Round 1 asked for `run_shell`; `echo` is on the safe list so it ran
unattended; round 2 answered. The result panel tells the whole story: the
command, its exit code and its output.

## When the call needs approval

This one modified state, so the harness stopped and asked (condensed):

```text
llama3.2:3b › Use write_file to write the text hello-from-the-model to /tmp/script-check.txt.
╭─ tool · write_file ─────────────────────────────────────────────────────────╮
│ { "append": false, "content": "hello-from-the-model",                        │
│   "path": "/tmp/script-check.txt" }                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
running write_file…
write_file needs confirmation: write_file modifies state
  1/y  run once
  2/a  always allow exactly this call
  3    always allow path: /tmp/script-check.txt (this session)
  4/A  auto-allow every later call  [CAREFUL]
  5/n  deny
choose (1): 1
╭─ tool · write_file · ok · 0.00s ─────────────────────────────────────────────╮
│ $ /tmp/script-check.txt (20 chars)                                           │
│ wrote 20 chars to /tmp/script-check.txt                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
llama3.2:3b ›
The written text is indeed "hello-from-the-model" and its length is 20 characters.
```

The choices are numbered and typed; `y`/`a`/`A`/`n` are accepted as the
letters for run / remember / auto / deny. Option 3 was a **suggested
generalized rule** for that path, stored in the session. The gate is covered
in [Approvals and policy](approvals.md).

## Failures are results

A tool failure does not break the turn — it becomes text the model reads.
Here is an unattended one-shot trying to write a file without `-y`:

```text
╭─ tool · write_file · failed ─────────────────────────────────────────────────╮
│ $ /tmp/unattended-demo.txt (5 chars)                                         │
│ tool not run: not auto-approved (write_file modifies state) and this run is  │
│ unattended                                                                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```

The model now knows the write did not happen and can try something else,
explain, or ask. With `-y` the same call runs (verified: the file is written
and the answer says done).

## What the model actually receives

- Each result's text is what the handler produced — stdout and exit status
  for `run_shell`, the program output for `run_python`, file content for
  `read_file`.
- `tools_max_output` (default 4000 chars) caps what goes back; a cut is
  marked `…[truncated: N of M chars omitted]…`. The panel may show fewer
  lines than the model got (`… N more lines (full output sent to the
  model)`), so a long listing never floods your terminal.
- `tools_timeout` (default 120 s) bounds a subprocess; a timeout is reported
  like any other failure.

## Cancelling a turn

Ctrl+C stops the current turn. Subprocess-backed tools are killed by process
group (through `run_cancellable()`), network tools are abandoned rather than
waited on, and a destructive call that arrives after the stop is refused
instead of run. Whatever completed before the cancel stays in the session.

## Where next

- [Approvals and policy](approvals.md) — the decision table in full.
- [Reference](reference.md) — per-tool parameters and examples.
- [The API's version of this loop](../../api/agent/turns.md) — same runtime,
  server-side, over SSE.
