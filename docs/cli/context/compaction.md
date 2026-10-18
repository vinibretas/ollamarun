# Compaction

Long conversations eventually press against the context window. Compaction
replaces the **older half** of the history with a summary the model wrote, so
the thread continues without the raw transcript.

## The warning

At 80% of the effective window, the REPL says exactly what to do:

```text
context 13112/16384 tokens (80%) — /summarize to compact
```

It is printed once per turn boundary (the same `context_warning()` call
means the agent API emits a `turn.warning` event at the start of a turn
instead of printing).

## `/summarize`

```text
llama3.2:3b › /summarize
summarize 2 older messages? [y/n] (y): y
summarizing…
compacted 2 messages → summary
```

What happens, in order:

1. Messages older than the tail are gathered. The tail that is kept is the
   last **2** messages when the count is even, otherwise the last **1** — so
   the split never cuts an exchange in half.
2. You confirm (`Y` by default). Fewer than four messages is a no-op
   (`not enough messages to summarize`).
3. A one-shot, non-streaming request is sent with the
   [`summarize.md` prompt](../../config/prompts.md) wrapping the transcript.
4. On success the history becomes:

   ```text
   user       [Summary of earlier conversation]
              <the model's summary>
   assistant  Understood — continuing from that summary.
   <the kept tail>
   ```

   An empty summary is refused (`empty summary; keeping messages`) and
   nothing changes.
5. The session is saved, and the last-prompt/completion counters reset to
   unknown until the next request measures again.

The replacement is written to the session file, so it is not undone by
[`/undo`](../sessions/undo-retry-edit.md) — the old messages are gone once it
lands.

## Automatic compaction

Config `auto_compact` picks the policy:

| Value | Behavior |
| --- | --- |
| `"ask"` (default) | Only the 80% warning; compaction stays in your hands. |
| `"auto"` | At **85%** with more than four messages the REPL prints `context almost full — summarizing older messages` and summarizes without asking. |

The agent API runs the `"auto"` path headlessly at turn start regardless of
the front end's configuration — a turn that would overflow compacts first
and reports the warning on its event stream.

Compaction is a model call like any other: it costs one round trip (with
`think` disabled) and produces a user/assistant pair that the next request
carries. If the summary quality matters to your workflow, keep `"ask"` and
edit the transcript yourself before saying yes.
