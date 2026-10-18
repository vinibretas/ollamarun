# Streaming markdown

Responses stream through a `rich.Live` view that re-renders the markdown as
tokens arrive — headings, lists, tables, code fences and
[display math](math.md) all appear already formatted while the model is
still writing.

## Modes

| `--render` | What you see | When |
| --- | --- | --- |
| `auto` (default) | `markdown` on a terminal, `plain` when piped | — |
| `markdown` | Styled markdown, math converted; degrades to plain off-terminal | Interactive use |
| `plain` | The text as the model wrote it (still through the live view) | When you want no markup interpretation |
| `raw` | Only the answer text on stdout, nothing else | Scripts — equivalent to `--raw` |

```console
$ ollamarun --no-tools --render raw llama3.2:3b "Reply with exactly: pong"
pong
```

`/render [markdown|plain|raw]` switches mid-session; `/render` alone shows
the current mode. `--raw` is the same as `--render raw`.

## While it streams

The view refreshes on a throttle, not per token — content stays readable and
the terminal is not redrawn hundreds of times a second. Two knobs matter:

- `max_render_chars` (config, default 12000): past this much content in one
  response, the renderer switches to raw streaming — the answer keeps
  arriving, it just stops re-parsing the markdown for speed.
- `--nowordwrap`: don't wrap long lines to the terminal width.
- `--pager` buffers the whole response and opens it in `$PAGER` when done —
  see [Pager](../pager.md).

The user message is framed in a small right-aligned box, the model's answer
flows under a rule, and a tool loop's final answer gets a centered
`response` divider (tool panels are their own boxes — see
[The tool loop](../tools/tool-loop.md)).

## Off a terminal

Piped and redirected output is always plain: the live view needs a terminal,
so `markdown` itself degrades to `plain` there. The response text is still
written line by line as it arrives. For machine output, `--raw`/`--json`
skip the presentation entirely — see
[Usage](../usage.md#output-and-rendering).

- [2D math](math.md) — what the display-math renderer does.
- [Thinking panel](thinking.md) — the collapsible reasoning view.
- [Footer](footer.md) — the tok/s and ctx line under the answer.
