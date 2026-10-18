# Thinking panel

Reasoning models stream their thinking alongside the answer. The harness
shows it live but collapsed, so it never buries the answer you asked for.

## The collapsed line

While the model thinks, a single animated line ticks in place:

```text
⠋  thinking · 161 chars · Ctrl+T to expand
```

The count grows as tokens arrive. When the answer starts, the line stays put
(the thinking is kept, just folded) and the response streams under it.

## Ctrl+T, and the three modes

**Ctrl+T** expands the panel in place — a boxed, dim, markdown-rendered
preview capped to the terminal's height — and pressing it again collapses
it. The stream never pauses for either.

| Start mode | Where | Effect |
| --- | --- | --- |
| `collapsed` | config `thinking` (default) | The ticking line. |
| `expanded` | `--show-thinking`, config | The panel starts open. |
| `hidden` | `--hidethinking`, config | Thinking is not shown at all. |

`/thinking [show|collapse|hide|toggle|full]` covers them from inside a
session — and `full` prints the whole stored reasoning after the fact, no
terminal-height cap:

```text
qwen3:1.7b › /thinking full
│ Okay, the user is asking for a two-item markdown bullet list of primary
│ colors. Let me think. The primary colors in painting are typically red
│ and blue, but wait, in light theory, it's red, green, and blue. …
```

The thinking is stored on the assistant message either way (session JSON,
exports and `--show` include it), so hiding it never loses it.

## `/think` is a different knob

Easy to mix up:

- **`/think [on|off|low|medium|high]`** — whether the *model* thinks at all
  (sent as Ollama's `think` parameter).
- **`/thinking [mode]`** — how the thinking is *displayed*.

`--think STR` is the launch flag for the former; the config key `thinking`
is the display mode.

## Off a terminal

Piped and redirected runs print the thinking text plainly before the answer
(no live panel exists there); `--hidethinking` removes it. `--json` carries
it in `message.thinking`. The API surface exposes reasoning models as
`delta.reasoning_content` on the OpenAI side — see
[Streaming(../../api/openai/streaming.md).
