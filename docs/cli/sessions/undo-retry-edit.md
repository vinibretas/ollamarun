# Undo, retry and edit

Three ways to redo the recent past. All of them operate on the **last
exchange**, where an exchange is everything from one user message through the
model's answer — tool rounds included, so a turn that called tools goes away
as one unit.

## `/undo` — drop the last exchange

```text
llama3.2:3b › /undo
undone
```

The last user message and everything after it are removed, the stats'
turn count drops by one, and the session is saved. With an empty history it
says `nothing to undo`.

## `/retry` — ask again

```text
llama3.2:3b › /retry
retrying…
llama3.2:3b › Blue
```

The last user message stays; every assistant/tool message after it is
discarded and the request is made again, so you get a fresh sample. Retrying
after a cancelled or failed answer is the usual reason.

## `/edit` — fix the question

`/edit` reopens the last user message in `$OLLAMA_EDITOR`, replaces it with
what you save, drops the old answer and regenerates — the scalpel of the
three:

```text
llama3.2:3b › Name one color. Answer with one word.
Blue
llama3.2:3b › /edit
resending edited message…
Apple
```

The editor was prefilled with the original message and returned
`Name one fruit. Answer with one word.`; the exchange was replaced, not
appended. An empty editor result cancels (`edit cancelled`). Full mechanics
on the [Editor](../editor.md) page.

## Which one?

| Goal | Command |
| --- | --- |
| Forget the last exchange and go on | `/undo` |
| Same question, another answer | `/retry` |
| Change the question itself | `/edit` |
| Fork instead of rewriting | [`/branch`](branching.md) |

All three keep the rest of the conversation (and every setting) intact, and
all three are saved immediately — closing the session afterwards does not
bring the undone exchange back.
