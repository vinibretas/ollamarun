# Editor

Two ways to bring a real editor into a conversation.

## Ctrl+G: edit the current input

Press **Ctrl+G** at the prompt and the current input opens in
`$OLLAMA_EDITOR` — or `$EDITOR` when `OLLAMA_EDITOR` is unset, falling back
to `vi`. The buffer is written to a temporary `.md` file, the editor is
launched on it, and whatever you save replaces the input line when the
editor exits:

```console
$ OLLAMA_EDITOR=nano ollamarun --no-tools llama3.2:3b
llama3.2:3b › a half-written question        ← press Ctrl+G here
```

In the editor you fix the text, save and quit, and the prompt comes back
with the new content — then Enter sends it. Tested end-to-end with a scripted
editor that replaced `draft text` with a question: the answer that came back
was for the *edited* prompt.

- An editor that cannot be started prints `editor failed: …` and the input
  is left untouched.
- `OLLAMA_EDITOR` is split shell-style, so `OLLAMA_EDITOR="code --wait"` or
  any command with arguments works.
- For quick aliases, export it once in your shell profile.

## `/edit`: redo the last message

`/edit` reopens the **last user message** in the same editor and resends it:
the edited text replaces the old message, everything after it is dropped and
the model answers again. It is the [undo/retry](sessions/undo-retry-edit.md)
family's scalpel — `/retry` resends unchanged, `/undo` just drops.

Tested flow:

```console
llama3.2:3b › Name one color. Answer with one word.
Blue
llama3.2:3b › /edit
resending edited message…
Apple
```

The editor was prefilled with the original message and returned
`Name one fruit. Answer with one word.`; the old exchange was replaced, not
appended. An empty result cancels (`edit cancelled`) and nothing is sent.

## Front ends

The inline REPL binds Ctrl+G to the current input. In the deprecated
[Textual TUI](../tui/index.md) the same key opens `$EDITOR`; `/edit` works in
both.
