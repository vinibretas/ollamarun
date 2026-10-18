# The REPL

The inline prompt_toolkit REPL is the default (`--ui inline`). It is a
superset of ollama's own commands, with [@path file mentions](file-mentions.md)
(inlined into the prompt), Tab completion for commands/session ids/models/
`@files`, prompt history (unless `OLLAMA_NOHISTORY` is set) and
[Ctrl+G](editor.md) to open `$OLLAMA_EDITOR` on the current input. The full
command set is in [Slash commands](commands.md).

While a response streams, Ctrl+T expands or collapses the thinking panel —
see [Thinking panel](rendering/thinking.md).

The alternative front end, [Textual TUI](../tui/index.md) (`--ui tui`), is
deprecated; the inline REPL is the supported one meanwhile.
