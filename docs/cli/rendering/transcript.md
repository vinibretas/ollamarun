# Transcript replay

`ollamarun --show ID` replays a saved conversation the way it looked live:
the title rule and metadata line, the user's boxes, the model's rendered
answers (markdown and [math](math.md)), the tool call and result panels with
their colours, expanded thinking panels, image markers, cancelled notes, the
system prompt when the session has one — and a stats line at the end.

```console
$ ollamarun --show 20261018-113558-13ec
────────── Use write_file to write the text hello-from-the-model to /t… ──────────
id 20261018-113558-13ec · model llama3.2:3b · created 2026-10-18 11:35:58 ·
updated 2026-10-18 11:37:11 · 4 messages
        ╭──────────────────────────────────────────────────────────────── you ─╮
        │ Use write_file to write the text hello-from-the-model to             │
        │ /tmp/olladocs-demo.txt. Then confirm you did it.                     │
        ╰──────────────────────────────────────────────────────────────────────╯
╭─ tool · write_file ──────────────────────────────────────────────────────────╮
│ { "append": false, "content": "hello-from-the-model",                        │
│   "path": "/tmp/olladocs-demo.txt" }                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ tool · write_file · ok ─────────────────────────────────────────────────────╮
│ wrote 20 chars to /tmp/olladocs-demo.txt                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
The written text is indeed "hello-from-the-model" and its length is 20 characters.
1 turns · 2937 prompt tok · 51 completion tok · 71s total
```

It is not a re-run: everything comes from the session file. That is what
makes the store the conversation's record — thinking, tool arguments, tool
results, approvals and settings are all still there weeks later. (This is
also why [Storage](../sessions/storage.md) says the JSON files, not the
index, are the truth.)

`--show` takes any unique id prefix or session reference; find one with
[`--list` or `--search`](../sessions/search-export.md), the `/sessions`
picker in the REPL, or `-r` at launch. For how replays are put together
under the hood, the pipeline is `renderer.render_transcript()` reading the
same panels the live view used — see
[Architecture](../../development/architecture.md).

!!! note

    In the REPL, `/show [topic]` is a different thing: it prints *model*
    details (`info`, `license`, `modelfile`, `parameters`, `system`,
    `template`), mirroring `ollama show`. Transcript replay is the CLI's
    `--show ID`, plus the browser in `/sessions`.
