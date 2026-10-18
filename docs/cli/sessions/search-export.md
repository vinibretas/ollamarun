# Search and export

Four inspection commands read the store without touching it, all sharing one
output convention: a rich table on a terminal, or `--json` / `--csv` /
`--markdown` with `--out FILE` for scripts.

## `--list` — every session

```text
$ ollamarun --list
                                 saved sessions
╭────────────┬────────────┬─────────────┬────────────┬───────┬──────────┬──────╮
│ id         │ title      │ model       │ updated_at │ turns │ messages │ tags │
├────────────┼────────────┼─────────────┼────────────┼───────┼──────────┼──────┤
│ 20261018-… │ new        │ llama3.2:3b │ 2026-10-1… │ 0     │ 0        │ []   │
│            │ session    │             │            │       │          │      │
│ 20261018-… │ Use        │ llama3.2:3b │ 2026-10-1… │ 1     │ 4        │ []   │
│            │ write_file │             │            │       │          │      │
│            │ to write…  │             │            │       │          │      │
╰────────────┴────────────┴─────────────┴────────────┴───────┴──────────┴──────╯
```

## `--search` — inside the messages

```text
$ ollamarun --search color
                                 search: color
╭───────────────┬───────────────┬───────────────┬───────────────┬──────────────╮
│ id            │ title         │ model         │ updated_at    │ match        │
├───────────────┼───────────────┼───────────────┼───────────────┼──────────────┤
│ 20261018-113… │ What color    │ minicpm-v4.6… │ 2026-10-18T1… │ What color   │
│               │ fills the     │               │               │ fills the    │
│               │ attached      │               │               │ attached     │
│               │ image? …      │               │               │ image? …     │
│ 20261018-113… │ Reply with a  │ llama3.2:3b   │ 2026-10-18T1… │ …wo-item     │
│               │ two-item …    │               │               │ markdown …   │
╰───────────────┴───────────────┴───────────────┴───────────────┴──────────────╯
```

Search covers what is in the session files — including thinking, tool calls
and results — since it scans the stored JSON.

## `--show` — replay a transcript

```text
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

It is the live view's rendering again — boxes, tool panels, colours,
thinking — plus a stats line. In the REPL, `/show [topic]` shows *model*
details (`info`, `license`, `modelfile`, …, ollama-style); the transcript
replay is the CLI's `--show ID`.

## `--export` — take a session away

`--export ID` writes the session out; `--export-format md|json|html` picks
the shape (default `md`). The markdown export serializes the tool calls as
JSON blocks:

````markdown
# Use write_file to write the text hello-from-the-model to /t…

- **id:** `20261018-113558-13ec`
- **model:** `llama3.2:3b`
- **created:** 2026-10-18T11:35:58.014222-03:00
- **updated:** 2026-10-18T11:37:11.740373-03:00

## User

Use write_file to write the text hello-from-the-model to /tmp/olladocs-demo.txt.
Then confirm you did it.

## Assistant

```json
[{"id": "call_1ftvnkvz", "function": {"name": "write_file",
  "arguments": {"path": "/tmp/olladocs-demo.txt", "content": "hello-from-the-model"}}}]
```
````

The JSON export is the session file plus the derived presentation fields
(`title`, `display_title`) — the machine-readable copy. The HTML one renders
the same transcript the replay shows into a self-contained styled document
(`recorder.export_html`), which is what `--out session.html` is for.

!!! note "Two open issues with the printed forms"

    - Printing a machine format to a **narrow terminal** can wrap long lines
      (rich printing), which breaks `--json`/`--csv` parsing; `--out FILE` is
      immune. Tracked as
      [issue #42](https://github.com/vinibretas/ollamarun/issues/42).
    - `--export-format html` currently prints the transcript to stdout before
      the document, even with `--out`; the file itself is correct. Tracked as
      [issue #43](https://github.com/vinibretas/ollamarun/issues/43).

## `--reindex`

`--reindex` rebuilds `index.json` from the session files. It is the repair
step when the index ever drifts (a manual file copy, a killed process) —
sessions themselves are the source of truth.

## The convention

| Flag | Effect |
| --- | --- |
| `--json` | JSON array of records (list/search) or the API payload (chat runs). |
| `--csv` | CSV with the table's columns. |
| `--markdown` | The same table as a markdown pipe table. |
| `--out FILE` | Write the format to a file instead of stdout. |

All four commands accept them; without a format flag you get the rich table.
