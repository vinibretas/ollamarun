# @file mentions

`@path` in a prompt inlines the file's contents into the message: the model
sees the text itself, not the path. It works in the REPL, in one-shot prompts
and for piped stdin alike, and is the quick way to hand a file to a model
without a tool call.

## Syntax

- `@` immediately before a path; the path runs to the next whitespace.
- `~` expands to your home; relative paths resolve against the working
  directory the harness runs in.
- Several mentions in one prompt are all inlined.
- In the REPL, **Tab** completes `@files` along with commands, ids and model
  names.

The contents are appended to the message as one block per file, so the model
can tell where a file starts and ends:

```text
<file name="/tmp/notes.txt">
The fjord holds a copper coin of moon.
Second line, unremarkable.
</file>
```

The path is part of the block header; the user message itself keeps the
mention as typed, and the session records the resolved paths so `/show` and
exports can list what was sent.

## Tested examples

With this file:

```console
$ printf 'The fjord holds a copper coin of moon.\nSecond line, unremarkable.\n' > /tmp/notes.txt
```

Ask about it — note the space before the `?`, which is what ends the path:

```console
$ ollamarun --no-tools --no-save llama3.2:3b "What is the first line of @/tmp/notes.txt ? Reply with just that line, verbatim."
llama3.2:3b ›
The fjord holds a copper coin of moon.
```

A missing file and an over-limit file both warn and are skipped (the model
simply does not get them):

```console
$ ollamarun --no-tools --no-save llama3.2:3b "Summarize @/tmp/nope.txt in one word."
@/tmp/nope.txt: no such file
llama3.2:3b ›
I don't see any file at `/tmp/nope.txt`.

$ ollamarun --no-tools --no-save llama3.2:3b "Summarize @/tmp/big.txt in one word."
@/tmp/big.txt: 300 KiB exceeds 256 KiB limit
```

## Limits

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_file_kb` (config) | 256 | Files larger than this are skipped with a warning. |
| — | — | Files are read as UTF-8 with undecodable bytes replaced. |

Because the content is appended to the prompt, a big file eats the context
window — the [context counter](../cli/rendering/footer.md) and
[`/context`](../cli/context/overview.md) show the damage, and
[compaction](../cli/context/compaction.md) can trim older messages.

!!! warning "Punctuation ends the mention"

    The path ends only at whitespace, so a sentence-final `.`/`,`/`)` right
    after the mention becomes part of the filename and the file is not
    inlined:

    ```console
    $ ollamarun --no-tools --no-save llama3.2:3b "Summarize @/tmp/notes.txt. One word."
    @/tmp/notes.txt.: no such file
    ```

    Put a space between the path and the punctuation
    (`@/tmp/notes.txt .`) or reorder the sentence. Tracked as
    [issue #39](https://github.com/vinibretas/ollamarun/issues/39).

## Where it fits

- [Images and vision](images.md) attaches images the same way, but as
  vision input instead of text.
- The [file tools](tools/reference.md) (`read_file`, `write_file`, …) let
  the *model* decide what to touch; `@file` is you deciding up front.
