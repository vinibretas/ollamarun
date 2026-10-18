# Picker and sorting

A model-less run opens the picker; `/model` opens it from inside a session.
It is a sortable table: type a number, a name, a `--sort KEY`, or `q` to
quit.

```text
                                                         models
╭────┬───────────────────────────────────────────────────────┬────────────┬─────────┬─────────────────────┬────────────╮
│  # │ name                                                  │       size │  params │ modified            │ family     │
├────┼───────────────────────────────────────────────────────┼────────────┼─────────┼─────────────────────┼────────────┤
│  1 │ gpuCurator:latest                                     │    6.1 GiB │    9.7B │ 2026-10-18 10:35:03 │ qwen35     │
│  2 │ d1:3b                                                 │    3.5 GiB │    2.7B │ 2026-10-17 03:56:52 │ lfm2       │
│  3 │ llama3.2:3b-instruct-q8_0                             │    3.2 GiB │    3.2B │ 2026-10-15 10:56:22 │ llama      │
│  4 │ llama3.2:3b                                           │    1.9 GiB │    3.2B │ 2026-10-15 10:56:22 │ llama      │
│  5 │ smollm2:1.7b                                          │    1.7 GiB │    1.7B │ 2026-10-14 12:42:04 │ llama      │
…
model number, name, --sort KEY or 'q' to quit (q):
```

Quitting without a choice is not an error — but a model-less run then has
nothing to do:

```text
model number, name, --sort KEY or 'q' to quit (q): q
no model given — pass MODEL or configure one
```

## Sorting

| Key | Order |
| --- | --- |
| `modified` (default) | Newest first. |
| `name` | Alphabetical. |
| `size` | Largest first. |
| `parameters` | Most parameters first (MoE totals included). |
| `context` | Largest context window first. |

Aliases like `recent`, `newest` and `mtime` map to `modified`.
`context` costs one lazy `/api/show` per model, fetched in parallel behind a
status line and cached per host — the first sorted-by-context prompt takes a
moment, later ones do not.

Ways in: `--sort KEY` seeds the picker at launch, the picker prompt itself
takes `--sort KEY` in place, and the session remembers the last choice for
the next run.

## From the REPL

`/model` alone opens the picker; the rest are table views:

```text
llama3.2:3b › /model --list --sort size
                                                    models · by size
╭────┬───────────────────────────────────────────────────────┬────────────┬─────────┬─────────────────────┬────────────╮
│  # │ name                                                  │       size │  params │ modified            │ family     │
├────┼───────────────────────────────────────────────────────┼────────────┼─────────┼─────────────────────┼────────────┤
…
```

| Command | Effect |
| --- | --- |
| `/model NAME` | Switch model (unloads the previous one — see [Housekeeping](housekeeping.md)). |
| `/model --list [--sort KEY]` | The table without the interactive prompt. |
| `/ps` | The resident models — `ollama ps` with VRAM, quant and unload time. |

```text
llama3.2:3b › /ps
                                ollama ps
╭─────────────────┬───────────┬──────────┬────────┬─────────────────────╮
│ model           │ size      │ VRAM     │ quant  │ until               │
├─────────────────┼───────────┼──────────┼────────┼─────────────────────┤
│ llama3.2:3b     │ 3.2 GiB   │ 2.7 GiB  │ Q4_K_M │ 2026-10-18 12:51:04 │
╰─────────────────┴───────────┴──────────┴────────┴─────────────────────╯
```

Tab-completion offers model names at `/model ` in both front ends, and the
same table is what [pulling](pulling.md) falls back to when you decline a
pull.
