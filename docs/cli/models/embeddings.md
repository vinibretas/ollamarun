# Embeddings

Paste a prompt and a model that advertises the `embedding` capability takes
the embeddings route instead of chat: the harness calls `/api/embed` and
prints a summary of the vectors rather than a completion.

```console
$ ollamarun nomic-embed-text "hello world"
       embedding ·
     nomic-embed-text
╭────────────┬───────────╮
│ field      │     value │
├────────────┼───────────┤
│ vectors    │         1 │
│ dimensions │       768 │
│ truncate   │      true │
│ min        │ -0.171560 │
│ max        │  0.122126 │
│ mean       │  0.000305 │
╰────────────┴───────────╯
first 8: [-0.0067, -0.0013, -0.1716, +0.0084, +0.0058, +0.0698, -0.0002,
-0.0431, …]
```

The table is identity and sanity data (count, dimension, range), with the
first few components of the first vector; the full vectors are what the API
serves — see [Embeddings API](../../api/openai/embeddings.md).

## Options

| Flag | Effect |
| --- | --- |
| `--dimensions N` | Truncate the output vectors to N (models that support it). |
| `--truncate [bool]` | Inputs longer than the context window: `true` (default) truncates, `--truncate=false` errors instead. |

```console
$ ollamarun --dimensions 8 nomic-embed-text "hello world"
│ vectors    │         1 │
│ dimensions │         8 │
│ truncate   │      true │
```

Both are ollama-compatible flags; on a non-embedding model they are ignored
with a warning (`--dimensions/--truncate only apply to embedding models;
ignored`). Embedding models need a prompt — an interactive run with one
reports `embedding models need a prompt (or stdin)` and exits 2, so use a
one-shot or pipe the text.

## Detection and lifecycle

The capability comes from `/api/show`'s `capabilities` list; older servers
without it fall back to family-name hints (`bert`, `nomic-embed`, …).
Embedding models are **never unloaded** by the harness's housekeeping — the
tiny cost of keeping an embedder warm is worth more than reloading it on
every request — and the API applies its own `embedding` capability gate to
`POST /v1/embeddings` (see the [API page](../../api/openai/embeddings.md)).

- [Parameters](parameters.md) — where the rest of a request's knobs live.
- [Model picker](picker.md) — embedders sort and list like any model.
