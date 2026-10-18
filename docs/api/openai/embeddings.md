# Embeddings

`POST /v1/embeddings` maps to Ollama's `/api/embed`. The request shape is
`{model, input, encoding_format, dimensions, user}`: `input` is a non-empty
string or array of non-empty strings (a client's token arrays are rejected
explicitly), `encoding_format` is `"float"` (default) or `"base64"` (the
little-endian float32 packing the official SDK sends by default), and
`dimensions` is forwarded for models that support truncated vectors.

```console
$ curl -s http://127.0.0.1:11440/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{"model": "nomic-embed-text", "input": "hello world"}'
```

```json
{
  "object": "list",
  "model": "nomic-embed-text",
  "data": [{"object": "embedding", "index": 0, "embedding": [-0.0068, -0.0014, -0.1713, …]}],
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

768 floats for that model — the same values whichever encoding you ask for:

```console
$ curl -s … -d '{"model": "nomic-embed-text", "input": "hello world", "encoding_format": "base64"}'
# data[0].embedding is a base64 string; decoded:
decoded dims: 768 first3 [-0.0068, -0.0014, -0.1713]
```

## Gates and limits

- The model must advertise the `embedding` capability
  (`model_does_not_support_embeddings` otherwise; an unverifiable *empty*
  capability list passes — older servers).
- The [`serve_models` allowlist](../security.md#model-allowlist) applies
  exactly as elsewhere.
- Inputs are charged to the same resolved-input budget as chat content
  (count, per-input bytes, total text), with `413 input_too_large` naming
  the limit.
- The token counts in `usage` come from the backend, not a client-side
  estimate.

The CLI has the same route for one-shot use — `ollamarun nomic-embed-text
"hello world"` prints a summary table, and `--dimensions`/`--truncate` map to
the same fields (see [Embeddings](../../cli/models/embeddings.md)).
