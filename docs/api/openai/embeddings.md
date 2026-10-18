# Embeddings

`POST /v1/embeddings` maps to `/api/embed`: `{model, input, encoding_format,
dimensions, user}` with `input` a non-empty string or array of non-empty
strings (a client's token arrays are rejected explicitly), `encoding_format`
`"float"` (default) or `"base64"` (the little-endian float32 packing the
official SDK sends by default), and `dimensions` forwarded to the backend for
models that support truncated vectors.

The response is the OpenAI list shape — `{object: "list", data: [{object:
"embedding", index, embedding}], model, usage: {prompt_tokens, total_tokens}}`
— with the token counts read from the backend.

The model must advertise the `embedding` capability
(`model_does_not_support_embeddings` otherwise; an unverifiable empty
capability list passes), the [`serve_models` allowlist](../security.md#model-allowlist)
applies, and the resolved-input budget bounds the inputs exactly like chat
content.
