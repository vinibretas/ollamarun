# Embeddings

Embedding-capability models route to `/api/embed` honoring
`--dimensions`/`--truncate` instead of chat.

The API surface exposes the same thing: `POST /v1/embeddings`, gated on the
model's `embedding` capability. See
[Embeddings](../../api/openai/embeddings.md).
