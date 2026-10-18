# Models

| Endpoint | What it does |
| --- | --- |
| `GET /v1/models` | installed models as OpenAI model objects |
| `GET /v1/models/{id}` | one model (404 + error envelope when unknown) |

Model selection is gated everywhere by the
[`serve_models` allowlist](../security.md#model-allowlist): a model outside it
answers exactly like an unknown one (`404 model_not_found`).
