# Models

| Endpoint | What it does |
| --- | --- |
| `GET /v1/models` | Installed models as OpenAI model objects. |
| `GET /v1/models/{id}` | One model (`404` + error envelope when unknown). |

```console
$ curl -s http://127.0.0.1:11440/v1/models
{"object": "list", "data": [{"id": "llama3.2:3b", "object": "model",
 "created": 1789360880, "owned_by": "ollamarun"}, …]}

$ curl -s http://127.0.0.1:11440/v1/models/llama3.2:3b
{"id": "llama3.2:3b", "object": "model", "created": 1789360880,
 "owned_by": "ollamarun"}

$ curl -s http://127.0.0.1:11440/v1/models/nope:latest
{"error": {"message": "model 'nope:latest' not found", "type": "invalid_request_error",
 "param": "model", "code": "model_not_found"}}
```

`created` is the model's modified time (Unix seconds); `owned_by` is always
`ollamarun`. The list is fetched from the upstream `/api/tags`, normalized to
tagged names and cached briefly, so a newly pulled model appears without
restarting the server.

The [serve_models allowlist](../security.md#model-allowlist) filters this
endpoint exactly like every other — a model outside it behaves like an
unknown id (`404 model_not_found`), not a permission error.
