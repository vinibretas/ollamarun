# Sessions

The durable record behind turns — the same JSON session store the CLI uses:

| Endpoint | What it does |
| --- | --- |
| `GET`/`POST /v1/sessions` | list / create sessions |
| `GET`/`PATCH`/`DELETE /v1/sessions/{id}` | inspect, edit (title/model/system/tags/parameters/`num_ctx`/`gpu_layers`) or trash a session; edits and deletes are refused (409 `session_busy`) while a turn or fit runs |

A session's context size and GPU placement are first-class fields:
`PATCH /v1/sessions/{id}` (and the turn `settings`) accept `num_ctx` (a
positive integer, refused past the model's trained maximum) and `gpu_layers`
(`N`, `P%`, `+N`/`-N`, `all`, `cpu` or `auto`; a change unloads the resident
model, exactly like `/gpu-layers`).

```console
$ curl -s http://127.0.0.1:11440/v1/sessions \
    -H 'Content-Type: application/json' \
    -d '{"model": "qwen3:8b", "title": "build log triage", "tags": ["infra"]}'
{"id": "ses_…", "object": "session", "title": "build log triage",
 "model": "qwen3:8b", "host": "http://127.0.0.1:11434", "tags": ["infra"],
 "messages": 0, "turns": 0, "system": null, "artifacts": [],
 "active_turn_id": null, "created_at": "…", "updated_at": "…"}

$ curl -s -X PATCH http://127.0.0.1:11440/v1/sessions/ses_… \
    -H 'Content-Type: application/json' \
    -d '{"num_ctx": 32768, "gpu_layers": "80%",
         "parameters": {"temperature": 0.4}}'
```

Session history lives under `~/etc/data/ollamarun/sessions/` — see
[Storage](../../cli/sessions/storage.md). Context fitting over the API is on
[Context fits](fits.md).
