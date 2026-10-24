# Sessions

The durable record behind turns — the same JSON session store the CLI uses
(see [Storage](../../cli/sessions/storage.md)).

| Endpoint | What it does |
| --- | --- |
| `GET`/`POST /v1/sessions` | List / create sessions. |
| `GET`/`PATCH`/`DELETE /v1/sessions/{id}` | Inspect, edit or trash one. |

## Create

```console
$ curl -s http://127.0.0.1:11440/v1/sessions \
    -H 'Content-Type: application/json' \
    -d '{"model": "llama3.2:3b", "title": "api demo session", "tags": ["infra", "demo"]}'
```

```json
{
  "id": "20261018-132254-7414",
  "object": "session",
  "title": "api demo session",
  "model": "llama3.2:3b",
  "host": "http://127.0.0.1:11434",
  "remote": null,
  "created_at": "2026-10-18T13:22:54.823230-03:00",
  "updated_at": "2026-10-18T13:22:54.823284-03:00",
  "tags": ["infra", "demo"],
  "messages": 0,
  "turns": 0,
  "system": null,
  "artifacts": [],
  "active_turn_id": null
}
```

`201` on create; the model must be installed and inside the
[allowlist](../security.md#model-allowlist).

## Edit

`PATCH` accepts `title`, `model`, `system`, `tags`, `parameters`,
`num_ctx` and `gpu_layers`:

```console
$ curl -s -X PATCH http://127.0.0.1:11440/v1/sessions/20261018-132254-7414 \
    -H 'Content-Type: application/json' \
    -d '{"title": "api demo (patched)", "num_ctx": 8192,
         "parameters": {"temperature": 0.4}}'
```

The `parameters` object is **merged** over the existing options (and
validated like a turn's settings); `num_ctx` is a positive integer (refused
past the model's trained maximum) and `gpu_layers` takes the CLI spec
(`N`, `P%`, `+N`/`-N`, `all`, `cpu`, `auto`) — a placement change unloads
the resident model so the next request reloads it at the new split.

## Read

`GET /v1/sessions/{id}` returns the summary plus `history`, `stats`,
`parameters`, `tool_allow`, `tool_rules` and `tools_disabled` — everything a
client needs to render a resumed session. `GET /v1/sessions` lists the same
summaries:

```json
{"object": "list", "data": [{"id": "…", "title": "…", "model": "…", "host": "…",
 "created_at": "…", "updated_at": "…", "messages": 2, "turns": 0,
 "prompt_tokens": 0, "completion_tokens": 0, "tags": []}, …]}
```

## Delete

`DELETE /v1/sessions/{id}` moves the session to the trash (`trash-put`,
recoverable) and returns the summary with `object: "session.deleted"`.
Deleting a session never deletes its artifacts early — artifact retention is
[TTL-only](artifacts.md#storage-and-retention).

## While a turn or fit runs

A session with an active turn (or a running fit) refuses edits and deletes
with `409 session_busy` naming the `active_turn_id`/`active_fit_id`, and a
second turn for the same session is refused the same way — see
[Turns](turns.md#records-and-limits).

## Maps to the CLI

| API | CLI equivalent |
| --- | --- |
| `POST /v1/sessions` | A first run with `--session NAME` / `--title`. |
| `GET /v1/sessions/{id}` | `ollamarun --show ID` / `--export ID`. |
| `PATCH … {num_ctx}` | [`/num-ctx N`](../../cli/context/num-ctx.md). |
| `PATCH … {gpu_layers}` | [`/gpu-layers SPEC`](../../cli/context/gpu-layers.md). |
| `DELETE` | `ollamarun --delete ID`. |
