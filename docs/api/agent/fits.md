# Context fits

Context and placement over the API come in two speeds.

`POST /v1/agent/fit` estimates synchronously (no model loads) —
`{model?, session_id?, mode: "gpu"|"cpu", cpu_percent, gpu_layers?, apply?}`
— and returns the `num_ctx`, the bound and a report (parameters, weights,
per-token KV, machine budget); `apply: true` writes it to the session.

```console
# estimate only -- never loads the model; `apply` writes it to the session
$ curl -s http://127.0.0.1:11440/v1/agent/fit \
    -H 'Content-Type: application/json' \
    -d '{"session_id": "ses_…", "mode": "gpu", "apply": true}'
{"object": "fit", "model": "qwen3:8b", "mode": "gpu", "num_ctx": 131072,
 "source": "estimate", "bound": "max_gpu", "gpu_layers": null, "applied": true,
 "report": {"weights_bytes": …, "kv_per_token": …, "free_bytes": …, …}}

# verify by loading: an asynchronous probe job (202), then poll it
$ curl -s http://127.0.0.1:11440/v1/agent/fits \
    -H 'Content-Type: application/json' \
    -d '{"session_id": "ses_…", "mode": "cpu", "cpu_percent": 80}'
{"id": "fit_…", "object": "fit", "status": "queued", "session_id": "ses_…",
 "mode": "cpu", "probe": true, "apply": false, "warnings": [],
 "status_url": "/v1/agent/fits/fit_…",
 "cancel_url": "/v1/agent/fits/fit_…/cancel"}

$ curl -s http://127.0.0.1:11440/v1/agent/fits/fit_…
{"id": "fit_…", "object": "fit", "status": "completed",
 "progress": {"stage": "probe", "mode": "cpu", "ctx": 65536},
 "applied": false, "result": {"model": "qwen3:8b", "mode": "cpu",
 "num_ctx": 65536, "source": "probe", "bound": "max_cpu", …}}
```

`POST /v1/agent/fits` starts the same fit as an asynchronous probe job that
verifies the boundary by actually loading the model: `202` with the job id
and warnings (other resident models or active turns make it slower; prior
measurements on this host make it faster), `GET /v1/agent/fits/{id}` for
status/progress/result, `POST .../cancel` to stop it. A job with a session
holds it — turns, edits and a second fit get `409 session_busy` with
`active_fit_id` — a global cap (`serve_max_active_fits`) answers `429`, and
completed jobs live until `serve_fit_ttl`.

Turn `settings.num_ctx` also accepts `"max-gpu"` / `"max-cpu"` (with
`settings.cpu_percent`): the turn estimates at start — never a blocking probe
— persists the fitted context and reports it as a typed `session.fit` event
and `result.fit` on the turn status; a failed fit leaves the context
unchanged.

`serve_fit` gates the whole surface: `"probe"` (default), `"estimate"` (no
probing) or `"off"`.

The CLI equivalents are in [Context and placement](../../cli/context/overview.md).
