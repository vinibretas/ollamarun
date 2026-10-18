# Context fits

Context and placement over the API, at two speeds: a synchronous estimate
and an asynchronous probe job.

## Estimate — `POST /v1/agent/fit`

Body: `{model?, session_id?, mode: "gpu"|"cpu", cpu_percent, gpu_layers?,
apply?}`. No model loads; the same arithmetic the CLI's
[`max-gpu`](../../cli/context/max-gpu.md)/[`max-cpu`](../../cli/context/max-cpu.md)
use. `apply: true` writes the result to the session.

```console
$ curl -s http://127.0.0.1:11440/v1/agent/fit -H 'Content-Type: application/json' \
    -d '{"model": "qwen3:0.6b", "mode": "gpu"}'
```

```json
{
  "model": "qwen3:0.6b", "mode": "gpu", "num_ctx": 22528,
  "source": "estimate", "bound": "max_gpu", "gpu_layers": null, "applied": false,
  "report": {
    "weights_bytes": 522653767, "weights_source": "/api/tags size",
    "kv_per_token": 114688, "kv_bytes_at_ctx": 2583691264,
    "free_bytes": 4220518400, "usable_bytes": 3146776576,
    "margin_bytes": 1073741824, "overhead_bytes": 0, "credited_bytes": 0,
    "total_layers": 29, "model_max": 40960, "capped_by_model": false,
    "ctx_no_margin": 31744, "calibrated": false,
    "devices": [{"total_bytes": 4096, "free_bytes": 4025}],
    "notes": ["all layers are placed on the GPU",
              "weights split: 103.0 MiB token embeddings + …"]
  },
  "object": "fit"
}
```

The `report` is the full working: weights, per-token KV, the usable budget
after the margin, what was credited back from residents, the weights split
and any notes. `bound` names which constraint chose the number
(`max_gpu`/`max_cpu`); `capped_by_model` says the model's own maximum was the
binding constraint.

When the machine cannot be read — no GPU probe, unreachable ssh — the
estimate fails structurally rather than inventing a number:

```json
{"error": {"message": "cannot read VRAM on http://100.107.146.36:11434",
 "type": "invalid_request_error", "param": null, "code": "fit_failed"}}   # 400
```

## Probe job — `POST /v1/agent/fits`

The same fit, verified by actually loading the model at candidate contexts
(the API version of `--probe`). `202` with the job id and warnings:

```json
{"id": "fit_bc6e2228c7af", "object": "fit", "status": "queued", "model": "qwen3:0.6b",
 "mode": "gpu", "probe": true, "apply": false, "warnings": [],
 "status_url": "/v1/agent/fits/fit_bc6e2228c7af",
 "cancel_url": "/v1/agent/fits/fit_bc6e2228c7af/cancel"}
```

`GET /v1/agent/fits/{id}` reports progress and then the result — the live
run, condensed:

```text
{"status": "running", "progress": {"stage": "machine", "mode": "gpu"}}
{"status": "running", "progress": {"stage": "probe", "mode": "gpu", "ctx": 22784, "attempt": 2, "max": 10}}
{"status": "running", "progress": {"stage": "probe", "mode": "gpu", "ctx": 22528, "attempt": 3, "max": 10}}
{"status": "completed", "progress": {"stage": "probe", "mode": "gpu", "ctx": 22272, "attempt": 4, "max": 10}}
```

```json
"result": {
  "model": "qwen3:0.6b", "mode": "gpu", "num_ctx": 22272,
  "source": "probe", "bound": "max_gpu",
  "report": {"kv_per_token": 114688, "kv_bytes_at_ctx": 2466250752, …}
}
```

The estimate said 22 528, the probe measured **22 272** — that margin is the
difference between arithmetic and a loaded runner. `POST
/v1/agent/fits/{id}/cancel` stops a job (progress and status reflect it);
completed jobs live until `serve_fit_ttl` (default 24 h). A job with a
`session_id` holds that session: turns, edits and a second fit get
`409 session_busy` with `active_fit_id`, and the global cap
(`serve_max_active_fits`, default 2) answers `429`.

## Turn-start fits

A turn may ask for a fit instead of a number: `"settings": {"num_ctx":
"max-gpu"}` (or `"max-cpu"` with `settings.cpu_percent`). The turn estimates
**at start — never a blocking probe**, persists the fitted context and
reports it as a typed `session.fit` event plus `result.fit` on the status:

```json
"result": {"content": "pong", "fit": {"model": "qwen3:0.6b", "mode": "gpu",
 "num_ctx": 22528, "source": "estimate", "bound": "max_gpu", "report": {…}}}
```

A failed fit leaves the context unchanged and reports why; the turn proceeds
either way.

## Gates

`serve_fit` decides what exists: `"probe"` (default) allows both endpoints,
`"estimate"` refuses probe jobs, `"off"` refuses the whole fit surface. The
same switch guards turn-start fits. See
[Config file](../../config/file.md#serve-keys).
