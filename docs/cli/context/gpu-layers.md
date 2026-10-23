# GPU layers

`/gpu-layers` pins how many of the model's layers run on the GPU; the rest
go to the CPU. It is stored on the session, re-resolved per model, and wins
over a raw `num_gpu` option.

## The spec

| Spec | Meaning |
| --- | --- |
| `N` | An absolute count. |
| `P%` | A percentage of the model's layers — `block_count` from `/api/show` (29 for `qwen3:0.6b` below). |
| `+N` / `-N` | Adjust the current value (baseline: the session's current spec, else a full offload). |
| `all` | Every layer on the GPU. |
| `cpu` | No layers on the GPU. |
| `auto` | Back to Ollama's own decision. |

```text
qwen3:0.6b › /gpu-layers 50%
gpu layers 50% → 15 of 29 layers
unloaded qwen3:0.6b
loading qwen3:0.6b…
loaded at 8192 (loaded runner)
qwen3:0.6b › /gpu-layers
gpu layers 50% → 15 of 29 layers
N (absolute), P% (of the model's layers), +N/-N (adjust the current value), all, cpu or auto
qwen3:0.6b › /gpu-layers auto
gpu layers auto (ollama decides)
```

Three things worth noting in that transcript:

- The percentage resolves against **this model's** layer count (29) at
  command time, so switching models later re-resolves the spec for the new
  model instead of reusing a stale number.
- Changing the placement **applies immediately**: `num_gpu` is a load-time
  property, so a resident model is unloaded and loaded again with the new
  split (a placement-only change reloads even when `num_ctx` did not move),
  and the command reports the window the runner came back with.
- `/gpu-layers` with no argument shows the current value plus the accepted
  forms.

## Where it is used

- Launch flag: `--gpu-layers SPEC`; config key `gpu_layers` seeds it.
- Session field: it persists across `--load`/`-c`, and the
  [context fits](max-gpu.md) honor it — when a placement is pinned, the
  output layer and the topmost blocks are counted as offloaded, not the
  whole model.
- API: `PATCH /v1/sessions/{id}` and turn `settings` accept `gpu_layers`
  with the same spec and the same unload behavior —
  see [Sessions API](../../api/agent/sessions.md).
- `/params` reports it as `gpu layers` (resolved label or `auto (ollama
  decides)`).

A pinned spec beats `/set parameter num_gpu` in the payload: the session
setting is the managed path, and the resolved number is what actually goes
into the request's `options`.
