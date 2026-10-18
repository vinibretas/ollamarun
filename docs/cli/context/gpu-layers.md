# GPU layers

`--gpu-layers SPEC` / `/gpu-layers` sets how many layers run on the GPU: `N`,
`P%` of the model's layers (`block_count` from `/api/show`), `+N`/`-N`
adjustments (baseline: the current value, else a full offload), `all`, `cpu`
or `auto`. It is stored in the session, re-resolved per model, wins over
`/set parameter num_gpu`, and unloads a resident model so the next request
reloads with the new placement.

The API exposes the same field per session — see
[Context fits](../../api/agent/fits.md).
