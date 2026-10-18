# max-gpu

`--max-gpu` picks the largest context that still keeps the model fully on the
GPU. The model is profiled first: `/api/show`'s GGUF metadata supplies the
per-layer KV geometry (hybrid attention, sliding windows and their per-layer
key/value dims, MLA's compressed latent, recurrent `ssm.*` state) and the
parameter count, `/api/tags` supplies the on-disk size and quantization, and
the two are cross-checked (`params × bytes/param`; the model row in the report
names all of it). Free VRAM (nvidia-smi, locally or over one ssh probe for
`--remote` machines) is then reduced by llama.cpp's fit margin
(`max_gpu_margin_mb`, 1 GiB by default); the separate `max_gpu_overhead_mb`
reserve defaults to 0 because the fit margin already covers the buffers
llama.cpp projects. Resident models' VRAM is credited back (it returns when
they reload at the new context), and a pinned
[gpu-layers](gpu-layers.md) is honoured: the output layer and the topmost
blocks are counted as the offloaded share, and the non-block weights (token
embeddings, output layer, vision projector) are estimated rather than split
evenly.

`--probe` then verifies by loading candidates and reading each real GPU split
back from `/api/ps`, solving the measured bytes-per-token line for the
capacity, and leaves the model warm at the largest 100%-GPU context — see
[Probes and calibration](probes.md).

From the REPL: `/num-ctx --max-gpu [--probe]`.
