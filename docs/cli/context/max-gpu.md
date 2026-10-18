# max-gpu

`/num-ctx --max-gpu` answers one question: **what is the largest context
window that still keeps every layer of this model on the GPU?** It profiles
the model, measures the machine, does the arithmetic, and writes the result
to the session.

```text
llama3.2:3b › /num-ctx --max-gpu
        context fit · llama3.2:3b on this machine
╭────────────┬───────────────────────────────────────────╮
│ model      │ 3.21B params · Q4_K_M · llama · 1.9 GiB   │
│ gpu memory │ 3.9 GiB free                              │
│ margin     │ 1.0 GiB per device → 2.9 GiB usable       │
│ weights    │ 1.9 GiB                                   │
│ kv cache   │ 112.0 KiB per token · 1008.0 MiB at 9,216 │
│ model max  │ 131,072 tokens                            │
╰────────────┴───────────────────────────────────────────╯
· all layers are placed on the GPU
· weights split: 236.2 MiB token embeddings + 236.2 MiB output layer (separate) + 1.4 GiB blocks
num_ctx 9,216
estimate only — /num-ctx --max-gpu --probe loads the model and verifies the real GPU split
num_ctx set to 9216
```

That run is from a 4 GB GTX 1050 with a 1.9 GiB Q4 model: 2.9 GiB usable
after llama.cpp's fit margin, 1.9 GiB of weights, 112 KiB of KV per token —
and 9,216 tokens is what fits. `/context` then reports `context length 9216`.

## How the number is computed

1. **Profile the model.** `/api/show`'s GGUF metadata supplies the per-layer
   KV geometry — key/value head counts, sliding-window settings, MLA's
   compressed latent, recurrent `ssm.*` state — plus the architecture and
   parameter count; `/api/tags` supplies the on-disk size and quantization.
   The two are cross-checked (`params × bytes/param`), and the weights split
   into token embeddings, output layer and blocks instead of being treated as
   one lump (the report's last line).
2. **Measure the machine.** Free VRAM from `nvidia-smi` — locally, or over
   one ssh probe for a `--remote` target.
3. **Subtract the fit margin.** `max_gpu_margin_mb` (default 1 GiB) is
   llama.cpp's own allowance for projected buffers; `max_gpu_overhead_mb`
   (default 0) adds an extra reserve if your stack needs one.
4. **Credit residents.** Other loaded models' VRAM is counted back in — it
   returns when they reload at the new context.
5. **Honor a pinned placement.** With `/gpu-layers` set, the output layer
   and the topmost blocks are counted as offloaded, not the whole model.

The result is the largest context whose KV cache plus weights fit in the
usable VRAM.

## Verifying with a probe

The report ends by saying so: this is an estimate. `--probe` loads the model
at candidate contexts, reads the real GPU split from `/api/ps`, solves the
measured bytes-per-token line and leaves the model warm at the largest
100%-GPU context. See [Probes and calibration](probes.md).

## Configuration and CLI forms

| Knob | Default | Meaning |
| --- | --- | --- |
| `max_gpu_margin_mb` | 1024 | llama.cpp's fit margin, subtracted from free VRAM. |
| `max_gpu_overhead_mb` | 0 | Extra reserve beyond the margin. |
| `max_gpu_probes` | 10 | Load attempts a probe may spend. |

CLI: `--num-ctx max-gpu` (with `--probe` to verify). The result is stored on
the session and wins over a raw `num_ctx` option — and a pinned
[`/gpu-layers`](gpu-layers.md) is respected when the layers are counted.

If the estimate can't be trusted (no GPU probe available, exotic metadata),
the command says so instead of inventing a number; `/num-ctx N` remains the
fallback, and the [CPU fit](max-cpu.md) is the other direction.
