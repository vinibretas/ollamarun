# max-cpu

`/num-ctx --max-cpu N` is the other direction: instead of keeping the model
on the GPU, it raises the context window until the model's **host RAM** fills
N% (1-100) of the DRAM available to it. The GPU keeps whatever layers the
budget allows; the rest runs on the CPU.

```text
llama3.2:3b › /num-ctx --max-cpu 80
          context fit · llama3.2:3b on this machine
╭────────────────┬───────────────────────────────────────────╮
│ model          │ 3.21B params · Q4_K_M · llama · 1.9 GiB   │
│ machine memory │ 31.2 GiB total · 16.0 GiB available       │
│ model now      │ 6.8 GiB of host RAM (back on reload)      │
│ budget         │ 22.8 GiB = available + the resident model │
│ target         │ 18.2 GiB = 80% of the budget              │
│ placement      │ gpu-layers ~1 of 29 (fitted)              │
│ cpu weights    │ 1.7 GiB of 1.9 GiB                        │
│ kv cache       │ 112.0 KiB per token · 14.0 GiB at 131,072 │
│ overhead       │ 256.0 MiB reserved                        │
│ model max      │ 131,072 tokens                            │
╰────────────────┴───────────────────────────────────────────╯
· the loaded model holds 6.8 GiB of host RAM -- counted back into the budget, a reload returns it
· calibrated from 1 measured context point(s): 112.0 KiB per token
· weights split: 236.2 MiB token embeddings + 236.2 MiB output layer (separate) + 1.4 GiB blocks
num_ctx 131,072 (capped by the model's max context)
· at the model's max the host holds 15.9 GiB = 69% of the budget (80% requested)
estimate only — /num-ctx --max-cpu --probe loads the model and verifies the real host RAM
num_ctx set to 131072
```

Reading that report:

- **budget** = MemAvailable (`/proc/meminfo` on the Ollama machine, one ssh
  probe for `--remote`) **plus** the RAM the loaded model already holds — a
  reload returns it.
- **target** = N% of the budget. If the target sits beyond the model's
  maximum context, the cap binds and the note at the bottom says by how
  much (`69% of the budget (80% requested)`).
- **cpu weights** = the share of the weights that will live in host RAM,
  given the placement the fit chose.
- **kv cache** = per-token KV (the same profile [max-gpu](max-gpu.md) uses)
  times the candidate context.
- **overhead** = `max_cpu_overhead_mb` (default 256 MiB) reserved for
  everything that is not weights or KV.

A pinned [`/gpu-layers`](gpu-layers.md) is honored: with `cpu` every layer
counts on the host side; with a fixed N, only the host share does. Without a
pin the fit itself decides how many layers stay on the GPU (the
`gpu-layers ~1 of 29 (fitted)` line above) before sizing the CPU side.

## Config and verification

| Knob | Default | Meaning |
| --- | --- | --- |
| `max_cpu_overhead_mb` | 256 | Host-RAM allowance beyond weights + KV. |
| `max_cpu_probes` | 10 | Load attempts a `--probe` may spend. |

CLI: `--num-ctx --max-cpu N` (with `--probe` to verify by loading — see
[Probes and calibration](probes.md)). Like the GPU fit, this is an estimate:
it is meant to pick a sensible starting window, and the probe exists for when
close counts.
