# Probes and calibration

The two fits ([max-gpu](max-gpu.md), [max-cpu](max-cpu.md)) are estimates:
they combine the model profile with machine numbers but never load anything.
`--probe` is the verification pass — load the model at candidate contexts,
read what the machine actually reported, and pick from measurements instead
of arithmetic.

```console
smollm2:360m › /num-ctx --max-gpu --probe
probe 1: smollm2:360m at num_ctx 8,192…
          context fit · smollm2:360m on this machine
╭────────────┬────────────────────────────────────────────────╮
│ model      │ 0.36B params · F16 · llama · 692.0 MiB         │
│ gpu memory │ 1.0 GiB free + 2.8 GiB resident → 3.8 GiB pool │
│ margin     │ 1.0 GiB per device → 2.8 GiB usable            │
│ weights    │ 692.0 MiB                                      │
│ kv cache   │ 40.0 KiB per token · 320.0 MiB at 8,192        │
│ model max  │ 8,192 tokens                                   │
╰────────────┴────────────────────────────────────────────────╯
· crediting smollm2:360m (215.5 MiB on the GPU) — it reloads at the new context and releases that memory
· crediting llama3.2:3b (2.6 GiB on the GPU) — it reloads at the new context and releases that memory
num_ctx 8,192 (capped by the model's max context)
               context probe · smollm2:360m
╭─────────┬───────────┬───────────┬─────┬─────────────────╮
│ num_ctx │      size │      vram │ gpu │ result          │
├─────────┼───────────┼───────────┼─────┼─────────────────┤
│   8,192 │ 633.8 MiB │ 215.5 MiB │ 34% │ partial offload │
╰─────────┴───────────┴───────────┴─────┴─────────────────╯
no probed context stayed fully on the GPU (best 34% at num_ctx 8,192)
```

That run also shows the honest side of probing: the *estimate* assumed the
two resident models would be evicted on reload and credited their VRAM back,
but the probe loaded while they were still resident, so the model only got
34% of its layers onto the GPU. The probe reports what happened instead of
the estimate's assumption.

What each probe does:

1. **Load** the model at one candidate context (`/api/chat` with the
   candidate `num_ctx` and a tiny generation, `keep_alive` from the
   session).
2. **Read back** the real split — `/api/ps` reports the bytes on the GPU vs
   the rest, for the running model.
3. **Solve the line.** A model's footprint is affine in the context size:
   `bytes = weights + kv_per_token × ctx`. Two measured points are enough to
   solve for the slope; the fit then computes the capacity directly instead
   of bisecting the candidate range.
4. **Finish warm.** For `--max-gpu` the model is left loaded at the largest
   100%-GPU context; for `--max-cpu` never above the requested target.

## When the estimate and the probe disagree

The estimate can be conservative — the fit margin is deliberately generous.
If the estimate declares a context infeasible but a probe might still make
it, the command says so and keeps probing:

```text
probing anyway — the estimate's buffers may be conservative
```

`max_gpu_probes` / `max_cpu_probes` (default 10 each) bound how many loads a
probe run may spend. A probe that cannot find a feasible context leaves the
session untouched and reports the reason instead of guessing.

## Calibration

Every good measurement — probe results **and** ordinary resident-model
readings — is recorded in a bounded cache at
`~/etc/data/ollamarun/ctxfit-calibration.json`:

- **Scoped** to the model digest, host, hardware, placement and KV settings,
  so data never leaks between machines or configurations.
- **Expires** after 180 days; incompatible or stale data is ignored, never
  trusted blindly.
- **Feeds back**: later estimates start from the measured slope (the fit
  report says `calibrated from N measured context point(s)`) and probe
  searches start near the previous answer, so the second run is faster than
  the first.

The profile formula remains the fallback whenever the cache is empty — the
calibration changes how *confident* the number is, not whether you get one.

## The same thing over the API

`POST /v1/agent/fits` runs a probe as an asynchronous job (202 + warnings,
progress, cancel) and `POST /v1/agent/fit` the synchronous estimate — see
[Context fits](../../api/agent/fits.md). Turn `settings.num_ctx:
"max-gpu"|"max-cpu"` always estimates (never a blocking probe) and reports
the result as a `session.fit` event.
