# num-ctx

`/num-ctx N` sets the session's context window; `--num-ctx N` does it for a
run. The value is capped to the model's trained maximum with a warning:

```text
qwen3:0.6b › /num-ctx 8192
num_ctx set to 8192
qwen3:0.6b › /num-ctx 99999999
99999999 exceeds qwen3:0.6b's max context (40960) — capped to 40960
num_ctx set to 40960
```

The maximum comes from `/api/show`'s model metadata and is cached per model
for the run. The window is a load-time property, so the change is applied to
the runner **right away**: a resident model is unloaded and loaded again at
the new context (a model that is not loaded yet is loaded now), and the
window is read back from `/api/ps` to confirm it:

```text
qwen3:0.6b › /num-ctx 8192
num_ctx set to 8192
unloaded qwen3:0.6b
loading qwen3:0.6b at 8,192 ctx…
loaded at 8192 (loaded runner)
```

Repeating a window the runner already uses does nothing (`already loaded at
8192`), and `--num-ctx N` on the command line is in force from the first
load, since the launch preload carries the session's settings.

## Relative adjustments

Instead of an absolute size, `+N`/`-N` add or remove that many tokens from
the window in force, and `+P%`/`-P%` grow or shrink it by a share of itself,
so on a `8192` window `+50%` gives `12288` and `-50%` gives `4096`:

```text
qwen3:0.6b › /num-ctx +50%
8192 +50% → 12288
num_ctx set to 12288
```

A decrease cannot exceed `100%` (it floors at one token); an increase is
unlimited, still capped to the model's trained maximum with the same warning.
The baseline is the window actually in force — a session `num_ctx`, else the
loaded runner's window, else the model's own default, else the service's
`OLLAMA_CONTEXT_LENGTH` — so a relative change works on the first turn, before
`/api/ps` reports anything. The amount is resolved immediately and stored as
an absolute number, so repeated calls compound off the current window. The CLI
mirrors every form: `--num-ctx +1000`, `--num-ctx +50%` and, because argparse
would read a leading minus as an option, `--num-ctx=-50%`.

## The two fit forms

`/num-ctx` also accepts the calculated forms, which pick the number for you:

| Form | Picks |
| --- | --- |
| `--max-gpu` | The largest context that keeps the model fully on the GPU ([details](max-gpu.md)). |
| `--max-cpu N` | The largest context where the model's host RAM fills N% of available DRAM ([details](max-cpu.md)). |
| Either with `--probe` | Same, but verified by actually loading the model ([details](probes.md)). |

The CLI mirrors them: `--num-ctx max-gpu`, `--num-ctx --max-cpu N`, and
`--probe` as a modifier.

## What "effective" means

The session value is the first of, in order: session `num_ctx` → the loaded
runner's window from `/api/ps` → the model's default from `/api/show`.
Setting `num_ctx` here is therefore authoritative until you change it; the
[`/context` table](overview.md#seeing-where-you-stand) and the
[footer counter](../rendering/footer.md) always show which window is really
in force.

```text
qwen3:0.6b › /context
│ context length │ 9216 │
```

**The raw option is not the managed path.** `/set parameter num_ctx 8192`
does land in the request's `options`, but it skips the cap warning, the
reload coordination and the effective-window bookkeeping — prefer
`/num-ctx`.
