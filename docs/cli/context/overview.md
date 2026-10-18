# Context and placement

Two things decide how the model runs: the **context window** and its **GPU
placement**.

- [/num-ctx](num-ctx.md) sizes the window; [max-gpu](max-gpu.md) and
  [max-cpu](max-cpu.md) fit it to the machine, and [probes](probes.md) verify
  by loading.
- [/gpu-layers](gpu-layers.md) pins how many layers run on the GPU.
- `/context` reports window usage, and the
  [footer](../rendering/footer.md)'s ctx counter ticks live — its
  denominator is the window the model is actually loaded with (`/api/ps`; a
  session `num_ctx` wins, the architectural maximum is only the fallback).
- [Compaction](compaction.md) keeps long sessions inside the window.

The CLI takes the same as `--num-ctx`, `--gpu-layers` and `--probe`; the API
exposes it per session — see [Context fits](../../api/agent/fits.md).
