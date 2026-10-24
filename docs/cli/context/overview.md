# Context and placement

Two numbers decide how a model runs: the **context window** (how much
conversation it can hold) and its **GPU placement** (how many layers run on
the GPU versus the CPU). Ollama picks both automatically; this section is
about overriding them deliberately — and about the tools that calculate what
the machine can actually afford.

## The managed knobs

| Knob | Command / flag | What it sets |
| --- | --- | --- |
| Window size | `/num-ctx N`, `--num-ctx N` | The session's `num_ctx`, capped to the model's trained maximum. |
| Window adjustment | `/num-ctx +N`/`-N`/`+P%`/`-P%`, `--num-ctx …` | Add or remove tokens, or a share, from the window in force. |
| Largest GPU window | `/num-ctx --max-gpu`, `--num-ctx max-gpu` | Estimates the largest context that keeps the model fully on the GPU. |
| DRAM fill | `/num-ctx --max-cpu N`, `--num-ctx --max-cpu N` | Raises the window until the model's host RAM fills N% of available DRAM. |
| Verification | `--probe` on either | Loads the model at candidate contexts and reads the real split back. |
| Layer split | `/gpu-layers SPEC`, `--gpu-layers SPEC` | Pins how many layers run on the GPU (`N`, `P%`, `+N`/`-N`, `all`, `cpu`, `auto`). |

Everything is stored **on the session**, so `/load` restores it, and it wins
over the equivalent raw options (`/set parameter num_ctx` / `num_gpu`). Both
knobs are load-time properties, so the change reaches the runner with the
next message: the request carries the new `num_ctx`/`num_gpu` and Ollama
reloads the model at them.

## Seeing where you stand

`/context` reports the working numbers, colour-coded so the relevant ones
stand out (the window and its source, a count per message role, and the used
tokens/share/free space, which turn yellow at 70% and red at 90%):

```text
qwen3:0.6b › /context
             context
╭────────────────────────┬──────────────────────────────────────╮
│ field                  │                                value │
├────────────────────────┼──────────────────────────────────────┤
│ model                  │                           qwen3:0.6b │
│ context length         │ 4096 tok (OLLAMA_CONTEXT_LENGTH=4096) │
│ model max              │                               40960 tok │
│ messages               │                                    0 │
│ last prompt tokens     │                                    ? │
│ last completion tokens │                                    ? │
│ context used           │                                    ? │
╰────────────────────────┴──────────────────────────────────────╯
```

`context length` is the window the model will *actually run with*, resolved
in this order: the session's `num_ctx` → the loaded runner's window
(`/api/ps`) → the model's own Modelfile `num_ctx` (`/api/show`) → the ollama
service environment's `OLLAMA_CONTEXT_LENGTH`, read from the running unit
(locally, or over ssh for `--remote`). The parenthesised label is that
measured provenance — the setting's own name and value, never an invented
"default" — and when nothing can be measured the cell reads `not loaded yet`.
The model's trained maximum is a separate `model max` row, never the window.
The same value feeds the [footer's ctx counter](../rendering/footer.md) and
the warning below, and the [startup banner](../repl.md) shows it too.

## The 80% warning, and 85% compaction

When `context used / context length` reaches 80%, the REPL prints:

```text
context 13112/16384 tokens (80%) — /summarize to compact
```

and [compaction](compaction.md) takes over from there — manually with
`/summarize`, or automatically at 85% when config `auto_compact` is
`"auto"`. The same warning runs headlessly at turn start for the
[agent API](../../api/agent/turns.md), where it is a `turn.warning` event.

## Where each piece lives

- [num-ctx](num-ctx.md) — setting the window, and the cap.
- [max-gpu](max-gpu.md) / [max-cpu](max-cpu.md) — the two fits, with real
  reports from a 4 GB GPU.
- [Probes and calibration](probes.md) — verifying by loading, and the
  measured-data cache.
- [GPU layers](gpu-layers.md) — the placement spec.
- [Compaction](compaction.md) — keeping long sessions inside the window.
