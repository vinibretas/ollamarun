# max-cpu

`--max-cpu N` fills DRAM instead: it raises the context until the model's host
RAM reaches N% (1-100) of the DRAM available to it — `/proc/meminfo`'s
MemAvailable on the Ollama machine (one ssh probe for `--remote` hosts) plus
the RAM the loaded model already holds (`/api/ps` `size - size_vram`, back
when it reloads), less a `max_cpu_overhead_mb` allowance. The same per-layer
KV, sliding-window, recurrent-state and weight-split profile decides the
placement: some layers may stay on the GPU unless
[gpu-layers](gpu-layers.md) is `cpu`; a pinned layer count is honoured and
only the host side counts.

`--probe` solves the measured footprint line the same way and never chooses a
context above the target — see [Probes and calibration](probes.md).
Compatible probe/resident data calibrates future CPU estimates as well;
automatic GPU placement is checked before a cached slope is applied.

From the REPL: `/num-ctx --max-cpu N [--probe]`.
