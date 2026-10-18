# Probes and calibration

`--probe` turns an estimate into a measurement: candidates are actually
loaded and each real GPU split is read back from `/api/ps`, solving the
measured bytes-per-token line for the capacity. Both [max-gpu](max-gpu.md) and
[max-cpu](max-cpu.md) take it (`/num-ctx --max-gpu --probe`,
`/num-ctx --max-cpu N --probe`).

Resident and probe measurements calibrate future estimates using a
conservative per-token slope and fixed overhead; the profile formula remains
the fallback. The bounded cache is scoped to model digest, host, hardware,
placement and KV settings, expires after 180 days, and is stored at
`~/etc/data/ollamarun/ctxfit-calibration.json`.

Config knobs: `max_gpu_margin_mb` (1 GiB), `max_gpu_overhead_mb` (0; the fit
margin already covers llama.cpp's projected buffers), `max_gpu_probes` (10),
`max_cpu_overhead_mb` (256 MiB) and `max_cpu_probes` (10).
