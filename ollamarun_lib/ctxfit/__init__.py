"""Context-window sizing: `/num-ctx --max-gpu [--probe]` and
`/num-ctx --max-cpu N [--probe]`.

The package is the pipeline, one module per stage: `defaults.py` (shared
numbers and the calibration path), `quantization.py` (labels and bytes per
parameter), `machine.py` (free VRAM/DRAM, locally or over one ssh probe),
`profile.py` (GGUF KV geometry, the weight split and the quantization
cross-check), `calibration.py` (measured samples scoped to model/host/
placement), `estimate.py` (the analytic GPU and CPU fits), `probe.py`
(load-and-read-back verification), `report.py` (the rich reports) and
`api.py` (`fit_context()`, `max_gpu_context()`/`max_cpu_context()` and the
argument parsers the REPL command and the CLI flag call). `models.py` holds
the dataclasses they exchange; this module re-exports the public surface.

`--max-gpu` looks for the largest `num_ctx` that still keeps the model's GPU
placement (the whole model, or the pinned `/gpu-layers`), so the next load
runs at the chosen context without a partial CPU offload.

`--max-cpu N` raises the context until the model's host RAM reaches N% of the
DRAM available to it: `/proc/meminfo`'s MemAvailable (or one ssh probe for
`--remote` hosts) plus the RAM the loaded model already holds -- read back
from `/api/ps` as `size - size_vram`, since it is released when the model
reloads at the new context -- less an overhead allowance
(`max_cpu_overhead_mb`). Some layers usually stay on the GPU: a pinned
`/gpu-layers` is honoured, otherwise llama.cpp's fit picks the split, and
only the host side counts toward the target.

What the estimate cannot see is llama.cpp's measured buffers, so `--probe`
verifies empirically: it loads the model at a candidate context and reads
`/api/ps` back -- for `--max-gpu` the split (`size` vs `size_vram` -> the
real GPU percentage), for `--max-cpu` the host footprint (`size - size_vram`,
which matches what the machine's MemAvailable gave up in steady state).
Valid measurements are kept in a bounded cache at
`~/etc/data/ollamarun/ctxfit-calibration.json`, scoped to the model digest,
host, hardware, placement and KV settings; entries expire after 180 days.
"""
from __future__ import annotations

import types

from .api import (FitOutcome, fit_context, max_cpu_context, max_gpu_context,
                  parse_fill_percent, parse_num_ctx_arg,
                  resolve_num_ctx_delta)
from .calibration import CtxCalibrationStore
from .defaults import (CALIBRATION_MAX_AGE, CALIBRATION_MAX_ENTRIES,
                       CALIBRATION_MAX_SAMPLES, CTX_STEP,
                       CTXFIT_CALIBRATION_PATH, DEFAULT_CPU_OVERHEAD_MB,
                       DEFAULT_CPU_PROBES, DEFAULT_MARGIN_MB,
                       DEFAULT_OVERHEAD_MB, DEFAULT_PROBES, ESTIMATE_STEP, MIB,
                       SWA_CACHE_BATCH)
from .estimate import estimate_max_cpu_ctx, estimate_max_gpu_ctx, raw_kv_share
from .machine import (MachineGpu, MachineRam, machine_ram_for,
                      machine_state_for, probe_machine_gpu, probe_machine_ram)
from .models import (CpuFit, FitCancelled, FitRequest, GpuFit, Probe,
                     ProbeResult)
from .probe import probe_max_cpu_ctx, probe_max_gpu_ctx, wait_for_ps
from .profile import (KvLayerSpec, ModelProfile, kv_bytes_at,
                      kv_bytes_per_token, kv_element_bytes, model_profile,
                      model_total_layers, trained_context_length)
from .quantization import (FILE_TYPE_LABELS, KV_ELEMENT_BYTES, NUM_CTX_HELP,
                           QUANT_BYTES_PER_PARAM, quantization_bytes_per_param,
                           quantization_label)
from .report import (print_cpu_fit_report, print_cpu_probe_report,
                     print_fit_report, print_probe_report)

#: Public surface re-exported by `ollamarun_lib/__init__.py`, minus the
#: submodules themselves.
__all__ = [name for name, value in globals().items()
           if not name.startswith("_") and not isinstance(value, types.ModuleType)]
