"""Shared numeric defaults for the context fits."""
from __future__ import annotations

from typing import Any

from ..constants import DATA_DIR

MIB = 1024 * 1024

DEFAULT_MARGIN_MB = 1024      # llama.cpp's --fit-target default, per device
DEFAULT_OVERHEAD_MB = 0       # extra reserve beyond the fit margin (llama.cpp's
                              # own fit projection already counts its buffers)
DEFAULT_CPU_OVERHEAD_MB = 256  # CPU-side buffers the --max-cpu estimate can't read
DEFAULT_PROBES = 10
DEFAULT_CPU_PROBES = 10
CTX_STEP = 256                # llama.cpp aligns fit-reduced context to 256 * streams
ESTIMATE_STEP = 1024          # the estimate rounds down to readable numbers
SWA_CACHE_BATCH = 1024        # cells llama.cpp adds past a sliding window for batching
CTXFIT_CALIBRATION_PATH = DATA_DIR / "ctxfit-calibration.json"
CALIBRATION_MAX_AGE = 180 * 24 * 60 * 60
CALIBRATION_MAX_ENTRIES = 256
CALIBRATION_MAX_SAMPLES = 16


def _align_down(value: int, step: int) -> int:
    return (value // step) * step


def _config_int(config: dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(config.get(key))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default
