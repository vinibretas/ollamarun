"""Quantization labels and bytes-per-parameter for the weight cross-check."""
from __future__ import annotations

from typing import Any, Optional


def _model_info(show_data: dict[str, Any]) -> dict[str, Any]:
    return show_data.get("model_info") or {}


KV_ELEMENT_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,   # 34 bytes / 32 values
    "q5_1": 0.75,     # 24 / 32
    "q5_0": 0.6875,   # 22 / 32
    "q4_1": 0.625,    # 20 / 32
    "q4_0": 0.5625,   # 18 / 32
}

# `general.file_type` (llama.cpp's llama_ftype enum) -> quantization label.
FILE_TYPE_LABELS = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4", 39: "NVFP4",
}

# Typical stored bytes per parameter: block-quantized tensors plus overhead.
QUANT_BYTES_PER_PARAM = {
    "F32": 4.0, "F16": 2.0, "BF16": 2.0, "FP8": 1.0, "FP16": 2.0,
    "Q8_0": 1.0625, "Q6_K": 0.8203, "Q5_K_M": 0.70, "Q5_K_S": 0.68,
    "Q5_1": 0.75, "Q5_0": 0.6875, "Q4_K_M": 0.61, "Q4_K_S": 0.58,
    "Q4_1": 0.625, "Q4_0": 0.5625, "Q3_K_L": 0.54, "Q3_K_M": 0.50,
    "Q3_K_S": 0.44, "Q2_K": 0.33, "IQ4_XS": 0.545, "IQ4_NL": 0.5625,
    "IQ3_M": 0.50, "IQ3_S": 0.44, "IQ3_XS": 0.42, "IQ3_XXS": 0.36,
    "IQ2_M": 0.33, "IQ2_S": 0.30, "IQ2_XS": 0.28, "IQ2_XXS": 0.25,
    "IQ1_M": 0.23, "IQ1_S": 0.19, "MXFP4": 0.53, "NVFP4": 0.53,
    "Q2_K_S": 0.30, "TQ1_0": 0.19, "TQ2_0": 0.21,
}


def quantization_label(show_data: dict[str, Any],
                       entry: Optional[dict[str, Any]] = None) -> str:
    """Human label for a model's quantization (tags first, GGUF enum after)."""
    details = (entry or {}).get("details") or {}
    label = str(details.get("quantization_level") or "").strip().upper()
    if label:
        return label
    file_type = _model_info(show_data).get("general.file_type")
    if isinstance(file_type, int):
        return FILE_TYPE_LABELS.get(file_type, f"file_type {file_type}")
    return ""


def quantization_bytes_per_param(label: str) -> Optional[float]:
    """Typical stored bytes per parameter for a quantization label, or None."""
    text = (label or "").strip().upper().replace("-", "_")
    if text in QUANT_BYTES_PER_PARAM:
        return QUANT_BYTES_PER_PARAM[text]
    for prefix in sorted(QUANT_BYTES_PER_PARAM, key=len, reverse=True):
        if text.startswith(prefix):
            return QUANT_BYTES_PER_PARAM[prefix]
    return None


NUM_CTX_HELP = """usage: /num-ctx N | +N | -N | +P% | -P% | --max-gpu [--probe] | --max-cpu N [--probe]
  N          set the context window (capped to the model's max)
  +N / -N    add or remove N tokens from the window in force
  +P% / -P%  grow or shrink the window by P percent of itself (a decrease
             cannot exceed 100%). The baseline is the session num_ctx, else
             the loaded runner's window, else the model's own default, else
             the ollama service environment's OLLAMA_CONTEXT_LENGTH
  --max-gpu  use the largest context that still runs the model on the GPU:
             per-layer KV geometry from /api/show (hybrid layers, sliding
             windows, MLA compression and recurrent state included), the
             on-disk size (/api/tags) cross-checked against the model's
             parameter count and quantization, and the machine's free VRAM
             (an nvidia-smi probe, locally or over ssh for --remote hosts),
             minus llama.cpp's fit margin (LLAMA_ARG_FIT_TARGET, or
             max_gpu_margin_mb; it already covers the buffers the fit
             projects, so the extra max_gpu_overhead_mb reserve defaults to
             0). Models already resident are credited
             back -- their VRAM is released when they reload at the new
             context; max_gpu_credit_residents=false limits that credit to
             the model being sized. A pinned /gpu-layers is honoured, with
             the output layer and the topmost blocks (not an even split)
             counted as the offloaded share.
  --max-cpu N  raise the context until the model's host RAM reaches N% (1-100)
             of the DRAM available to it: /proc/meminfo's MemAvailable -- one
             ssh probe for --remote hosts -- plus the RAM the loaded model
             already holds (/api/ps size - size_vram; a reload returns it),
             less an overhead allowance (max_cpu_overhead_mb). Some layers
             stay on the GPU unless /gpu-layers is cpu; a pinned /gpu-layers
             is honoured and only the host side counts toward the target.
             The per-layer KV and recurrent-state model above applies here
             too.
  --probe    verify by loading: starting from the estimate, the model is
             warmed at candidate contexts and the real reading is taken back
             from /api/ps -- the GPU split for --max-gpu, the host RAM for
             --max-cpu -- until the boundary is bracketed
             (max_gpu_probes / max_cpu_probes, default 10). Successful probes
             calibrate later estimates; compatible cached measurements are
             reused, otherwise the profile estimate is used."""


