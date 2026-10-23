"""The model profile: GGUF KV geometry, weights split and quantization facts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..helpers import human_bytes
from ..interactive import model_block_count, model_parameter_count
from .defaults import SWA_CACHE_BATCH
from .quantization import (FILE_TYPE_LABELS, KV_ELEMENT_BYTES,
                           QUANT_BYTES_PER_PARAM, _model_info,
                           quantization_bytes_per_param, quantization_label)


def _arch_prefix(info: dict[str, Any]) -> str:
    """The architecture prefix of the GGUF metadata keys ('' for bare names)."""
    for key in info:
        if key == "block_count":
            return ""
        if key.endswith(".block_count"):
            return key[: -len(".block_count")]
    return ""


def _arch_key(info: dict[str, Any], arch: str, suffix: str) -> Any:
    return info.get(f"{arch}.{suffix}" if arch else suffix)


def trained_context_length(show_data: dict[str, Any]) -> Optional[int]:
    """The model's trained context window (the real cap), ignoring parameter overrides."""
    for key, value in _model_info(show_data).items():
        if key.endswith(".context_length") and isinstance(value, int) and value > 0:
            return value
    return None


def model_total_layers(show_data: dict[str, Any]) -> Optional[int]:
    """Layers as `num_gpu` counts them: the repeating blocks plus the output layer."""
    blocks = model_block_count(show_data)
    return blocks + 1 if blocks else None


def kv_element_bytes(cache_type: Optional[str]) -> float:
    """Bytes per KV element for an OLLAMA_KV_CACHE_TYPE (f16 when unrecognized/unset)."""
    return KV_ELEMENT_BYTES.get((cache_type or "").strip().lower(), 2.0)


@dataclass
class KvLayerSpec:
    """One repeating block's cache geometry."""
    heads: int
    key_length: float
    value_length: float
    window: Optional[int] = None
    mla: bool = False

    def bytes_per_token(self, element_bytes: float) -> int:
        if self.mla:
            return max(1, int(round((self.key_length + self.value_length)
                                    * element_bytes)))
        if self.heads <= 0:
            return 0
        return max(1, int(round(self.heads * (self.key_length + self.value_length)
                                * element_bytes)))

    def cells(self, ctx: int, batch: int = SWA_CACHE_BATCH) -> int:
        if not self.window:
            return ctx
        return min(ctx, self.window + max(1, batch))


def _as_list(value: Any, blocks: int) -> Optional[list[Any]]:
    """A per-layer metadata list (or a scalar repeated for every block)."""
    if isinstance(value, list):
        if not value:
            return None
        return [value[index] if index < len(value) else value[-1]
                for index in range(blocks)]
    if isinstance(value, (int, float, bool)):
        return [value] * blocks
    return None


def _swa_flags(info: dict[str, Any], arch: str, blocks: int) -> tuple[list[bool], str]:
    """Per-block sliding-window flags plus a short description of the source."""
    layer_types = _arch_key(info, arch, "attention.layer_types")
    if isinstance(layer_types, list) and layer_types:
        flags = []
        for index in range(blocks):
            value = str(layer_types[index] if index < len(layer_types)
                        else layer_types[-1]).lower()
            flags.append("slid" in value)
        return flags, "layer_types"
    pattern = _arch_key(info, arch, "attention.sliding_window_pattern")
    if isinstance(pattern, list) and pattern:
        flags = []
        for index in range(blocks):
            value = pattern[index] if index < len(pattern) else pattern[-1]
            if isinstance(value, str):
                flags.append("slid" in value.lower())
            else:
                flags.append(bool(value))
        return flags, "sliding_window_pattern"
    if isinstance(pattern, int) and pattern > 1:
        # Every Nth layer is full attention; the rest use the window.
        return [((index + 1) % pattern != 0) for index in range(blocks)], \
            f"every {pattern}th layer full"
    if arch == "gemma3":
        return [((index + 1) % 6 != 0) for index in range(blocks)], \
            "gemma3 every 6th layer full"
    if arch == "gemma2":
        return [((index + 1) % 2 != 0) for index in range(blocks)], \
            "gemma2 every 2nd layer full"
    window = _arch_key(info, arch, "attention.sliding_window")
    return ([bool(window)] * blocks if _positive(window) else [False] * blocks), \
        ("all layers" if _positive(window) else "")


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def _vocab_size(info: dict[str, Any], arch: str) -> Optional[int]:
    """Vocabulary size from the GGUF: the explicit key, tokens, or token ids.

    Ollama elides the tokenizer arrays from `/api/show`, so models without an
    explicit `<arch>.vocab_size` still expose their special-token ids; the
    largest id + 1 is a close lower bound for the embedding/output rows.
    """
    explicit = _arch_key(info, arch, "vocab_size")
    if _positive(explicit):
        return int(explicit)
    tokens = info.get("tokenizer.ggml.tokens")
    if isinstance(tokens, list) and tokens:
        return len(tokens)
    ids = [int(value) for key, value in info.items()
           if key.startswith("tokenizer.ggml.") and key.endswith("_token_id")
           and isinstance(value, (int, float)) and not isinstance(value, bool)
           and value >= 0]
    if not ids:
        return None
    # sentencepiece models expose only tiny special ids here; a real
    # vocabulary starts well above them.
    guess = max(ids) + 1
    return guess if guess >= 1024 else None


def _layer_kv_specs(show_data: dict[str, Any]) -> tuple[Optional[list[KvLayerSpec]], str]:
    """Per-block KV geometry, or (None, reason) when the metadata cannot size it."""
    info = _model_info(show_data)
    arch = _arch_prefix(info)
    blocks = model_block_count(show_data)
    if not blocks or not arch:
        return None, "block count or architecture missing"
    if _arch_key(info, arch, "attention.causal") is False:
        return None, "the model has no causal attention KV cache"
    heads_raw = _arch_key(info, arch, "attention.head_count_kv")
    if heads_raw is None:
        # A missing GQA field means plain multi-head attention: every head
        # keeps a K/V pair. More useful than refusing to size the model.
        heads_raw = _arch_key(info, arch, "attention.head_count")
    if heads_raw is None and (
            _positive(_arch_key(info, arch, "ssm.state_size"))
            or _positive(_arch_key(info, arch, "shortconv.l_cache"))):
        heads_raw = 0  # pure recurrent/convolution model: no KV cache
    heads = _as_list(heads_raw, blocks)
    if heads is None:
        return None, "attention head count missing"
    try:
        heads = [max(0, int(value)) for value in heads
                 if isinstance(value, (int, float))]
    except (TypeError, ValueError):
        return None, "attention head count is not numeric"
    if len(heads) < blocks:
        return None, "attention head count list is short"
    heads = heads[:blocks]

    embedding = _arch_key(info, arch, "embedding_length")
    head_count = _arch_key(info, arch, "attention.head_count")
    key = _arch_key(info, arch, "attention.key_length")
    value = _arch_key(info, arch, "attention.value_length")
    key_swa = _arch_key(info, arch, "attention.key_length_swa")
    value_swa = _arch_key(info, arch, "attention.value_length_swa")
    key_default = key if _positive(key) else None
    value_default = value if _positive(value) else key_default
    mla = False
    if not key_default:
        # MLA caches one compressed latent plus a rope tail per layer; this
        # must win over the embedding/head-count fallback.
        lora = _arch_key(info, arch, "attention.kv_lora_rank")
        if _positive(lora):
            key_default = lora
            value_default = _arch_key(info, arch, "attention.qk_rope_head_dim")
            if not _positive(value_default):
                value_default = 0
            mla = True
    if not key_default and _positive(embedding) and _positive(head_count):
        key_default = value_default = embedding / head_count
    if not key_default:
        return None, "attention key/value dimensions missing"
    if not mla and not _positive(value_default):
        value_default = key_default

    window = _arch_key(info, arch, "attention.sliding_window")
    window = int(window) if _positive(window) else None
    swa, swa_source = _swa_flags(info, arch, blocks)
    key_swa = key_swa if _positive(key_swa) else key_default
    value_swa = value_swa if _positive(value_swa) else value_default

    specs: list[KvLayerSpec] = []
    for index in range(blocks):
        sliding = bool(window) and swa[index]
        specs.append(KvLayerSpec(
            heads=heads[index],
            key_length=float(key_swa if sliding else key_default),
            value_length=float(value_swa if sliding else value_default),
            window=window if sliding else None,
            mla=mla,
        ))
    note = ""
    if window:
        count = sum(1 for spec in specs if spec.window)
        note = (f"{count}/{blocks} layers use a {window:,}-token sliding window"
                f" ({swa_source}); the cache keeps window + batch cells")
    return specs, note


def kv_bytes_per_token(show_data: dict[str, Any], *, parallel: int = 1,
                       element_bytes: float = 2.0) -> Optional[int]:
    """KV-cache bytes per context token at full context, or None when unknown."""
    specs, _ = _layer_kv_specs(show_data)
    if not specs:
        return None
    per_token = sum(spec.bytes_per_token(element_bytes) for spec in specs)
    if not per_token:
        return None
    return max(1, per_token) * max(1, int(parallel))


def kv_bytes_at(show_data: dict[str, Any], ctx: int, *, parallel: int = 1,
                element_bytes: float = 2.0, batch: int = SWA_CACHE_BATCH,
                block_range: Optional[tuple[int, int]] = None) -> Optional[int]:
    """Exact KV-cache bytes at `ctx`, accounting for saturated sliding windows."""
    specs, _ = _layer_kv_specs(show_data)
    if not specs:
        return None
    start, end = block_range or (0, len(specs))
    start = max(0, min(len(specs), start))
    end = max(start, min(len(specs), end))
    total = sum(spec.cells(max(0, int(ctx)), batch)
                * spec.bytes_per_token(element_bytes)
                for spec in specs[start:end])
    return max(0, total) * max(1, int(parallel))


def _recurrent_bytes(show_data: dict[str, Any], *, parallel: int = 1) -> tuple[int, int, str]:
    """(bytes per sequence, recurrent layer count, note) for SSM/hybrid models."""
    info = _model_info(show_data)
    arch = _arch_prefix(info)
    blocks = model_block_count(show_data) or 0
    specs, _ = _layer_kv_specs(show_data)
    if not specs:
        return 0, 0, ""
    recurrent_layers = sum(1 for spec in specs if spec.heads <= 0)
    if not recurrent_layers:
        return 0, 0, ""
    inner = _arch_key(info, arch, "ssm.inner_size")
    state = _arch_key(info, arch, "ssm.state_size")
    conv = _arch_key(info, arch, "ssm.conv_kernel")
    if _positive(inner) and _positive(state):
        conv = int(conv) if _positive(conv) else 1
        # f32 recurrent state, plus the short-convolution state llama.cpp
        # keeps alongside it. The observed qwen35 layout is S = inner * state
        # per layer and R = inner * kernel per layer (rounded up by runtime).
        per_layer = int(inner) * int(state) * 4 + int(inner) * max(1, conv) * 4
        total = per_layer * recurrent_layers * max(1, int(parallel))
        return total, recurrent_layers, (
            f"{recurrent_layers}/{blocks} recurrent layers "
            f"(SSM state {human_bytes(total)})")
    shortconv = _arch_key(info, arch, "shortconv.l_cache")
    embedding = _arch_key(info, arch, "embedding_length")
    if _positive(shortconv) and _positive(embedding):
        # lfm2-style short convolutions: one f32 cache per channel.
        per_layer = int(embedding) * max(1, int(shortconv)) * 4
        total = per_layer * recurrent_layers * max(1, int(parallel))
        return total, recurrent_layers, (
            f"{recurrent_layers}/{blocks} short-convolution layers "
            f"(state {human_bytes(total)})")
    return 0, recurrent_layers, (
        f"{recurrent_layers} non-attention layers have no state metadata; "
        f"their fixed state is not sized")




@dataclass
class ModelProfile:
    """Everything the estimate can learn about the model before sizing it."""
    architecture: str = ""
    blocks: Optional[int] = None
    embedding_length: Optional[int] = None
    feed_forward_length: Optional[int] = None
    context_length: Optional[int] = None
    vocab_size: Optional[int] = None
    expert_count: Optional[int] = None
    expert_used: Optional[int] = None
    params: Optional[int] = None
    params_source: str = ""
    quantization: str = ""
    weights_bytes: int = 0
    weights_source: str = ""
    projector_bytes: int = 0
    projector_params: Optional[int] = None
    tied_embeddings: bool = False
    embedding_bytes: int = 0
    output_bytes: int = 0
    block_bytes: int = 0
    split_estimate: bool = False
    kv_layers: list[KvLayerSpec] = field(default_factory=list)
    kv_note: str = ""
    recurrent_bytes: int = 0          # per sequence, all recurrent layers
    recurrent_layers: int = 0
    recurrent_note: str = ""
    notes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def total_layers(self) -> Optional[int]:
        return self.blocks + 1 if self.blocks else None

    def _recurrent_per_layer(self) -> int:
        return self.recurrent_bytes // self.recurrent_layers \
            if self.recurrent_layers else 0

    def kv_bytes(self, ctx: int, *, element_bytes: float = 2.0, parallel: int = 1,
                 batch: int = SWA_CACHE_BATCH,
                 block_range: Optional[tuple[int, int]] = None) -> int:
        if not self.kv_layers:
            return 0
        start, end = block_range or (0, len(self.kv_layers))
        start = max(0, min(len(self.kv_layers), start))
        end = max(start, min(len(self.kv_layers), end))
        total = sum(spec.cells(max(0, int(ctx)), batch)
                    * spec.bytes_per_token(element_bytes)
                    for spec in self.kv_layers[start:end])
        return max(0, total) * max(1, int(parallel))

    def kv_slope(self, *, element_bytes: float = 2.0, parallel: int = 1,
                 block_range: Optional[tuple[int, int]] = None,
                 asymptotic: bool = True) -> int:
        """Bytes per extra token: every layer, or only the non-window ones."""
        if not self.kv_layers:
            return 0
        start, end = block_range or (0, len(self.kv_layers))
        start = max(0, min(len(self.kv_layers), start))
        end = max(start, min(len(self.kv_layers), end))
        total = sum(spec.bytes_per_token(element_bytes)
                    for spec in self.kv_layers[start:end]
                    if not asymptotic or not spec.window)
        return max(0, total) * max(1, int(parallel))

    def gpu_weight_share(self, layers: int) -> tuple[int, int]:
        """(GPU, CPU) weight bytes for `layers` offloaded (output layer counted)."""
        total_layers = self.total_layers or 0
        layers = max(0, min(total_layers, int(layers)))
        if layers >= total_layers:
            return self.weights_bytes, 0   # full offload keeps every weight
        if not self.split_estimate or not self.blocks:
            gpu = self.weights_bytes * layers // total_layers if total_layers else 0
            return gpu, self.weights_bytes - gpu
        # The output layer is offloaded first, then blocks from the top down.
        offloaded_blocks = max(0, min(self.blocks, layers - 1))
        gpu = (self.output_bytes if layers >= 1 else 0) \
            + int(self.block_bytes * offloaded_blocks / self.blocks)
        return min(gpu, self.weights_bytes), max(0, self.weights_bytes - gpu)

    def recurrent_share(self, layers: int) -> tuple[int, int]:
        """(GPU, CPU) recurrent-state bytes for `layers` offloaded (output counted)."""
        total_layers = self.total_layers or 0
        layers = max(0, min(total_layers, int(layers)))
        if not self.recurrent_layers or not self.blocks:
            return 0, self.recurrent_bytes
        offloaded_blocks = max(0, min(self.blocks, layers - 1))
        per_layer = self._recurrent_per_layer()
        gpu_layers = sum(1 for spec in self.kv_layers[-offloaded_blocks:]
                         if spec.heads <= 0) if offloaded_blocks else 0
        gpu = per_layer * gpu_layers
        return gpu, self.recurrent_bytes - gpu

    def kv_block_range(self, layers: int) -> Optional[tuple[int, int]]:
        """The repeating blocks on the GPU for `layers` offloaded layers."""
        if not self.blocks:
            return None
        if layers >= self.blocks + 1:
            return (0, self.blocks)
        offloaded = max(0, min(self.blocks, layers - 1))
        return (self.blocks - offloaded, self.blocks)

    def summary(self) -> str:
        bits = []
        if self.params:
            bits.append(f"{self.params / 1e9:.2f}B params")
        if self.quantization:
            bits.append(self.quantization)
        if self.architecture:
            bits.append(self.architecture)
        if self.weights_bytes:
            bits.append(human_bytes(self.weights_bytes))
        return " · ".join(bits)


def _quantization_for_size(size_bytes: int, params: Optional[int]) -> str:
    """The quantization label whose bytes/param best explains a file size."""
    if not size_bytes or not params:
        return ""
    ratio = size_bytes / params
    best = min(QUANT_BYTES_PER_PARAM.items(),
               key=lambda item: abs(item[1] - ratio))
    return best[0] if abs(best[1] - ratio) / max(ratio, 1e-9) < 0.2 else ""


def model_profile(show_data: dict[str, Any],
                  entry: Optional[dict[str, Any]] = None) -> ModelProfile:
    """Fold GGUF metadata and the /api/tags entry into one sizing profile."""
    info = _model_info(show_data)
    arch = _arch_prefix(info)
    details = (entry or {}).get("details") or {}
    profile = ModelProfile(
        architecture=arch,
        blocks=model_block_count(show_data),
        embedding_length=_int_or_none(_arch_key(info, arch, "embedding_length")
                                      or details.get("embedding_length")),
        feed_forward_length=_int_or_none(_arch_key(info, arch, "feed_forward_length")),
        context_length=_int_or_none(trained_context_length(show_data)
                                    or details.get("context_length")),
        vocab_size=_vocab_size(info, arch),
        expert_count=_int_or_none(_arch_key(info, arch, "expert_count")),
        expert_used=_int_or_none(_arch_key(info, arch, "expert_used_count")),
    )
    params = _int_or_none(info.get("general.parameter_count"))
    if params:
        profile.params_source = "GGUF general.parameter_count"
    else:
        billions = model_parameter_count(entry or {})
        if billions:
            params = int(billions * 1e9)
            profile.params_source = "/api/tags parameter_size"
    profile.params = params

    profile.quantization = quantization_label(show_data, entry)
    per_param = quantization_bytes_per_param(profile.quantization)
    size = _int_or_none((entry or {}).get("size")) or 0
    expected = int(params * per_param) if params and per_param else 0
    if entry and (entry.get("remote_model") or entry.get("remote_host")):
        profile.notes.append("the tag points at a remote/cloud model — "
                             "weights cannot be sized locally")
    elif size and expected and size < expected * 3 // 4:
        implied = _quantization_for_size(size, params)
        detail = f" (the file size matches {implied})" if implied else ""
        profile.notes.append(
            f"tags size {human_bytes(size)} is well under the "
            f"{human_bytes(expected)} implied by {profile.quantization} × "
            f"{params / 1e9:.2f}B params{detail} — using the larger figure")
        profile.weights_bytes = expected
        profile.weights_source = "parameters × quantization"
    elif size:
        profile.weights_bytes = size
        profile.weights_source = "/api/tags size"
        if expected and size > expected * 3 // 2:
            implied = _quantization_for_size(size, params)
            detail = f" (the file size matches {implied})" if implied else ""
            profile.notes.append(
                f"tags size {human_bytes(size)} is well above the "
                f"{human_bytes(expected)} implied by {profile.quantization} × "
                f"{params / 1e9:.2f}B params{detail} — using the disk size")
    elif expected:
        profile.weights_bytes = expected
        profile.weights_source = f"{params / 1e9:.2f}B params × {profile.quantization}"

    projector = show_data.get("projector_info") or {}
    projector_params = _int_or_none(projector.get("general.parameter_count"))
    if projector_params:
        profile.projector_params = projector_params
        file_type = projector.get("general.file_type")
        label = FILE_TYPE_LABELS.get(file_type) if isinstance(file_type, int) else ""
        projector_bpp = quantization_bytes_per_param(label) or per_param
        if projector_bpp:
            profile.projector_bytes = min(profile.weights_bytes,
                                          int(projector_params * projector_bpp))
        profile.notes.append(
            f"vision projector: {projector_params / 1e6:.0f}M params "
            f"≈ {human_bytes(profile.projector_bytes)}")

    tied = _arch_key(info, arch, "tie_word_embeddings")
    if isinstance(tied, bool):
        profile.tied_embeddings = tied
    main_params = params - (profile.projector_params or 0) if params else None
    main_bytes = max(0, profile.weights_bytes - profile.projector_bytes)
    if (main_params and main_params > 0 and main_bytes
            and profile.vocab_size and profile.embedding_length):
        bpp = main_bytes / main_params
        embed_params = profile.vocab_size * profile.embedding_length
        output_params = 0 if profile.tied_embeddings else embed_params
        if embed_params + output_params < main_params:
            profile.embedding_bytes = int(embed_params * bpp)
            profile.output_bytes = int(output_params * bpp)
            profile.block_bytes = max(
                0, main_bytes - profile.embedding_bytes - profile.output_bytes)
            profile.split_estimate = True
            shape = "tied" if profile.tied_embeddings else "separate"
            profile.notes.append(
                f"weights split: {human_bytes(profile.embedding_bytes)} token "
                f"embeddings + {human_bytes(profile.output_bytes)} output layer "
                f"({shape}) + {human_bytes(profile.block_bytes)} blocks")

    specs, kv_note = _layer_kv_specs(show_data)
    if specs:
        profile.kv_layers = specs
        profile.kv_note = kv_note
    else:
        profile.error = kv_note or "cannot size the KV cache"
    recurrent, recurrent_layers, recurrent_note = _recurrent_bytes(show_data)
    profile.recurrent_bytes = recurrent
    profile.recurrent_layers = recurrent_layers
    profile.recurrent_note = recurrent_note
    return profile


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
