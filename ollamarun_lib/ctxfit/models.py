"""The dataclasses shared by every context-fit stage.

`FitRequest`/`FitCancelled` are the call contract; `GpuFit`/`CpuFit` are the
estimates, `Probe`/`ProbeResult` the measured runs. Annotations are deferred,
so this module imports nothing from its siblings at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .machine import MachineGpu, MachineRam
    from .profile import ModelProfile


@dataclass
class GpuFit:
    """The estimate: inputs, what goes on the GPU, and the resulting context."""
    ctx: int = 0
    feasible: bool = False
    kv_per_token: int = 0
    kv_gpu_per_token: int = 0
    weights_bytes: int = 0          # GPU share of the weights
    weights_total: int = 0
    usable_bytes: int = 0
    free_bytes: int = 0
    margin_bytes: int = 0           # per device
    overhead_bytes: int = 0
    credited_bytes: int = 0         # resident models' VRAM credited back (size_vram sum)
    residents: list[tuple[str, int]] = field(default_factory=list)
    ctx_no_margin: Optional[int] = None     # what would fit without the margin
    model_max: Optional[int] = None
    capped_by_model: bool = False
    pinned_layers: Optional[int] = None
    total_layers: Optional[int] = None
    parallel: int = 1
    element_bytes: float = 2.0
    devices: list[tuple[int, int]] = field(default_factory=list)
    source: str = ""
    profile: Optional[ModelProfile] = None
    kv_bytes_at_ctx: int = 0
    kv_bytes_total_at_ctx: int = 0
    recurrent_bytes: int = 0
    recurrent_gpu_bytes: int = 0
    weights_source: str = ""
    calibrated: bool = False
    notes: list[str] = field(default_factory=list)
    error: str = ""




@dataclass
class CpuFit:
    """The estimate: the RAM budget, the placement, and the resulting context."""
    ctx: int = 0
    feasible: bool = False
    percent: int = 0
    total_bytes: int = 0                # MemTotal
    available_bytes: int = 0            # MemAvailable as measured
    resident_bytes: int = 0             # the model's host RAM already in /api/ps
    budget_bytes: int = 0               # available + resident
    target_bytes: int = 0               # percent% of the budget
    footprint_bytes: int = 0            # the estimate's host RAM at ctx
    weights_total: int = 0
    cpu_weights: int = 0
    kv_per_token: int = 0
    cpu_kv_per_token: int = 0
    overhead_bytes: int = 0             # CPU buffers the estimate cannot read
    gpu_overhead_bytes: int = 0         # the GPU reserve the placement fit uses
    gpu_layers: Optional[int] = None    # layers llama.cpp's fit should keep on the GPU
    total_layers: Optional[int] = None
    pinned_layers: Optional[int] = None
    model_max: Optional[int] = None
    budget_ctx: Optional[int] = None    # where the analytic footprint hits the budget
    capped_by_model: bool = False
    devices: list[tuple[int, int]] = field(default_factory=list)
    free_bytes: int = 0
    margin_bytes: int = 0
    usable_bytes: int = 0
    source: str = ""
    profile: Optional[ModelProfile] = None
    cpu_kv_bytes: int = 0
    recurrent_bytes: int = 0
    recurrent_cpu_bytes: int = 0
    weights_source: str = ""
    calibrated: bool = False
    notes: list[str] = field(default_factory=list)
    error: str = ""




@dataclass
class Probe:
    ctx: int
    total: int = 0
    vram: int = 0
    pct: int = 0
    host: int = 0                       # size - size_vram (the --max-cpu measure)
    failed: str = ""


@dataclass
class ProbeResult:
    ctx: Optional[int] = None
    probes: list[Probe] = field(default_factory=list)
    capped_by_model: bool = False
    reason: str = ""
    target_bytes: int = 0               # --max-cpu: the host-RAM fill target
    budget_bytes: int = 0               # --max-cpu: the budget it is a share of


@dataclass
class FitRequest:
    """Everything the estimate/probe needs about the target machine and model."""
    client: Any
    model: str
    config: dict[str, Any] = field(default_factory=dict)
    host_label: Optional[str] = None
    ssh_target: Optional[str] = None
    ssh_user: Optional[str] = None
    pinned_spec: Optional[str] = None       # session gpu-layers ("50%", "all", ...)
    num_gpu: Optional[int] = None           # absolute num_gpu fallback
    base_options: Optional[dict[str, Any]] = None
    keepalive: Optional[str] = None
    machine: Optional[MachineGpu] = None    # injectable (tests / reuse)
    refresh_gpu: Optional[Callable[[], Optional[MachineGpu]]] = None
    ram: Optional[MachineRam] = None        # injectable (tests / reuse)
    calibration_key: str = ""
    calibration_path: Optional[str] = None
    #: A turn/job CancelToken: checked at every checkpoint; set means the fit
    #: stops promptly (raise `FitCancelled`) instead of loading more models.
    cancel: Optional[Any] = None
    #: Called with `{"stage", "mode", ...}` progress dicts as the fit moves
    #: (estimate → probe, per candidate); never allowed to break the fit.
    on_progress: Optional[Callable[[dict[str, Any]], None]] = None


class FitCancelled(Exception):
    """A fit was stopped at a checkpoint (the job reports it cancelled)."""


