"""The target machine: free VRAM and DRAM, locally or over one ssh probe."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..helpers import (host_is_self, is_local_host, local_gpu_probe,
                       local_ram_probe, ssh_gpu_probe, ssh_ram_probe)


@dataclass
class MachineGpu:
    """GPU totals/free (MiB per device), the ollama unit env, and where they came from."""
    devices: list[tuple[int, int]] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    source: str = ""            # "local" or "ssh:HOST"
    error: str = ""


def probe_machine_gpu(*, ssh_target: Optional[str] = None,
                      ssh_user: Optional[str] = None) -> MachineGpu:
    """Read free VRAM from this machine or an ssh host (never raises)."""
    if ssh_target:
        got = ssh_gpu_probe(ssh_target, user=ssh_user)
        if got is None:
            return MachineGpu(source=f"ssh:{ssh_target}",
                              error=f"no GPU data from ssh {ssh_target}")
        devices, env = got
        return MachineGpu(devices=devices, env=env, source=f"ssh:{ssh_target}")
    devices, env = local_gpu_probe()
    if not devices:
        return MachineGpu(env=env, source="local", error="no nvidia-smi GPU data")
    return MachineGpu(devices=devices, env=env, source="local")


def machine_state_for(client_base: str, *, ssh_target: Optional[str] = None,
                      ssh_user: Optional[str] = None) -> Optional[MachineGpu]:
    """MachineGpu for a client's base URL: local when it is this machine, ssh
    for --remote hosts, and None when neither applies."""
    if client_base and host_is_self(client_base):
        return probe_machine_gpu()
    if ssh_target:
        return probe_machine_gpu(ssh_target=ssh_target, ssh_user=ssh_user)
    if is_local_host(client_base or ""):
        return probe_machine_gpu()
    return None


# ── machine state (available DRAM for the --max-cpu fill) ───────────────────

@dataclass
class MachineRam:
    """DRAM totals/available in bytes, and where they came from."""
    total_bytes: int = 0
    available_bytes: int = 0    # /proc/meminfo's MemAvailable
    source: str = ""            # "local" or "ssh:HOST"
    error: str = ""


def probe_machine_ram(*, ssh_target: Optional[str] = None,
                      ssh_user: Optional[str] = None) -> MachineRam:
    """Read DRAM from this machine or an ssh host (never raises)."""
    if ssh_target:
        got = ssh_ram_probe(ssh_target, user=ssh_user)
        if got is None:
            return MachineRam(source=f"ssh:{ssh_target}",
                              error=f"no memory data from ssh {ssh_target}")
        total, available = got
        return MachineRam(total_bytes=total, available_bytes=available,
                          source=f"ssh:{ssh_target}")
    got = local_ram_probe()
    if got is None:
        return MachineRam(source="local", error="cannot read /proc/meminfo")
    total, available = got
    return MachineRam(total_bytes=total, available_bytes=available, source="local")


def machine_ram_for(client_base: str, *, ssh_target: Optional[str] = None,
                    ssh_user: Optional[str] = None) -> Optional[MachineRam]:
    """MachineRam for a client's base URL: local when it is this machine, ssh
    for --remote hosts, and None when neither applies."""
    if client_base and host_is_self(client_base):
        return probe_machine_ram()
    if ssh_target:
        return probe_machine_ram(ssh_target=ssh_target, ssh_user=ssh_user)
    if is_local_host(client_base or ""):
        return probe_machine_ram()
    return None

