"""`system_status`: RAM/GPU/disk of the machine running Ollama."""
from __future__ import annotations

from typing import Any

from ..helpers import (host_is_self, human_bytes, is_local_host,
                       local_free_disk, local_gpu_probe, local_ram_probe,
                       ssh_free_disk, ssh_gpu_probe, ssh_ram_probe)
from .spec import ToolContext, ToolError


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    target = str(args.get("target") or "ollama").strip().lower()
    if target not in {"ollama", "local"}:
        raise ToolError("'target' must be 'ollama' or 'local'")
    base = str(getattr(ctx.client, "base", "") or "")
    ssh_target = None
    if target == "ollama" and base and not (host_is_self(base)
                                            or is_local_host(base)):
        if not ctx.ssh_target:
            raise ToolError(
                f"the Ollama host {base} is remote; run with --remote NAME so "
                f"its resources can be probed over ssh, or pass target=\"local\" "
                f"for this machine")
        ssh_target = ctx.ssh_target
    if target == "local":
        label = "this machine"
    elif ssh_target:
        label = f"{ctx.ssh_target} (ollama at {base})"
    elif base:
        label = f"this machine (ollama at {base})"
    else:
        label = "this machine"
    lines = [label]

    if ssh_target:
        probe_cancel = getattr(ctx, "cancel", None)
        ram = ssh_ram_probe(ssh_target, user=ctx.ssh_user, cancel=probe_cancel)
        gpu = ssh_gpu_probe(ssh_target, user=ctx.ssh_user, cancel=probe_cancel)
        disk = ssh_free_disk(ssh_target, user=ctx.ssh_user, cancel=probe_cancel)
    else:
        ram = local_ram_probe()
        gpu = local_gpu_probe()
        disk = local_free_disk()
    if ram:
        total, available = ram
        lines.append(f"RAM: {human_bytes(available)} free of {human_bytes(total)}")
    else:
        lines.append("RAM: unknown")
    devices = gpu[0] if gpu else []
    if devices:
        for index, (total_mb, free_mb) in enumerate(devices):
            lines.append(f"GPU {index}: {human_bytes(free_mb * 1024 * 1024)} free "
                         f"of {human_bytes(total_mb * 1024 * 1024)}")
    else:
        lines.append("GPU: no nvidia-smi data")
    if disk:
        free, total, where = disk
        lines.append(f"disk (models): {human_bytes(free)} free of {human_bytes(total)} "
                     f"at {where}")
    else:
        lines.append("disk (models): unknown")
    return "\n".join(lines)
