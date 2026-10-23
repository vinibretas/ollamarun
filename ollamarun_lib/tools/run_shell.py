"""`run_shell`: one bash -c command, killed with its process group on cancel."""
from __future__ import annotations

import os
import subprocess
from typing import Any

from ..helpers import run_cancellable
from .process import combine_exit
from .spec import ToolContext, ToolError, _clamp_timeout
from .workspace import _resolve_in_workspace


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    command = str(args.get("command") or "").strip()
    if not command:
        raise ToolError("missing 'command'")
    cwd = _resolve_in_workspace(
        str(args.get("cwd") or getattr(ctx, "workspace", None) or ctx.cwd),
        ctx)
    if not cwd.is_dir():
        raise ToolError(f"working directory does not exist: {cwd}")
    timeout = _clamp_timeout(args.get("timeout"), ctx.timeout)
    environment = {**os.environ, "BASH_ENV": os.devnull}
    try:
        proc = run_cancellable(["bash", "-c", command], cwd=str(cwd),
                               env=environment, timeout=timeout,
                               cancel=getattr(ctx, "cancel", None))
    except subprocess.TimeoutExpired:
        raise ToolError(f"command timed out after {timeout:.0f}s") from None
    except OSError as exc:
        raise ToolError(f"could not run command: {exc}") from exc
    return combine_exit(proc.returncode, proc.stdout, proc.stderr)
