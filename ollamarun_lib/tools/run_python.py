"""`run_python`: an isolated `python -I -` snippet with a timeout."""
from __future__ import annotations

import subprocess
import sys
from typing import Any

from ..helpers import run_cancellable
from .process import combine_exit
from .spec import ToolContext, ToolError, _clamp_timeout
from .workspace import _workspace_base


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ToolError("missing 'code'")
    timeout = _clamp_timeout(args.get("timeout"), ctx.timeout)
    workdir = _workspace_base(ctx)
    try:
        proc = run_cancellable([sys.executable, "-I", "-"], input_text=code,
                               cwd=str(workdir), timeout=timeout,
                               cancel=getattr(ctx, "cancel", None))
    except subprocess.TimeoutExpired:
        raise ToolError(f"python timed out after {timeout:.0f}s") from None
    except OSError as exc:
        raise ToolError(f"could not run python: {exc}") from exc
    return combine_exit(proc.returncode, proc.stdout, proc.stderr)
