"""`delete_file`: move a path to the trash; never a permanent unlink."""
from __future__ import annotations

import shutil
import subprocess
from typing import Any

from ..helpers import run_cancellable
from .spec import ToolContext, ToolError
from .workspace import _resolve_in_workspace


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    raw = str(args.get("path") or "").strip()
    if not raw:
        raise ToolError("missing 'path'")
    path = _resolve_in_workspace(raw, ctx)
    if not path.exists() and not path.is_symlink():
        raise ToolError(f"no such file or directory: {path}")
    trash = shutil.which("trash-put")
    if not trash:
        raise ToolError("trash-put not found; refusing to delete without a trash")
    try:
        proc = run_cancellable([trash, str(path)], timeout=ctx.timeout,
                               cancel=getattr(ctx, "cancel", None))
    except subprocess.TimeoutExpired:
        raise ToolError("trash-put timed out") from None
    except OSError as exc:
        raise ToolError(f"could not run trash-put: {exc}") from exc
    if proc.returncode != 0:
        raise ToolError(f"trash-put failed: "
                        f"{(proc.stderr or proc.stdout).strip() or 'unknown error'}")
    return f"moved to trash: {path}"
