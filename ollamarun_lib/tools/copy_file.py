"""`copy_file`: copy a file (or a tree) without overwriting by default."""
from __future__ import annotations

import shutil
from typing import Any

from .fs import transfer_paths
from .spec import ToolContext, ToolError, ToolHandlerResult


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    source, destination, overwrite = transfer_paths(args, ctx)
    if not source.exists() and not source.is_symlink():
        raise ToolError(f"source does not exist: {source}")
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise ToolError(f"destination exists: {destination} (pass overwrite=true "
                            f"to replace it)")
        if destination.is_dir():
            raise ToolError(f"destination is an existing directory: {destination}")
    try:
        if source.is_dir():
            shutil.copytree(str(source), str(destination))
        else:
            shutil.copy2(str(source), str(destination))
    except (OSError, shutil.Error) as exc:
        raise ToolError(f"could not copy {source} to {destination}: {exc}") from exc
    message = f"copied {source} → {destination}"
    if not source.is_dir() and destination.is_file():
        return ToolHandlerResult(message, [str(destination)])
    return message
