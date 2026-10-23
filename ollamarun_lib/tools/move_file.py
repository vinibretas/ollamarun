"""`move_file`: move or rename a file or directory."""
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
        if destination.is_dir():
            raise ToolError(f"destination is an existing directory: {destination}")
        if not overwrite:
            raise ToolError(f"destination exists: {destination} (pass overwrite=true "
                            f"to replace it)")
    try:
        shutil.move(str(source), str(destination))
    except (OSError, shutil.Error) as exc:
        raise ToolError(f"could not move {source} to {destination}: {exc}") from exc
    message = f"moved {source} → {destination}"
    if destination.is_file():  # directories are never recursive bundles
        return ToolHandlerResult(message, [str(destination)])
    return message
