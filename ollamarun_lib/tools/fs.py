"""Shared path plumbing for the file-transfer tools."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .spec import ToolContext, ToolError
from .workspace import _resolve_in_workspace


def transfer_paths(args: dict[str, Any],
                    ctx: ToolContext) -> tuple[Path, Path, bool]:
    source = str(args.get("source") or "").strip()
    destination = str(args.get("destination") or "").strip()
    if not source:
        raise ToolError("missing 'source'")
    if not destination:
        raise ToolError("missing 'destination'")
    return (_resolve_in_workspace(source, ctx),
            _resolve_in_workspace(destination, ctx),
            bool(args.get("overwrite")))
