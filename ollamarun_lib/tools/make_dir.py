"""`make_dir`: create a directory tree (an existing one is not an error)."""
from __future__ import annotations

from typing import Any

from .spec import ToolContext, ToolError
from .workspace import _resolve_in_workspace


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    raw = str(args.get("path") or "").strip()
    if not raw:
        raise ToolError("missing 'path'")
    path = _resolve_in_workspace(raw, ctx)
    if path.is_dir():
        return f"directory already exists: {path}"
    if path.exists():
        raise ToolError(f"a file exists at {path}")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolError(f"could not create {path}: {exc}") from exc
    return f"created directory {path}"
