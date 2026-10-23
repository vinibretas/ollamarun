"""`write_file`: write or append a whole text file."""
from __future__ import annotations

from typing import Any

from .spec import ToolContext, ToolError, ToolHandlerResult
from .workspace import _resolve_in_workspace


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    path = _resolve_in_workspace(str(args.get("path") or ""), ctx)
    if "content" not in args:
        raise ToolError("missing 'content'")
    content = str(args.get("content") or "")
    append = bool(args.get("append"))
    try:
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        raise ToolError(f"could not write {path}: {exc}") from exc
    verb = "appended" if append else "wrote"
    return ToolHandlerResult(f"{verb} {len(content)} chars to {path}", [str(path)])
