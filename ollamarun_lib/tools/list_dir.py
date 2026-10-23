"""`list_dir`: one directory's entries with sizes and mtimes."""
from __future__ import annotations

import time
from typing import Any

from ..helpers import short_stamp
from .spec import ToolContext, ToolError
from .workspace import _resolve_in_workspace

#: Most entries one call lists.
MAX_LIST_ENTRIES = 200


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    path = _resolve_in_workspace(
        str(args.get("path") or getattr(ctx, "workspace", None) or ctx.cwd),
        ctx)
    if not path.is_dir():
        raise ToolError(f"not a directory: {path}")
    try:
        entries = sorted(path.iterdir(), key=lambda entry: (not entry.is_dir(),
                                                            entry.name.lower()))
    except OSError as exc:
        raise ToolError(f"could not list {path}: {exc}") from exc
    lines = [f"{path} ({len(entries)} entries)"]
    for entry in entries[:MAX_LIST_ENTRIES]:
        try:
            info = entry.stat()
            detail = "dir" if entry.is_dir() else f"{info.st_size} B"
            stamp = short_stamp(time.strftime("%Y-%m-%dT%H:%M:%S",
                                             time.localtime(info.st_mtime)))
        except OSError:
            detail, stamp = "?", "?"
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{entry.name}{suffix}  ({detail}, {stamp})")
    if len(entries) > MAX_LIST_ENTRIES:
        lines.append(f"… {len(entries) - MAX_LIST_ENTRIES} more entries")
    return "\n".join(lines)
