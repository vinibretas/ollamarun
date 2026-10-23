"""`edit_file`: exact-fragment replacement with a diff preview."""
from __future__ import annotations

import os
import tempfile
from typing import Any, Optional

from ..helpers import diff_preview
from .spec import ToolContext, ToolError, ToolHandlerResult
from .workspace import _resolve_in_workspace


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    path = _resolve_in_workspace(str(args.get("path") or ""), ctx)
    old = args.get("old_string")
    new = args.get("new_string")
    if not str(path) or str(path) == ".":
        raise ToolError("missing 'path'")
    if not isinstance(old, str) or not old:
        raise ToolError("missing 'old_string' (must be a non-empty exact fragment)")
    if not isinstance(new, str):
        raise ToolError("missing 'new_string'")
    if not path.is_file():
        raise ToolError(f"not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"not a utf-8 text file: {path}") from exc
    except OSError as exc:
        raise ToolError(f"could not read {path}: {exc}") from exc
    count = text.count(old)
    if count == 0:
        raise ToolError(f"old_string not found in {path} — read the file first and "
                        f"copy the exact text to replace")
    replace_all = bool(args.get("replace_all"))
    if count > 1 and not replace_all:
        raise ToolError(f"old_string appears {count} times in {path}; include more "
                        f"surrounding context or pass replace_all=true")
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    tmp_name: Optional[str] = None
    try:
        mode = path.stat().st_mode
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                        dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
        os.chmod(tmp_name, mode & 0o7777)
        os.replace(tmp_name, path)
        tmp_name = None
    except OSError as exc:
        raise ToolError(f"could not write {path}: {exc}") from exc
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    replaced = count if replace_all else 1
    if replaced == 1:
        first = text[:text.find(old)].count("\n") + 1
        where = f" at line {first}"
    else:
        where = ""
    diff = diff_preview(old, new, limit=30)
    result = f"edited {path}: replaced {replaced} occurrence(s){where}"
    text = f"{result}\n{diff}" if diff else result
    return ToolHandlerResult(text, [str(path)])
