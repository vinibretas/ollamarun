"""`read_file`: a numbered text window from one file."""
from __future__ import annotations

from typing import Any

from .spec import ToolContext, ToolError
from .workspace import _resolve_in_workspace

#: Most characters one call reads before the offset/limit window applies.
MAX_READ_CHARS = 200_000


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    path = _resolve_in_workspace(str(args.get("path") or ""), ctx)
    if not path.is_file():
        raise ToolError(f"not a file: {path}")
    try:
        with path.open("rb") as handle:
            head = handle.read(5)
    except OSError as exc:
        raise ToolError(f"could not read {path}: {exc}") from exc
    if head == b"%PDF-":
        raise ToolError(
            f"{path} is a PDF, which read_file cannot show as text; call "
            f"read_pdf with this path (it extracts the text layer and reads "
            f"scans with a vision model)")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(MAX_READ_CHARS)
    except OSError as exc:
        raise ToolError(f"could not read {path}: {exc}") from exc
    lines = text.splitlines()
    try:
        offset = max(0, int(args.get("offset") or 0))
    except (TypeError, ValueError):
        offset = 0
    limit = args.get("limit")
    try:
        limit = max(1, int(limit)) if limit is not None else None
    except (TypeError, ValueError):
        limit = None
    window = lines[offset:offset + limit] if limit else lines[offset:]
    numbered = "\n".join(f"{offset + index + 1}: {line}"
                         for index, line in enumerate(window))
    return f"{path} ({len(lines)} lines shown)\n{numbered}"
