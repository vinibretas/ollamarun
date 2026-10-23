"""`publish_artifact`: snapshot a workspace file as a downloadable artifact."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .spec import ToolContext, ToolError, ToolHandlerResult


def run(args: dict[str, Any], ctx: ToolContext) -> "str | ToolHandlerResult":
    """Snapshot one workspace file as a downloadable artifact.

    The explicit path for files created by `run_python`/`run_shell` (the
    file tools publish their own results automatically). Jailed to the
    workspace root (the API) or the working directory (the CLI), and capped
    by the store's per-file limit.
    """
    raw = str(args.get("path") or "").strip()
    if not raw:
        raise ToolError("missing 'path'")
    publisher = getattr(ctx, "artifacts", None)
    if publisher is None:
        raise ToolError("publishing artifacts needs the serve artifact store")
    base = Path(getattr(ctx, "workspace", None) or ctx.cwd).resolve()
    candidate = Path(raw).expanduser()
    target = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if target != base and base not in target.parents:
        raise ToolError(f"refused: {raw} is outside the working directory")
    if not target.is_file():
        raise ToolError(f"no such file: {raw}")
    limit = publisher.store.max_bytes
    if 0 < limit < target.stat().st_size:
        raise ToolError(f"refused: {raw} exceeds the {limit} byte artifact limit")
    return ToolHandlerResult(f"published {raw} as a downloadable artifact",
                             [str(target)])
