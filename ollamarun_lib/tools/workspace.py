"""Workspace path jail for the serve API's file tools."""
from __future__ import annotations

from pathlib import Path

from .spec import ToolContext, ToolError


def _resolve_in_workspace(raw: str, ctx: ToolContext) -> Path:
    """Resolve one tool path argument under the API workspace, if set.

    With no workspace (the CLI) the path is left exactly as before. With
    one (serve), a relative path resolves against the root and the resolved
    path — symlinks followed — must stay inside it, else the call is
    refused.
    """
    path = Path(raw).expanduser()
    workspace = getattr(ctx, "workspace", None)
    if not workspace:
        return path
    root = Path(workspace).resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ToolError(f"refused: {raw or '.'} is outside the workspace "
                        f"root ({root})")
    return resolved


def _workspace_base(ctx: ToolContext) -> Path:
    """The base for default path/cwd arguments (the workspace when set)."""
    workspace = getattr(ctx, "workspace", None)
    return Path(workspace).resolve() if workspace else Path(ctx.cwd)
