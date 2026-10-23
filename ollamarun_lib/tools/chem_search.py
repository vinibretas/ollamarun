"""`chem_search`: search containers and orders in chemInventoryAPI."""
from __future__ import annotations

from typing import Any

from .chem_cli import _chem_row, _chem_run
from .spec import ToolContext, ToolError


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ToolError("missing 'query'")
    argv = ["search", query]
    kind = str(args.get("type") or "").strip()
    if kind:
        argv += ["--type", kind]
    payload = _chem_run(ctx, argv)
    if not isinstance(payload, dict):
        payload = {}
    containers = payload.get("containers") or []
    orders = payload.get("orders") or []
    if not containers and not orders:
        return f"no containers match {query!r}"
    try:
        limit = int(args.get("limit") or 10)
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(limit, 50))
    lines = [f"{len(containers)} container(s) matching {query!r}:"]
    for row in containers[:limit]:
        lines.append("- " + _chem_row(row))
    if len(containers) > limit:
        lines.append(f"… {len(containers) - limit} more container(s) (raise limit)")
    if orders:
        lines.append(f"{len(orders)} matching order request(s)")
    return "\n".join(lines)


