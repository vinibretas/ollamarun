"""`search_sessions`: search saved conversations through the session store."""
from __future__ import annotations

from typing import Any

from ..helpers import short_stamp
from .spec import ToolContext, ToolError


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    store = ctx.store
    if store is None:
        raise ToolError("session search is unavailable in this run")
    query = str(args.get("query") or "").strip()
    if not query:
        raise ToolError("missing 'query'")
    try:
        limit = int(args.get("limit") or 5)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(limit, 20))
    try:
        results = store.search(query, limit)
    except OSError as exc:
        raise ToolError(f"could not search sessions: {exc}") from exc
    if not results:
        return f"no saved session matches {query!r}"
    lines = [f"{len(results)} saved session(s) matching {query!r}:"]
    for session, snippet in results:
        title = " ".join(str(session.display_title()).split())
        lines.append(f"- {session.id} · {title} · model {session.model or '?'} "
                     f"· updated {short_stamp(session.updated_at)}")
        lines.append(f"  …{' '.join(str(snippet).split())}")
    lines.append("(Use /show SESSION in the REPL to read a full conversation.)")
    return "\n".join(lines)
