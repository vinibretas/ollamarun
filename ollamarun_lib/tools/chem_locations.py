"""`chem_locations`: the location tree, or the containers in one location."""
from __future__ import annotations

from typing import Any

from .chem_cli import _chem_row, _chem_run, _flatten
from .spec import ToolContext


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    location = str(args.get("location") or "").strip()
    if location:
        payload = _chem_run(ctx, ["location", "containers", location])
        containers = payload.get("containers") or [] if isinstance(payload, dict) else []
        if not containers:
            return f"no containers in {location!r}"
        lines = [f"{len(containers)} container(s) in {location!r}:"]
        for row in containers[:40]:
            lines.append("- " + _chem_row(row))
        if len(containers) > 40:
            lines.append(f"… {len(containers) - 40} more container(s)")
        return "\n".join(lines)
    payload = _chem_run(ctx, ["location", "list"])
    entries = payload if isinstance(payload, list) else []
    if not entries:
        return "no locations"
    children: dict[Any, list] = {}
    for loc in entries:
        children.setdefault(loc.get("parent") or 0, []).append(loc)
    for bucket in children.values():
        bucket.sort(key=lambda item: _flatten(item.get("name") or "").lower())
    ordered: list[tuple[int, dict[str, Any]]] = []
    seen: set[Any] = set()

    def walk(parent: Any, depth: int) -> None:
        for loc in children.get(parent, []):
            if loc.get("id") in seen:
                continue
            seen.add(loc.get("id"))
            ordered.append((depth, loc))
            walk(loc.get("id"), depth + 1)

    walk(0, 0)
    for loc in entries:  # orphans (parent id not present) still get listed
        if loc.get("id") not in seen:
            ordered.append((0, loc))
    lines = [f"{len(entries)} location(s) (id · containers):"]
    for depth, loc in ordered:
        name = _flatten(loc.get("name") or "?")
        count = loc.get("numbercontainers", 0)
        lines.append(f"- {'  ' * depth}{name} (#{loc.get('id')}, {count})")
    return "\n".join(lines)
