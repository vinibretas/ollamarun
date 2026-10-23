"""The chemInventoryAPI subprocess plumbing shared by the chem tools."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from ..helpers import run_cancellable
from .spec import ToolContext, ToolError


def run_chem_inventory(binary: str, argv: list[str], timeout: float,
                       cancel: Optional[Any] = None) -> Any:
    """Run `chemInventoryAPI … --json` and parse the JSON it prints."""
    try:
        proc = run_cancellable([binary, *argv, "--json"], timeout=timeout,
                               cancel=cancel)
    except subprocess.TimeoutExpired:
        raise ToolError(f"chemInventoryAPI timed out after {timeout:.0f}s") from None
    except OSError as exc:
        raise ToolError(f"could not run chemInventoryAPI: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        if len(detail) > 400:
            detail = detail[:399] + "…"
        raise ToolError(f"chemInventoryAPI failed: {detail or 'unknown error'}")
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        raise ToolError(f"chemInventoryAPI returned invalid JSON: {exc}") from exc


def _chem_run(ctx: ToolContext, argv: list[str]) -> Any:
    binary = find_chem_inventory(ctx)
    if not binary:
        raise ToolError("chemInventoryAPI not found (PATH or config chem_inventory)")
    return run_chem_inventory(binary, argv, min(ctx.timeout, 90),
                              cancel=getattr(ctx, "cancel", None))


def _flatten(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(item) for item in value)
    return " ".join(str(value).split())


def _chem_row(row: dict[str, Any]) -> str:
    """One compact `#id · name · CAS · amount · location` line from a record."""
    bits = []
    if row.get("id") is not None:
        bits.append(f"#{row['id']}")
    name = row.get("name") or row.get("containername")
    if name:
        bits.append(_flatten(name))
    if row.get("cas"):
        bits.append(f"CAS {_flatten(row['cas'])}")
    amount = " ".join(str(part) for part in (row.get("size"), row.get("unit")) if part)
    if amount.strip():
        bits.append(amount.strip())
    location = row.get("locationtext") or row.get("locationString")
    if location:
        bits.append(f"at {_flatten(location)}")
    if row.get("barcode"):
        bits.append(f"barcode {_flatten(row['barcode'])}")
    if row.get("supplier"):
        bits.append(f"from {_flatten(row['supplier'])}")
    return " · ".join(bits) or "(empty record)"


def find_chem_inventory(ctx: ToolContext) -> Optional[str]:
    """The chemInventoryAPI CLI: config override, PATH, then the submodule."""
    candidate = ctx.chem_inventory or "chemInventoryAPI"
    found = shutil.which(candidate) if os.path.sep not in candidate else (
        candidate if Path(candidate).exists() else None)
    if found:
        return found
    # chemInventoryAPI lives in a submodule: ~/bin/chemInventoryAPI/chemInventoryAPI
    fallback = Path.home() / "bin" / "chemInventoryAPI"
    if fallback.is_file():
        return str(fallback)
    entry = fallback / "chemInventoryAPI"
    return str(entry) if entry.is_file() else None


def chem_inventory_available() -> bool:
    """True when the chemInventoryAPI CLI can be found (the `chem` gate)."""
    return find_chem_inventory(ToolContext()) is not None
