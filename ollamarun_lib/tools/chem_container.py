"""`chem_container`: one container's details and field values."""
from __future__ import annotations

from typing import Any

from .chem_cli import _chem_run, _flatten
from .spec import ToolContext, ToolError


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    raw = args.get("id")
    if raw is None or not str(raw).strip():
        raise ToolError("missing 'id'")
    data = _chem_run(ctx, ["container", "info", str(raw)])
    if not isinstance(data, dict) or not data:
        return f"container {raw}: nothing found"
    lines = [f"container #{data.get('id', raw)}"]
    pairs = [
        ("name", data.get("containername") or data.get("name")),
        ("CAS", data.get("cas")),
        ("location", data.get("locationString") or data.get("locationtext")),
        ("amount", " ".join(str(part) for part in (data.get("size"), data.get("unit"))
                            if part).strip()),
        ("supplier", data.get("supplier")),
        ("acquired", data.get("dateacquired")),
        ("barcode", data.get("barcode")),
        ("formula", data.get("molecularformula")),
        ("MW", data.get("molecularweight")),
        ("exact mass", data.get("exactmass")),
        ("IUPAC", data.get("iupacname")),
        ("SMILES", data.get("isomericsmiles")),
        ("comments", data.get("comments")),
    ]
    for label, value in pairs:
        if value not in (None, "", "N/A"):
            lines.append(f"{label}: {_flatten(value)}")
    for prefix, header in (("cf-", "custom fields"), ("sf-", "substance fields")):
        fields = [(key[len(prefix):], value) for key, value in data.items()
                  if key.startswith(prefix) and not key.endswith("-value")
                  and value not in (None, "")]
        if fields:
            lines.append(header + ": " + "; ".join(
                f"{key}={_flatten(value)}" for key, value in fields[:20]))
    codes = []
    for entry in data.get("ghs") or []:
        codes += [code for code in str(entry.get("hazardstatements") or "").split("|")
                  if code]
    if codes:
        lines.append("GHS: " + ", ".join(codes))
    if data.get("needsretrieval"):
        lines.append("(substance info not retrieved yet)")
    return "\n".join(lines)


