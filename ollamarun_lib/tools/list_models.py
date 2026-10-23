"""`list_models`: installed models with the loaded ones marked."""
from __future__ import annotations

from typing import Any

from ..client import OllamaError
from ..helpers import canonical_model_name, human_bytes, short_stamp
from .spec import ToolContext, ToolError


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    client = ctx.client
    if client is None:
        raise ToolError("no Ollama server connection in this run")
    loaded_only = bool(args.get("loaded_only"))
    try:
        loaded = client.ps()
        installed = [] if loaded_only else client.tags()
    except OllamaError as exc:
        raise ToolError(str(exc)) from exc
    by_name = {canonical_model_name(entry.get("name", "")): entry for entry in loaded}
    if loaded_only:
        if not loaded:
            return "no models currently loaded"
        lines = [f"{len(loaded)} model(s) loaded on {client.base}:"]
        for entry in loaded:
            lines.append("- " + _resident_summary(entry))
        return "\n".join(lines)
    if not installed:
        return f"no models installed on {client.base}"
    lines = [f"{len(installed)} model(s) on {client.base} ({len(loaded)} loaded):"]
    for entry in installed:
        name = str(entry.get("name") or "?")
        details = entry.get("details") or {}
        bits = [name]
        if entry.get("size"):
            bits.append(human_bytes(entry["size"]))
        if details.get("parameter_size"):
            bits.append(str(details["parameter_size"]))
        if details.get("quantization_level"):
            bits.append(str(details["quantization_level"]))
        if entry.get("modified_at"):
            bits.append("modified " + short_stamp(entry["modified_at"]))
        resident = by_name.get(canonical_model_name(name))
        if resident is not None:
            bits.append("loaded (" + ", ".join(_resident_details(resident)) + ")")
        lines.append("- " + " · ".join(bits))
    return "\n".join(lines)


def _resident_summary(entry: dict[str, Any]) -> str:
    """`name · X VRAM · unloads at TIME` for one /api/ps entry."""
    name = str(entry.get("name") or entry.get("model") or "?")
    return " · ".join([name, *_resident_details(entry)])


def _resident_details(entry: dict[str, Any]) -> list[str]:
    """The VRAM/keep-alive bits of one /api/ps entry, without the name."""
    bits = []
    size = entry.get("size")
    vram = entry.get("size_vram")
    if vram:
        bit = f"{human_bytes(vram)} VRAM"
        if size and size > vram:
            bit += f" of {human_bytes(size)}"
        bits.append(bit)
    elif size:
        bits.append(human_bytes(size))
    if entry.get("expires_at"):
        bits.append("unloads " + short_stamp(entry["expires_at"]))
    return bits
