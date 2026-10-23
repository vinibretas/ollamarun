"""Tool definitions (defs/tools/*.json), their loader and the default set."""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from ..constants import DEFS_DIR, DEFS_OVERRIDE_DIR, console
from . import (chem_container, chem_locations, chem_search, context_status,
               copy_file, delete_file, edit_file, fetch_url, list_dir,
               list_models, list_voices, make_dir, move_file, publish_artifact,
               read_file, read_pdf, run_python, run_shell, search_sessions,
               session_info, speak, system_status, web_search, write_file)
from .chem_cli import chem_inventory_available
from .spec import ToolContext, ToolHandlerResult, ToolSpec
from .web_search import search_available


@dataclass
class ToolDefinition:
    """One tool's declaration, loaded from its JSON file."""
    name: str
    description: str
    parameters: dict[str, Any]
    handler: str
    read_only: bool = False
    always_safe: bool = False
    #: capability the tool needs to appear: None, "search" or "tts"
    requires: Optional[str] = None
    #: names models tend to invent for this tool
    aliases: list[str] = field(default_factory=list)


#: `handler` name in a tool JSON -> the function that executes it.
#: Every tool module exposes the same `run(args, ctx)` entry point; handlers
#: return plain text, or a `ToolHandlerResult` when they produced files
#: (`execute_tool()` registers them through `ctx.artifacts`).
HANDLERS: dict[str, Callable[[dict[str, Any], ToolContext],
                             Union[str, ToolHandlerResult]]] = {
    "shell": run_shell.run,
    "python": run_python.run,
    "read_file": read_file.run,
    "read_pdf": read_pdf.run,
    "write_file": write_file.run,
    "edit_file": edit_file.run,
    "make_dir": make_dir.run,
    "move_file": move_file.run,
    "copy_file": copy_file.run,
    "delete_file": delete_file.run,
    "list_dir": list_dir.run,
    "publish_artifact": publish_artifact.run,
    "search_sessions": search_sessions.run,
    "list_models": list_models.run,
    "system_status": system_status.run,
    "context_status": context_status.run,
    "session_info": session_info.run,
    "chem_search": chem_search.run,
    "chem_container": chem_container.run,
    "chem_locations": chem_locations.run,
    "web_search": web_search.run,
    "fetch_url": fetch_url.run,
    "speak": speak.run,
    "list_voices": list_voices.run,
}


REQUIREMENTS = {"search", "tts", "chem", "artifacts"}

#: `requires` value -> what `/tools NAME help` says a missing tool needs.
REQUIREMENT_HINTS: dict[str, str] = {
    "search": "the ddgs package (pip install ddgs)",
    "tts": "the tts-remote command",
    "chem": "the chemInventoryAPI command (config: chem_inventory)",
    "artifacts": "the serve artifact store (`ollamarun serve`)",
}

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_DEFS_CACHE: Optional[list[ToolDefinition]] = None
_ALIASES_CACHE: Optional[dict[str, str]] = None


def tool_definition_dirs() -> list[Path]:
    """The definition layers, lowest priority first (config overrides ship on top)."""
    return [DEFS_DIR / "tools", DEFS_OVERRIDE_DIR / "tools"]


def _parse_definition(path: Path) -> ToolDefinition:
    """One JSON file -> ToolDefinition; raises ValueError with the reason."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"bad JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    name = str(data.get("name") or "").strip()
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"bad tool name {name!r}")
    description = str(data.get("description") or "").strip()
    if not description:
        raise ValueError("missing 'description'")
    parameters = data.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise ValueError("'parameters' must be a JSON-schema object")
    handler = str(data.get("handler") or "").strip()
    if handler not in HANDLERS:
        raise ValueError(f"unknown handler {handler!r}")
    requires = data.get("requires")
    if requires is not None and requires not in REQUIREMENTS:
        raise ValueError(f"unknown requirement {requires!r} (search/tts/chem/artifacts)")
    aliases = data.get("aliases") or []
    if not isinstance(aliases, list) or not all(
            isinstance(alias, str) for alias in aliases):
        raise ValueError("'aliases' must be a list of strings")
    return ToolDefinition(
        name=name, description=description, parameters=parameters,
        handler=handler,
        read_only=bool(data.get("read_only", False)),
        always_safe=bool(data.get("always_safe", False)),
        requires=requires,
        aliases=[alias.strip().lower() for alias in aliases if alias.strip()])


def load_tool_definitions(
        directories: Optional[list[Path]] = None) -> list[ToolDefinition]:
    """Every tool definition; a same-named file in a later layer wins.

    Bad files and duplicate tool names are skipped with a warning: a broken
    edit must never take the whole tool harness down.
    """
    if directories is not None:
        folders = [Path(directory) for directory in directories]
    else:
        folders = tool_definition_dirs()
    files: dict[str, Path] = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            files[path.stem] = path  # the later folder's file replaces it
    definitions: list[ToolDefinition] = []
    seen: dict[str, Path] = {}
    for stem in sorted(files):
        path = files[stem]
        try:
            definition = _parse_definition(path)
        except ValueError as exc:
            console.print(f"[warn]ignoring tool definition {path}: {exc}[/warn]")
            continue
        if definition.name in seen:
            console.print(f"[warn]ignoring tool definition {path}: duplicate name "
                          f"{definition.name!r} (already defined by "
                          f"{seen[definition.name]})[/warn]")
            continue
        seen[definition.name] = path
        definitions.append(definition)
    return definitions


def tool_definitions() -> list[ToolDefinition]:
    """The cached default-path definitions (a restart picks up file edits)."""
    global _DEFS_CACHE
    if _DEFS_CACHE is None:
        _DEFS_CACHE = load_tool_definitions()
    return _DEFS_CACHE


def tool_aliases() -> dict[str, str]:
    """alias -> tool name, built from each definition's `aliases` list."""
    global _ALIASES_CACHE
    if _ALIASES_CACHE is None:
        aliases: dict[str, str] = {}
        for definition in tool_definitions():
            for alias in definition.aliases:
                aliases.setdefault(alias, definition.name)
        _ALIASES_CACHE = aliases
    return _ALIASES_CACHE


def default_registry(*, tts: Optional[bool] = None,
                     search: Optional[bool] = None,
                     chem: Optional[bool] = None,
                     artifacts: Optional[bool] = None) -> dict[str, ToolSpec]:
    """The tool set, built from the definition files (defs/tools/*.json).

    `tts` forces the speech tools on/off; None auto-detects a `tts-remote`
    binary (the tools are simply absent when there is none). `search` does
    the same for `web_search` against the `ddgs` package, and `chem` for
    the chemInventoryAPI tools against the CLI. `artifacts` says whether
    the host runs an artifact store (the serve API does; the CLI does not),
    gating `publish_artifact`.
    """
    if search is None:
        search = search_available()
    if tts is None:
        fallback = Path.home() / "bin" / "tts-remote"
        tts = bool(shutil.which("tts-remote")) \
            or fallback.is_file() or (fallback / "tts-remote").is_file()
    if chem is None:
        chem = chem_inventory_available()
    available = {"search": bool(search), "tts": bool(tts), "chem": bool(chem),
                 "artifacts": bool(artifacts)}
    registry: dict[str, ToolSpec] = {}
    for definition in tool_definitions():
        if definition.requires and not available.get(definition.requires):
            continue
        registry[definition.name] = ToolSpec(
            definition.name, definition.description, definition.parameters,
            HANDLERS[definition.handler],
            read_only=definition.read_only, always_safe=definition.always_safe,
            requires=definition.requires, aliases=list(definition.aliases))
    return registry


def tool_schemas(registry: dict[str, ToolSpec],
                 disabled: Optional[set[str]] = None) -> list[dict[str, Any]]:
    disabled = disabled or set()
    schemas = []
    for name, spec in registry.items():
        if name in disabled:
            continue
        schemas.append({"type": "function", "function": {
            "name": spec.name, "description": spec.description,
            "parameters": spec.parameters}})
    return schemas


#: Tool names models tend to invent for the web tools are mapped onto the
#: real ones through each definition's `aliases` list; `tool_aliases()`
#: builds that map and `resolve_tool_alias()` uses it.
#: Tool-call ids and namespaces also leak into the function name itself
#: (Ollama's `call_` ids, OpenAI's `functions.`/`tool.` styles), so an
#: unknown name is retried once with that prefix stripped -- `call_fetch_url`
#: still runs `fetch_url` instead of failing as an unknown tool.
_TOOL_NAME_PREFIXES = ("call_", "call.", "tool_", "tool.",
                       "functions.", "function.")


def _strip_tool_name_prefix(key: str) -> str:
    """The name left after one conventional tool-call prefix (else '')."""
    for prefix in _TOOL_NAME_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return ""


def resolve_tool_alias(name: str) -> str:
    """The real tool for a name a model may have invented (identity otherwise).

    The alias map is tried first; on a miss the name is stripped of a
    conventional id/namespace prefix and retried against the aliases and
    the canonical definition names, so `call_fetch_url` -> `fetch_url` and
    `tool_run_shell` -> `run_shell`.
    """
    key = str(name or "").strip().lower().replace("-", "_")
    aliases = tool_aliases()
    match = aliases.get(key)
    if match:
        return match
    stripped = _strip_tool_name_prefix(key)
    if stripped:
        match = aliases.get(stripped)
        if match:
            return match
        if stripped in {definition.name for definition in tool_definitions()}:
            return stripped
    return name
