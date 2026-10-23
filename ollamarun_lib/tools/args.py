"""Tool-call arguments: fingerprints, panel summaries and normalization."""
from __future__ import annotations

import json
from typing import Any, Optional



def call_fingerprint(name: str, args: dict[str, Any]) -> str:
    basis = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    return f"{name}:{basis}"


def call_detail(name: str, args: dict[str, Any]) -> str:
    """One-line human summary of a tool call for the result panel."""
    if name in {"run_shell", "run_python"}:
        text = str(args.get("command") if name == "run_shell" else args.get("code") or "")
    elif name == "write_file":
        text = f"{args.get('path')} ({len(str(args.get('content') or ''))} chars)"
    elif name in {"read_file", "read_pdf", "list_dir"}:
        text = str(args.get("path") or args.get("url") or ".")
    elif name == "edit_file":
        old = str(args.get("old_string") or "")
        new = str(args.get("new_string") or "")
        text = (f"{args.get('path')} ({len(old.splitlines())} → "
                f"{len(new.splitlines())} lines)")
    elif name in {"make_dir", "delete_file"}:
        text = str(args.get("path") or "")
    elif name in {"move_file", "copy_file"}:
        text = f"{args.get('source')} → {args.get('destination')}"
    elif name == "search_sessions":
        text = str(args.get("query") or "")
    elif name == "list_models":
        text = "loaded only" if args.get("loaded_only") else "installed"
    elif name == "system_status":
        text = str(args.get("target") or "ollama")
    elif name == "context_status":
        text = "window and usage"
    elif name == "session_info":
        text = "session and environment"
    elif name == "chem_search":
        text = str(args.get("query") or "")
    elif name == "chem_container":
        text = f"container {args.get('id')}"
    elif name == "chem_locations":
        text = str(args.get("location") or "all locations")
    elif name == "fetch_url":
        text = str(args.get("url") or "")
    elif name == "web_search":
        text = str(args.get("query") or "")
    elif name == "speak":
        text = str(args.get("text") or "")
    else:
        text = json.dumps(args, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= 160 else text[:159] + "…"


def normalize_arguments(raw: Any) -> Optional[dict[str, Any]]:
    """Tool-call arguments may arrive as a dict or a JSON string."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except ValueError:
            return None
        return value if isinstance(value, dict) else None
    return None
