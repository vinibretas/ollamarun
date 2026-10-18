"""One-shot and embedding runs: the non-REPL execution paths.

`run_once()` sends a single prompt through an already-built `Repl` (reusing
its payload/rendering plumbing) and `run_embed()` routes embedding-capable
models to `/api/embed` instead of chat.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from rich import box
from rich.markup import escape
from rich.table import Table

from .client import OllamaClient, OllamaError
from .constants import console
from .files import expand_file_mentions
from .helpers import stdout_text
from .renderer import GenerationResult, stats_from_final, user_message_box
from .repl import Repl
from .runtime import AgentRuntime
from .session import Session, SessionStore


def run_embed(client: OllamaClient, model: str, prompt: str, *,
              truncate: bool, dimensions: Optional[int], json_out: bool,
              options: Optional[dict[str, Any]] = None) -> int:
    try:
        response = client.embed(model, [prompt], truncate=truncate,
                                dimensions=dimensions, options=options)
    except OllamaError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return 1
    embeddings = response.get("embeddings") or []
    if json_out:
        stdout_text(json.dumps(response, indent=2))
        return 0
    if not embeddings:
        console.print("[err]no embeddings returned[/err]")
        return 1
    vector = embeddings[0]
    table = Table(title=f"embedding · {model}", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("vectors", str(len(embeddings)))
    table.add_row("dimensions", str(len(vector)))
    table.add_row("truncate", str(truncate).lower())
    table.add_row("min", f"{min(vector):.6f}")
    table.add_row("max", f"{max(vector):.6f}")
    table.add_row("mean", f"{sum(vector) / len(vector):.6f}")
    console.print(table)
    preview = ", ".join(f"{value:+.4f}" for value in vector[:8])
    console.print(f"[dim]first 8: [{preview}, …][/dim]")
    return 0


def _run_once_json(args: argparse.Namespace, client: OllamaClient, store: SessionStore,
                   session: Session, repl: Repl) -> int:
    """`--json`: one final API response, with or without tool rounds."""
    final: dict[str, Any] = {}

    def present_once(payload: dict[str, Any], after_tools: bool,
                     cancel: Any) -> GenerationResult:
        response = client.chat_once(payload)
        final.clear()
        final.update(response)
        message = response.get("message") or {}
        return GenerationResult(content=message.get("content", ""),
                                thinking=message.get("thinking") or "",
                                tool_calls=message.get("tool_calls") or [],
                                stats=stats_from_final(response))

    hooks = repl._turn_hooks(presenter=present_once, allow_prompt=False,
                             quiet_tools=True)
    # `--json --format json` never sends tools (the format wins), exactly as
    # the hand-written loop this replaced did.
    hooks.settings.tools_enabled = lambda: repl.tools_enabled and not session.format
    result = AgentRuntime(hooks).run()
    if result.error:
        console.print(f"[err]{escape(str(result.error))}[/err]")
        return 1
    stdout_text(json.dumps(final, indent=2))
    return 0


def run_once(args: argparse.Namespace, config: dict[str, Any], client: OllamaClient,
             store: SessionStore, session: Session, prompt_text: str,
             repl: Repl) -> int:
    augmented, included, warnings = expand_file_mentions(prompt_text,
                                                          max_kb=config.get("max_file_kb", 256))
    for warning in warnings:
        console.print(f"[warn]{escape(warning)}[/warn]")
    images = list(repl.pending_images)
    repl.pending_images = []
    session.add_message("user", augmented, images=images or None,
                        files=included or None)
    if not args.no_save:
        store.save(session)
    if not session.title:
        repl._maybe_title(prompt_text)
    quiet_unload = bool(args.raw or args.json or not console.is_terminal)
    if args.json:
        code = _run_once_json(args, client, store, session, repl)
        if args.unload:
            repl.unload_model(quiet=quiet_unload)
        return code
    if console.is_terminal:
        console.print(user_message_box(prompt_text, console))
    result = repl._generate()
    if args.unload:
        repl.unload_model(quiet=quiet_unload)
    if result.error:
        return 1
    if result.cancelled:
        return 130
    return 0
