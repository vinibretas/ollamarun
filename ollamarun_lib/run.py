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
from .renderer import stats_from_final, user_message_box
from .repl import Repl
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
        console.print(json.dumps(response, indent=2), markup=False, highlight=False)
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
    with_tools = repl.tools_enabled and not session.format
    rounds = 0
    response: dict[str, Any] = {}
    while True:
        payload = repl._payload(session.message_payloads(with_tools=with_tools),
                                stream=False, with_tools=with_tools)
        try:
            response = client.chat_once(payload)
        except OllamaError as exc:
            console.print(f"[err]{escape(str(exc))}[/err]")
            return 1
        message = response.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        session.add_message("assistant", message.get("content", ""),
                            thinking=message.get("thinking") or None,
                            tool_calls=tool_calls or None)
        json_stats = stats_from_final(response)
        session.record_turn(json_stats, count_turn=(rounds == 0))
        session.stats["last_prompt_tokens"] = json_stats.get("prompt_tokens")
        session.stats["last_completion_tokens"] = json_stats.get("completion_tokens")
        if not args.no_save:
            store.save(session)
        if not tool_calls or not with_tools:
            break
        rounds += 1
        if rounds > repl.tools_max_rounds:
            for call in tool_calls:
                function = (call.get("function") or {}) if isinstance(call, dict) else {}
                session.add_message("tool", "not executed: tool-call limit reached",
                                    tool_name=str(function.get("name") or "?"))
            break
        repl._tool_round = rounds
        repl._execute_tool_calls(tool_calls, quiet=True, allow_prompt=False)
    console.print(json.dumps(response, indent=2), markup=False, highlight=False)
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
    session.add_message("user", prompt_text, images=images or None, files=included or None)
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
