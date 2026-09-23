"""Output convention for the inspection flags: `emit()` plus the `cli_*`
functions implementing `--list`/`--search`/`--show`/`--export`/`--delete`.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Optional

from rich import box
from rich.markup import escape
from rich.prompt import Confirm
from rich.table import Table

from .constants import console
from .renderer import render_transcript
from .session import SessionStore, export_html, export_json, export_markdown


def emit(records: list[dict[str, Any]], title: str, args: argparse.Namespace,
         columns: Optional[list[str]] = None) -> None:
    if not records:
        console.print("[warn]no matching sessions[/warn]")
        return
    columns = columns or list(records[0].keys())
    if args.json:
        text = json.dumps(records, indent=2, default=str) + "\n"
    elif args.csv:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
        text = buffer.getvalue()
    elif args.markdown:
        rows = ["| " + " | ".join(columns) + " |",
                "| " + " | ".join("---" for _ in columns) + " |"]
        rows += ["| " + " | ".join(str(record.get(column, "")) for column in columns) + " |"
                 for record in records]
        text = "\n".join(rows) + "\n"
    else:
        if args.out:
            console.print("[err]--out needs --json, --csv or --markdown[/err]")
            raise SystemExit(2)
        table = Table(title=title, box=box.ROUNDED, header_style="bold magenta",
                      row_styles=["", "dim"])
        for column in columns:
            table.add_column(column)
        for record in records:
            table.add_row(*[escape(str(record.get(column, ""))) for column in columns])
        console.print(table)
        return
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        console.print(f"[ok]wrote[/ok] {args.out}")
    else:
        console.print(text, markup=False, highlight=False)


def cli_list(args: argparse.Namespace, store: SessionStore) -> int:
    rows = store.entries()
    if args.search:
        rows = [entry for entry in rows
                if args.search.lower() in json.dumps(entry, default=str).lower()]
    columns = ["id", "title", "model", "updated_at", "turns", "messages", "tags"]
    emit([{key: entry.get(key) for key in columns} for entry in rows],
         "saved sessions", args, columns)
    return 0


def cli_search(args: argparse.Namespace, store: SessionStore) -> int:
    hits = store.search(args.search)
    records = [{
        "id": session.id,
        "title": session.display_title(),
        "model": session.model,
        "updated_at": session.updated_at,
        "match": snippet,
    } for session, snippet in hits]
    emit(records, f"search: {args.search}", args,
         ["id", "title", "model", "updated_at", "match"])
    return 0


def cli_show(args: argparse.Namespace, store: SessionStore, config: dict[str, Any]) -> int:
    session = store.resolve(args.show)
    if session is None:
        console.print(f"[err]no such session: {args.show}[/err]")
        return 1
    if args.json:
        console.print(export_json(session), markup=False, highlight=False)
        return 0
    render_transcript(session, console, pager=args.pager or config.get("pager", False))
    return 0


def cli_export(args: argparse.Namespace, store: SessionStore) -> int:
    session = store.resolve(args.export)
    if session is None:
        console.print(f"[err]no such session: {args.export}[/err]")
        return 1
    fmt = args.export_format
    if fmt == "json":
        text = export_json(session)
    elif fmt == "html":
        text = export_html(session)
    else:
        text = export_markdown(session)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        console.print(f"[ok]wrote[/ok] {args.out}")
    else:
        console.print(text, markup=False, highlight=False)
    return 0


def cli_delete(args: argparse.Namespace, store: SessionStore) -> int:
    target = store.resolve(args.delete)
    if target is None:
        console.print(f"[err]no such session: {args.delete}[/err]")
        return 1
    if not args.yes:
        if not console.is_terminal:
            console.print("[err]refusing to delete without --yes on a non-tty[/err]")
            return 1
        if not Confirm.ask(f"Move session [key]{target.id}[/key] "
                           f"({escape(target.display_title())!r}) to trash?", default=False):
            console.print("[dim]kept[/dim]")
            return 0
    store.delete(target.id, use_trash=True)
    console.print(f"[ok]trashed[/ok] {target.id}")
    return 0
