"""Attachments and the transcript: `/image`, `/copy`, `/export`."""
from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rich.markup import escape

from ...constants import console
from ...helpers import human_bytes
from ...images import encode_image
from ...interactive import _CODE_BLOCK_RE
from ...session import export_html, export_json, export_markdown

if TYPE_CHECKING:
    from ..core import Repl

from ..view import _last_assistant


def _virtual_image_path(root: Optional[Path], text: str) -> Optional[Path]:
    """`/NAME` inside the screenshots root (`repl.screenshots_dir`), if a file."""
    if root is None or not text.startswith("/"):
        return None
    candidate = root / text[1:]
    return candidate if candidate.is_file() else None


def _attach_image(repl: Repl, path: Path) -> None:
    try:
        repl.pending_images.append(encode_image(path, repl.image_storage))
        console.print(f"[ok]attached[/ok] {escape(path.name)}")
    except (OSError, ValueError) as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")


def cmd_image(repl: Repl, argument: str) -> None:
    if not argument or argument == "list":
        if repl.pending_images:
            for image in repl.pending_images:
                console.print(f"[dim]pending:[/dim] {escape(image['name'])} "
                              f"({human_bytes(image.get('size'))})")
        else:
            console.print("[dim]no pending images[/dim]")
        return
    if argument == "clear":
        repl.pending_images = []
        console.print("[ok]cleared[/ok]")
        return
    # The whole argument first: a `/`-prefixed path may resolve inside the
    # screenshots root, spaces included (`/image /Screenshot from 2026….png`).
    whole = _virtual_image_path(repl.screenshots_dir, argument.strip())
    if whole is not None:
        _attach_image(repl, whole)
        return
    try:
        tokens = shlex.split(argument)
    except ValueError as exc:
        console.print(f"[err]{exc}[/err]")
        return
    for token in tokens:
        path = _virtual_image_path(repl.screenshots_dir, token)
        _attach_image(repl, path or Path(os.path.expanduser(token)))


def cmd_copy(repl: Repl, argument: str) -> None:
    message = _last_assistant(repl)
    if message is None:
        console.print("[dim]no response to copy[/dim]")
        return
    content = message["content"]
    if argument:
        blocks = _CODE_BLOCK_RE.findall(content)
        if not blocks:
            console.print("[warn]no code blocks in the last response[/warn]")
            return
        try:
            index = int(argument)
            content = blocks[index - 1]
        except (ValueError, IndexError):
            console.print(f"[err]pick a code block 1..{len(blocks)}[/err]")
            return
    ok, message = repl.ui.copy(content)
    if ok:
        console.print(f"[ok]{escape(message)}[/ok]")
    else:
        console.print(f"[err]{escape(message)}[/err]")


def cmd_export(repl: Repl, argument: str) -> None:
    fmt = "md"
    out: Optional[Path] = None
    try:
        parts = shlex.split(argument)
    except ValueError:
        parts = argument.split()
    index = 0
    while index < len(parts):
        token = parts[index]
        if token in {"--format", "-f"} and index + 1 < len(parts):
            fmt = parts[index + 1]
            index += 2
            continue
        out = Path(os.path.expanduser(token))
        index += 1
    if fmt not in {"md", "markdown", "json", "html"}:
        console.print("[err]usage: /export [--format md|json|html] [FILE][/err]")
        return
    if fmt == "json":
        text = export_json(repl.session)
    elif fmt == "html":
        text = export_html(repl.session)
    else:
        text = export_markdown(repl.session)
    if out is None:
        extension = "md" if fmt in {"md", "markdown"} else fmt
        slug = re.sub(r"[^\w.-]+", "-", repl.session.display_title().lower()).strip("-")
        out = Path.cwd() / f"{slug or 'session'}-{repl.session.id}.{extension}"
    out.write_text(text, encoding="utf-8")
    console.print(f"[ok]wrote[/ok] {out}")


