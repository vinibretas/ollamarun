"""`@file` mentions: inlining local files into a prompt."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_FILE_MENTION_RE = re.compile(r"(?<!\w)@([^\s@]+)")


def expand_file_mentions(text: str, base: Optional[Path] = None,
                         max_kb: int = 256) -> tuple[str, list[str], list[str]]:
    """Inline @path files as <file> blocks; returns (text, included, warnings)."""
    base = base or Path.cwd()
    included: list[str] = []
    warnings: list[str] = []
    appended: list[str] = []
    for match in _FILE_MENTION_RE.finditer(text):
        raw = match.group(1)
        candidate = Path(os.path.expanduser(raw))
        if not candidate.is_absolute():
            candidate = base / candidate
        if not candidate.exists() or not candidate.is_file():
            warnings.append(f"@{raw}: no such file")
            continue
        try:
            size_kb = candidate.stat().st_size / 1024
            if size_kb > max_kb:
                warnings.append(f"@{raw}: {size_kb:.0f} KiB exceeds {max_kb} KiB limit")
                continue
            content = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            warnings.append(f"@{raw}: {exc}")
            continue
        included.append(str(candidate))
        appended.append(f'<file name="{candidate}">\n{content.rstrip()}\n</file>')
    if appended:
        text = text + "\n\n" + "\n\n".join(appended)
    return text, included, warnings
