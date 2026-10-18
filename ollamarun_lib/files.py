"""`@file` mentions: inlining local files into a prompt."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_FILE_MENTION_RE = re.compile(r"(?<!\w)@([^\s@]+)")

#: Sentence punctuation a mention may run into (`@notes.txt.`); the path is
#: tried exactly as typed first, then with these stripped, so a file whose
#: name really ends in punctuation still resolves.
_MENTION_TRAILING = ".,;:!?)]}>'\"…"


def _mention_file(raw: str, base: Path) -> Optional[Path]:
    """The file a mention capture names, or None.

    The raw path wins; the fallback strips trailing sentence punctuation
    before deciding "no such file", leaving the punctuation in the prompt.
    """
    for candidate_text in (raw, raw.rstrip(_MENTION_TRAILING)):
        if not candidate_text:
            continue
        candidate = Path(os.path.expanduser(candidate_text))
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def expand_file_mentions(text: str, base: Optional[Path] = None,
                         max_kb: int = 256) -> tuple[str, list[str], list[str]]:
    """Inline @path files as <file> blocks; returns (text, included, warnings)."""
    base = base or Path.cwd()
    included: list[str] = []
    warnings: list[str] = []
    appended: list[str] = []
    for match in _FILE_MENTION_RE.finditer(text):
        raw = match.group(1)
        candidate = _mention_file(raw, base)
        if candidate is None:
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
