"""Model-facing prompt text, loaded from files instead of code.

The defaults ship as files in `ollamarun_lib/defs/prompts/` (`system.md`,
`system_tools.md`, `title.md`, `summarize.md`); a same-named file in
`~/etc/config_files/ollamarun-defs/prompts/` replaces the shipped one, so
prompts can be edited without touching Python. `title.md` and
`summarize.md` use a `{{content}}` placeholder for the text they wrap
(missing placeholder: the content is appended after a blank line).

Every loader degrades gracefully: a missing or unreadable file warns once
and returns None so the caller can fall back, and default-path reads are
cached for the life of the process (edits are picked up on the next launch).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .constants import DEFS_DIR, DEFS_OVERRIDE_DIR, console

#: The prompt files the app expects to find (for docs and tests).
PROMPT_NAMES = ("system.md", "system_tools.md", "title.md", "summarize.md")

_CACHE: dict[str, Optional[str]] = {}
_WARNED: set[str] = set()


def prompt_dirs() -> list[Path]:
    """The prompt layers, lowest priority first (config overrides ship last)."""
    return [DEFS_DIR / "prompts", DEFS_OVERRIDE_DIR / "prompts"]


def load_prompt(name: str,
                directories: Optional[list[Path]] = None) -> Optional[str]:
    """One prompt file: the override layer wins, the shipped default backs it.

    Without explicit `directories` the default layers are used and the result
    is cached; a file that exists but cannot be read warns and reads as None.
    """
    if directories is None and name in _CACHE:
        return _CACHE[name]
    if directories is not None:
        folders = [Path(directory) for directory in directories]
    else:
        folders = prompt_dirs()
    text: Optional[str] = None
    for folder in reversed(folders):  # highest-priority layer first
        path = folder / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            if str(path) not in _WARNED:
                _WARNED.add(str(path))
                console.print(f"[warn]cannot read prompt {path}: {exc}[/warn]")
            text = None
        break
    else:
        # Nothing in any layer: only a real problem for the shipped set, and
        # worth saying once -- a user editing overrides gets no false alarm.
        shipped = folders[0] / name if folders else None
        if shipped is not None and str(shipped) not in _WARNED:
            _WARNED.add(str(shipped))
            console.print(f"[warn]prompt file not found: {shipped}[/warn]")
    if directories is None:
        _CACHE[name] = text
    return text


def render_prompt(template: str, content: str) -> str:
    """Substitute `{{content}}`; append the content when the marker is absent."""
    if "{{content}}" in template:
        return template.replace("{{content}}", content)
    return f"{template}\n\n{content}"


def default_system_prompt(with_tools: bool = True) -> Optional[str]:
    """The built-in system prompt: base text plus the tool rules when enabled.

    None when `system.md` is missing -- the caller should then send nothing.
    """
    base = load_prompt("system.md")
    if base is None:
        return None
    parts = [base]
    if with_tools:
        tools = load_prompt("system_tools.md")
        if tools:
            parts.append(tools)
    return "\n\n".join(parts)


def title_prompt(first_text: str) -> Optional[str]:
    """The auto-title request for a chat's first message (None: no file)."""
    template = load_prompt("title.md")
    return None if template is None else render_prompt(template, first_text[:1500])


def summarize_prompt(transcript: str) -> Optional[str]:
    """The `/summarize` request wrapping the transcript (None: no file)."""
    template = load_prompt("summarize.md")
    return None if template is None else render_prompt(template, transcript)
