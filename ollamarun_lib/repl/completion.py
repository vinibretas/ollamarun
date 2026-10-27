"""Tab completion for the inline prompt and the TUI input box."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..constants import Completer, Completion
from ..interactive import MODEL_SORT_KEYS
from ..tools import resolve_tool_alias
from .commands.tables import (COMMAND_INFO, TOOLS_POLICIES, TOOLS_SUBCOMMANDS,
                              TOOLS_TOOL_ACTIONS)

if TYPE_CHECKING:
    from .core import Repl


#: `/image` takes a whole path as one argument, spaces included, so its
#: completion replaces everything after this prefix rather than the last word.
_IMAGE_PREFIX = "/image "


def completion_fragment(text: str) -> str:
    """The part of `text` a completion replaces: the last token, an `@file`
    mention (including its `@`), or `/image`'s whole path argument.

    Shared with the TUI input box.
    """
    if text.startswith(_IMAGE_PREFIX):
        return text[len(_IMAGE_PREFIX):]
    at_match = re.search(r"@([^\s@]*)$", text)
    if at_match:
        return at_match.group(0)
    return text.split(" ")[-1]


def completion_candidates(repl: "Repl", text: str) -> list[str]:
    """Everything that completes `text`: `/commands`, their arguments, `@files`.

    Shared by the prompt_toolkit completer and the TUI input box so Tab
    suggests the same things in both front ends.
    """
    if text.startswith(_IMAGE_PREFIX):
        return _image_candidates(repl, text[len(_IMAGE_PREFIX):])
    at_match = re.search(r"@([^\s@]*)$", text)
    if at_match:
        fragment = at_match.group(1)
        parent = Path(fragment).parent if fragment else Path(".")
        try:
            entries = sorted(parent.iterdir())
        except OSError:
            return []
        return ["@" + entry.name + ("/" if entry.is_dir() else "")
                for entry in entries if entry.name.startswith(Path(fragment).name)]
    if not text.startswith("/"):
        return []
    if " " not in text:
        return [name for command in COMMAND_INFO for name in command.split(", ")
                if name.startswith(text)]
    command, _, argument = text.partition(" ")
    if command in {"/model", "/models"} and argument.startswith("--sort "):
        fragment = argument[len("--sort "):]
        return [key for key in MODEL_SORT_KEYS + ("fuzzy",)
                if key.startswith(fragment)]
    if command == "/tools":
        return _tools_completion(repl, argument)
    if " " in argument:
        return []
    candidates: list[str] = []
    if command in {"/model", "/models"}:
        candidates += [flag for flag in ("--list", "--sort")
                       if flag.startswith(argument)]
    if command in {"/model", "/models", "/load", "/save", "/show"}:
        candidates += [name for name in repl.model_names if name.startswith(argument)]
    if command in {"/load", "/sessions", "/branch"}:
        candidates += [entry["id"] for entry in repl.store.entries()[:25]
                       if entry["id"].startswith(argument)]
    if command == "/queue":
        candidates += [word for word in ("list", "clear", "rm") if word.startswith(argument)]
    if command == "/thinking":
        candidates += [word for word in ("show", "collapse", "hide", "toggle", "full")
                       if word.startswith(argument)]
    return candidates


def _image_candidates(repl: "Repl", argument: str) -> list[str]:
    """Paths for `/image`: `/NAME` inside the screenshots root, else files.

    A `/`-prefixed argument is the virtual root first -- the screenshots
    directory (`repl.screenshots_dir`) -- and falls back to a real absolute
    path when nothing there matches the prefix. Other arguments complete
    relative to the cwd (`~` expands); a bare word also offers `list`/`clear`.
    """
    if argument.startswith("/"):
        candidates = _screenshot_candidates(repl.screenshots_dir, argument)
        return candidates or _path_candidates(argument)
    if argument.startswith("~") or "/" in argument:
        return _path_candidates(argument)
    words = [word for word in ("list", "clear") if word.startswith(argument)]
    if not argument:
        return words
    return words + _path_candidates(argument)


def _screenshot_candidates(root: Optional[Path],
                           fragment: str) -> list[str]:
    """`/NAME` completion inside the screenshots root, newest first.

    Screenshot filenames carry dates, so the latest entries are what a bare
    `/` usually wants. Empty when the root is unset or missing, so the caller
    falls back to a real absolute path.
    """
    if root is None:
        return []
    relative = fragment[1:]
    head, _, name = relative.rpartition("/")
    directory = root / head if head else root
    try:
        entries = list(directory.iterdir())
    except OSError:
        return []

    def mtime(entry: Path) -> float:
        try:
            return entry.stat().st_mtime
        except OSError:
            return 0.0

    prefix = f"/{head}/" if head else "/"
    candidates: list[str] = []
    for entry in sorted(entries, key=lambda item: (-mtime(item), item.name)):
        if entry.name.startswith(name):
            candidates.append(prefix + entry.name
                              + ("/" if entry.is_dir() else ""))
    return candidates


def _path_candidates(fragment: str) -> list[str]:
    """Filesystem completion for one path fragment (quotes preserved).

    A leading quote is stripped for the lookup and re-applied to every
    candidate; an unquoted name with spaces is quoted too, because `/image`
    still splits its argument with `shlex` outside the virtual root.
    """
    quote = fragment.startswith('"')
    text = fragment[1:] if quote else fragment
    if text.endswith('"'):
        quote = True
        text = text[:-1]
    expanded = os.path.expanduser(text)
    if expanded.endswith("/"):
        parent = Path(expanded)
        name = ""
    else:
        parent = Path(expanded).parent
        name = Path(text).name
    head = text[: len(text) - len(name)] if name else text
    try:
        entries = sorted(parent.iterdir())
    except OSError:
        return []
    candidates: list[str] = []
    for entry in entries:
        if not entry.name.startswith(name):
            continue
        label = head + entry.name + ("/" if entry.is_dir() else "")
        if quote or " " in entry.name:
            label = f'"{label}"'
        candidates.append(label)
    return candidates


def tool_name_candidates(repl: "Repl", partial: str) -> list[str]:
    """Tool names for a `/tools` completion, plus aliases once a prefix exists.

    An empty `partial` lists just the registry's own names — the alias lists
    are long (they collect every invented name a model might use) and would
    flood a bare Tab; a typed prefix is a clear intent, so aliases join in.
    """
    names: set[str] = set(repl.tool_registry)
    if partial:
        for spec in repl.tool_registry.values():
            names.update(spec.aliases)
    return sorted(name for name in names if name.startswith(partial))


def _tools_completion(repl: "Repl", argument: str) -> list[str]:
    """Tab completion inside `/tools …`: subcommands, tool names, per-tool
    actions and the `rules`/`policy` keywords. `argument` keeps its trailing
    space, so the token being typed is the last one (possibly empty)."""
    tokens = argument.split(" ")
    partial = tokens[-1].lower()
    if len(tokens) == 1:
        return ([word for word in TOOLS_SUBCOMMANDS if word.startswith(partial)]
                + tool_name_candidates(repl, partial))
    first = tokens[0].lower()
    if first in {"enable", "disable"}:
        return tool_name_candidates(repl, partial)
    if first == "help":
        return tool_name_candidates(repl, partial) if len(tokens) == 2 else []
    if first == "policy":
        return ([mode for mode in TOOLS_POLICIES if mode.startswith(partial)]
                if len(tokens) == 2 else [])
    if first == "rules":
        if len(tokens) == 2:
            return [word for word in ("add", "rm") if word.startswith(partial)]
        if len(tokens) == 3 and tokens[1].lower() == "add":
            return [word for word in ("shell", "path", "python", "call")
                    if word.startswith(partial)]
        return []
    if resolve_tool_alias(first) not in repl.tool_registry:
        return []
    if len(tokens) == 2:
        return [action for action in TOOLS_TOOL_ACTIONS if action.startswith(partial)]
    return []


class SlashCompleter(Completer):  # type: ignore[misc]
    def __init__(self, repl: "Repl"):
        self.repl = repl

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        fragment = completion_fragment(text)
        for candidate in completion_candidates(self.repl, text):
            yield Completion(candidate, start_position=-len(fragment))
