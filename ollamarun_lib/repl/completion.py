"""Tab completion for the inline prompt and the TUI input box."""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from ..constants import Completer, Completion
from ..interactive import MODEL_SORT_KEYS
from ..tools import resolve_tool_alias
from .commands.tables import (COMMAND_INFO, TOOLS_POLICIES, TOOLS_SUBCOMMANDS,
                              TOOLS_TOOL_ACTIONS)

if TYPE_CHECKING:
    from .core import Repl


def completion_fragment(text: str) -> str:
    """The part of `text` a completion replaces: the last token, or an `@file`
    mention (including its `@`). Shared with the TUI input box."""
    at_match = re.search(r"@([^\s@]*)$", text)
    if at_match:
        return at_match.group(0)
    return text.split(" ")[-1]


def completion_candidates(repl: "Repl", text: str) -> list[str]:
    """Everything that completes `text`: `/commands`, their arguments, `@files`.

    Shared by the prompt_toolkit completer and the TUI input box so Tab
    suggests the same things in both front ends.
    """
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
