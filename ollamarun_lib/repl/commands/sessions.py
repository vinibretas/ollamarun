"""Session commands: `/load`, `/save`, `/clear`, `/new`, `/sessions`, `/title`, `/tag`, `/undo`, `/retry`, `/edit`, `/branch`."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rich.markup import escape

from ...client import OllamaError
from ...constants import console
from ...helpers import new_session_id, now_iso, remote_node, shortcut
from ...interactive import ensure_model, pick_session
from ...session import Session
from ..session import _switch_session

if TYPE_CHECKING:
    from ..core import Repl


def cmd_load(repl: Repl, argument: str) -> None:
    if not argument:
        session = pick_session(repl.store, ui=repl.ui)
        if session is not None:
            _switch_session(repl, session)
        return
    wants_model = argument.startswith("--model")
    if not wants_model:
        session = repl.store.resolve(argument)
        if session is not None:
            _switch_session(repl, session)
            return
    name = argument[len("--model"):].strip() if wants_model else argument
    if not name:
        console.print("[err]usage: /load [MODEL|SESSION][/err]")
        return
    ssh_target = remote_node(repl.session.remote, repl.config) if repl.session.remote else None
    name = ensure_model(repl.client, name, repl.config, ssh_target=ssh_target,
                        host_label=repl.session.remote, ui=repl.ui)
    if not name:
        return
    if name == repl.session.model:
        console.print("[dim]already using that model[/dim]")
        return
    if repl.session.messages:
        keep = repl.ui.confirm(f"switch to [key]{escape(name)}[/key] and keep the "
                               f"{len(repl.session.messages)}-message context?",
                               default=False)
    else:
        keep = False
    repl.unload_model()
    repl.session.model = name
    if not keep:
        repl.session.messages = []
        repl.session.stats = {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0,
                              "duration_s": 0.0}
    repl.session.touch()
    console.print(f"[ok]model[/ok] {escape(name)}"
                  + ("" if keep else " · context cleared"))


def cmd_save(repl: Repl, argument: str) -> None:
    name = argument.strip()
    if not name:
        console.print("[err]usage: /save NAME[/err]")
        return
    if repl.client.model_exists(name) and not repl.ui.confirm(
            f"[warn]{escape(name)} already exists — overwrite?[/warn]", default=False):
        return
    console.print(f"[dim]creating {escape(name)} from {escape(repl.session.model)}…[/dim]")

    def progress(status: str) -> None:
        console.print(f"[dim]  {escape(status)}[/dim]")

    try:
        repl.client.create_model(name, repl.session.model,
                                 system=repl.effective_system(),
                                 parameters=repl.session.parameters,
                                 progress=progress)
    except OllamaError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    console.print(f"[ok]created[/ok] {escape(name)} — `ollamarun {escape(name)}`")



def cmd_clear(repl: Repl, argument: str) -> None:
    repl.session.messages = []
    repl.session.stats = {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0,
                          "duration_s": 0.0}
    repl.session.id = new_session_id()
    repl.session.created_at = repl.session.updated_at = now_iso()
    repl.session.title = ""
    if not repl.no_save:
        repl.store.save(repl.session)
    console.print("[ok]context cleared[/ok]")


def cmd_new(repl: Repl, argument: str) -> None:
    title = argument.strip()
    model = repl.session.model
    system = repl.session.system
    parameters = dict(repl.session.parameters)
    if not repl.no_save:
        repl.store.save(repl.session)
    repl.session = Session.create(model, repl.client.base, title=title, system=system,
                                  parameters=parameters,
                                  gpu_layers=repl.session.gpu_layers)
    console.print(f"[ok]new session[/ok] {repl.session.id}")


def cmd_sessions(repl: Repl, argument: str) -> None:
    session = pick_session(repl.store, filter_text=argument, ui=repl.ui)
    if session is not None:
        _switch_session(repl, session)


def cmd_title(repl: Repl, argument: str) -> None:
    if not argument:
        console.print(f"title: [key]{escape(repl.session.display_title())}[/key]")
        return
    if argument == "--auto":
        first = next((message.get("content", "") for message in repl.session.messages
                      if message.get("role") == "user"), "")
        repl.session.title = ""
        repl._maybe_title(first)
        console.print(f"[ok]title[/ok] {escape(repl.session.display_title())}")
        return
    repl.session.title = shortcut(argument, 80)
    repl.session.touch()
    console.print(f"[ok]title[/ok] {escape(repl.session.title)}")


def cmd_tag(repl: Repl, argument: str) -> None:
    parts = argument.split()
    if not parts or parts[0] == "list":
        console.print("tags: " + (", ".join(repl.session.tags) or "[dim]none[/dim]"))
        return
    if parts[0] == "add":
        for tag in parts[1:]:
            if tag not in repl.session.tags:
                repl.session.tags.append(tag)
        repl.session.touch()
        console.print(f"[ok]tags[/ok] {', '.join(repl.session.tags)}")
    elif parts[0] in {"rm", "remove", "del"}:
        for tag in parts[1:]:
            if tag in repl.session.tags:
                repl.session.tags.remove(tag)
        repl.session.touch()
        console.print(f"[ok]tags[/ok] {', '.join(repl.session.tags) or 'none'}")
    else:
        console.print("[err]usage: /tag [list|add TAG…|rm TAG][/err]")



def cmd_undo(repl: Repl, argument: str) -> None:
    if not repl.session.messages:
        console.print("[dim]nothing to undo[/dim]")
        return
    # Drop everything up to and including the last user message, so a turn
    # that used tools (assistant tool-call + tool results + answer) goes away
    # as one unit.
    while repl.session.messages and repl.session.messages[-1].get("role") != "user":
        repl.session.messages.pop()
    if repl.session.messages and repl.session.messages[-1].get("role") == "user":
        repl.session.messages.pop()
    repl.session.stats["turns"] = max(0, repl.session.stats.get("turns", 1) - 1)
    repl.session.touch()
    if not repl.no_save:
        repl.store.save(repl.session)
    console.print("[ok]undone[/ok]")


def cmd_retry(repl: Repl, argument: str) -> None:
    index = repl.session.last_user_index()
    if index is None:
        console.print("[dim]nothing to retry[/dim]")
        return
    del repl.session.messages[index + 1:]
    if not repl.no_save:
        repl.store.save(repl.session)
    console.print("[dim]retrying…[/dim]")
    repl._generate()


def cmd_edit(repl: Repl, argument: str) -> None:
    index = repl.session.last_user_index()
    if index is None:
        console.print("[dim]no message to edit[/dim]")
        return
    original = repl.session.messages[index].get("content", "")
    edited = repl.ui.editor(original)
    if edited is None or not edited.strip():
        console.print("[warn]edit cancelled[/warn]")
        return
    repl.session.messages[index]["content"] = edited.strip()
    del repl.session.messages[index + 1:]
    repl.session.touch()
    if not repl.no_save:
        repl.store.save(repl.session)
    console.print("[dim]resending edited message…[/dim]")
    repl._generate()


def cmd_branch(repl: Repl, argument: str) -> None:
    branch = Session.create(repl.session.model, repl.client.base,
                            title=argument.strip() or f"{repl.session.display_title()} (branch)",
                            system=repl.session.system,
                            parameters=dict(repl.session.parameters))
    branch.messages = json.loads(json.dumps(repl.session.messages))
    branch.think = repl.session.think
    branch.keepalive = repl.session.keepalive
    branch.format = repl.session.format
    branch.gpu_layers = repl.session.gpu_layers
    branch.tags = list(repl.session.tags)
    branch.stats = dict(repl.session.stats)
    branch.tool_allow = list(repl.session.tool_allow)
    branch.tool_rules = [dict(entry) for entry in repl.session.tool_rules]
    branch.tools_disabled = list(repl.session.tools_disabled)
    branch.tts_auto = repl.session.tts_auto
    branch.tts_voice = repl.session.tts_voice
    branch.tts_lang = repl.session.tts_lang
    if not repl.no_save:
        repl.store.save(repl.session)
        repl.store.save(branch)
    repl.session = branch
    console.print(f"[ok]branched[/ok] {branch.id}")


