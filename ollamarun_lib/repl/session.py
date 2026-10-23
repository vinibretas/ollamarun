"""Switching the active session (`/load`, `/sessions`, `/branch`)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markup import escape

from ..constants import console
from ..session import Session

if TYPE_CHECKING:
    from .core import Repl


def _switch_session(repl: Repl, session: Session) -> None:
    if repl.switch_guard is not None and not repl.switch_guard(session):
        return
    current_model = repl.session.model
    target_model = session.model
    if target_model and current_model and target_model != current_model:
        if bool(repl.session.messages):
            console.print(f"[dim]session {session.id} was last used with[/dim] "
                          f"[key]{escape(target_model)}[/key]")
            answer = repl.ui.ask(
                f"load it with [c]urrent model ({escape(current_model)}) or "
                f"[l]ast model ({escape(target_model)})?", choices=["c", "l"],
                default="l")
        else:
            answer = "l"
        if answer == "c":
            session.model = current_model
            console.print(f"[dim]keeping {escape(current_model)} for this session[/dim]")
        else:
            repl.unload_model()
    if not repl.no_save:
        repl.store.save(repl.session)
    repl.session = session
    repl.session.host = repl.client.base
    repl.session.touch()
    if session.tts_auto is not None:
        repl.tts_auto = bool(session.tts_auto)
    if session.tts_voice:
        repl.tts_voice = session.tts_voice
    if session.tts_lang:
        repl.tts_lang = session.tts_lang
    console.print(f"[ok]loaded[/ok] {session.id} · {escape(session.display_title())} "
                  f"· model {escape(session.model)}")
    repl.refresh_models()

