"""REPL: `Repl` drives the interactive session; each slash command
(`/set`, `/show`, `/load`, `/save`, `/undo`, `/retry`, `/edit`, `/branch`,
`/context`, `/summarize`, `/image`, `/render`, `/math`, `/pager`, `/ps`,
`/num-ctx`, ...) is a standalone `cmd_NAME(repl, argument)` function
dispatched by name through `COMMAND_HANDLERS`; add a new slash command by
adding one function here and wiring it into that table.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .client import OllamaClient, OllamaError, build_chat_payload
from .constants import (
    DEFAULT_TIMEOUT,
    HAS_PROMPT_TOOLKIT,
    HAS_PYPERCLIP,
    HISTORY_PATH,
    THINKING_MODES,
    Completer,
    Completion,
    FileHistory,
    KeyBindings,
    PromptSession,
    console,
    pyperclip,
)
from .files import expand_file_mentions
from .helpers import (
    coerce_value,
    human_bytes,
    human_chars,
    human_duration,
    naive_title,
    new_session_id,
    now_iso,
    parse_bool,
    parse_think,
    short_stamp,
    shortcut,
)
from .images import encode_image
from .interactive import (
    _CODE_BLOCK_RE,
    model_context_length,
    open_in_editor,
    pick_model,
    pick_session,
)
from .renderer import GenerationResult, Renderer, thinking_panel, user_message_box
from .session import Session, SessionStore, export_html, export_json, export_markdown


COMMAND_INFO: dict[str, tuple[str, str]] = {
    "/help, /?": ("[topic]", "show commands (topic: shortcuts)"),
    "/set": ("parameter NAME VALUE | system TEXT", "session parameters / system prompt"),
    "/show": ("[info|license|modelfile|parameters|system|template]", "model details"),
    "/load": ("[MODEL|SESSION]", "load a saved session or switch model"),
    "/save": ("NAME", "create an Ollama model from this session"),
    "/clear": ("", "forget the context, keep settings, start a new session"),
    "/new": ("", "start a brand new session"),
    "/sessions": ("[filter]", "browse and switch saved sessions"),
    "/title": ("[TEXT|--auto]", "show or set the session title"),
    "/tag": ("[list|add TAG…|rm TAG]", "session tags"),
    "/model": ("[NAME|--list]", "switch model"),
    "/ps": ("", "show resident models (ollama ps)"),
    "/num-ctx": ("[N]", "context window size, capped to the model's max"),
    "/params": ("", "show session parameters"),
    "/think": ("[on|off|low|medium|high]", "thinking mode"),
    "/thinking": ("[show|collapse|hide|toggle|full]", "thinking display (Ctrl+T while streaming)"),
    "/system": ("[TEXT|clear]", "show or set the system prompt"),
    "/undo": ("", "drop the last exchange"),
    "/retry": ("", "regenerate the last response"),
    "/edit": ("", "edit the last message in $EDITOR and resend"),
    "/branch": ("[TITLE]", "fork this conversation into a new session"),
    "/context": ("", "context-window usage"),
    "/summarize": ("", "compact older messages into a summary"),
    "/image": ("PATH… | clear", "attach images to the next message"),
    "/render": ("[markdown|plain|raw]", "response rendering mode"),
    "/math": ("[on|off]", "LaTeX→Unicode math conversion"),
    "/pager": ("[on|off]", "page long transcripts"),
    "/tps": ("[on|off]", "tokens/s footer while streaming"),
    "/copy": ("[N]", "copy the last response (or its Nth code block)"),
    "/export": ("[--format md|json|html] [FILE]", "export this session"),
    "/stats": ("", "session and last-generation stats"),
    "/bye, /exit": ("", "leave"),
}


class SlashCompleter(Completer):  # type: ignore[misc]
    def __init__(self, repl: "Repl"):
        self.repl = repl

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        at_match = re.search(r"@([^\s@]*)$", text)
        if at_match:
            fragment = at_match.group(1)
            parent = Path(fragment).parent if fragment else Path(".")
            try:
                entries = sorted(parent.iterdir())
            except OSError:
                return
            for entry in entries:
                name = entry.name + ("/" if entry.is_dir() else "")
                if name.startswith(Path(fragment).name):
                    yield Completion("@" + name, start_position=-len(at_match.group(1)) - 1)
            return
        if not text.startswith("/"):
            return
        if " " not in text:
            for command in COMMAND_INFO:
                for name in command.split(", "):
                    if name.startswith(text):
                        yield Completion(name, start_position=-len(text))
            return
        command, _, argument = text.partition(" ")
        if " " in argument:
            return
        if command in {"/model", "/load", "/save", "/show"}:
            for name in self.repl.model_names:
                if name.startswith(argument):
                    yield Completion(name, start_position=-len(argument))
        if command in {"/load", "/sessions", "/branch"}:
            for entry in self.repl.store.entries()[:25]:
                if entry["id"].startswith(argument):
                    yield Completion(entry["id"], start_position=-len(argument))


class Repl:
    def __init__(self, args: argparse.Namespace, config: dict[str, Any],
                 client: OllamaClient, store: SessionStore, session: Session):
        self.args = args
        self.config = config
        self.client = client
        self.store = store
        self.session = session
        self.model_names: list[str] = []
        self.pending_images: list[dict[str, Any]] = []
        self.auto_title = args.auto_title or config.get("auto_title", False)
        self.image_storage = config.get("image_storage", "embed")
        self.max_file_kb = config.get("max_file_kb", 256)
        self.max_render_chars = config.get("max_render_chars", 12000)
        self.timeout = args.timeout or config.get("timeout") or DEFAULT_TIMEOUT
        self.pager = args.pager or config.get("pager", False)
        self.render_mode = args.render or config.get("render", "markdown")
        self.math = config.get("math", True) and not args.no_math
        if args.hidethinking:
            self.thinking_mode = "hidden"
        elif args.show_thinking:
            self.thinking_mode = "expanded"
        else:
            self.thinking_mode = config.get("thinking", "collapsed")
        if self.thinking_mode not in THINKING_MODES:
            self.thinking_mode = "collapsed"
        self.tps_footer = False if args.no_tps else config.get("tps_footer", True)
        if args.unload:
            self.unload_on_exit = True
        elif args.keep_loaded:
            self.unload_on_exit = False
        else:
            self.unload_on_exit = config.get("unload_on_exit", True)
        self.show_stats = bool(args.verbose or config.get("show_stats", False))
        self.verbose = bool(args.verbose)
        self.no_save = bool(args.no_save)
        self._pt_session: Optional[Any] = None
        self._ps_shown_for: set[str] = set()
        self._preload_thread: Optional[threading.Thread] = None

    # ── prompt plumbing ─────────────────────────────────────────────────────

    def refresh_models(self) -> None:
        try:
            self.model_names = [model.get("name", "") for model in self.client.tags()]
        except OllamaError:
            self.model_names = []

    def _build_prompt_session(self) -> Optional[Any]:
        if not HAS_PROMPT_TOOLKIT:
            return None
        history = None
        if not os.environ.get("OLLAMA_NOHISTORY"):
            try:
                HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
                history = FileHistory(str(HISTORY_PATH))
            except OSError:
                history = None
        bindings = KeyBindings()

        @bindings.add("c-g")
        def _open_editor(event) -> None:  # pragma: no cover - interactive
            buffer = event.current_buffer
            edited = open_in_editor(buffer.text)
            if edited is not None:
                buffer.text = edited
                buffer.cursor_position = len(edited)

        @bindings.add("c-j")
        def _newline(event) -> None:  # pragma: no cover - interactive
            event.current_buffer.insert_text("\n")

        try:
            return PromptSession(history=history, completer=SlashCompleter(self),
                                 key_bindings=bindings, complete_while_typing=True)
        except Exception:
            return None

    def _read_line(self, message: str) -> str:
        if self._pt_session is not None:
            return self._pt_session.prompt(message)
        return input(message)

    # ── rendering ───────────────────────────────────────────────────────────

    def _renderer(self) -> Renderer:
        mode = self.render_mode
        if mode == "auto":
            mode = "markdown" if console.is_terminal else "plain"
        if not console.is_terminal and mode == "markdown":
            mode = "plain"
        return Renderer(console, mode=mode, math=self.math,
                        nowordwrap=self.args.nowordwrap,
                        thinking_mode=self.thinking_mode,
                        tps_footer=self.tps_footer,
                        show_stats=self.show_stats, verbose=self.verbose,
                        max_render_chars=self.max_render_chars, pager=self.pager,
                        model=self.session.model)

    # ── model housekeeping ──────────────────────────────────────────────────

    def unload_model(self, model: Optional[str] = None, quiet: bool = False) -> bool:
        """Unload a resident model (default: the session's). Never raises."""
        target = model or self.session.model
        if not target:
            return False
        try:
            if not self.client.is_resident(target):
                return False
            self.client.unload(target)
        except OllamaError as exc:
            console.print(f"[warn]could not unload {escape(target)}: "
                          f"{escape(str(exc))}[/warn]", stderr=quiet)
            return False
        if not quiet:
            console.print(f"[dim]unloaded {escape(target)}[/dim]")
        return True

    def _preload_model(self) -> None:
        """Starts loading the session's model in the background so it's
        already warm by the time the first message is sent, instead of
        waiting for the user to type something."""
        model = self.session.model
        if not model:
            return
        keepalive = self.session.keepalive or self.args.keepalive
        if console.is_terminal:
            console.print(f"[dim]loading {escape(model)}…[/dim]")

        def _load() -> None:
            try:
                self.client.load(model, keepalive=keepalive)
            except OllamaError:
                return
            if console.is_terminal:
                self._maybe_show_ps()

        thread = threading.Thread(target=_load, daemon=True)
        self._preload_thread = thread
        thread.start()

    def _maybe_show_ps(self) -> None:
        """Prints `ollama ps` the first time this session's model shows up loaded."""
        model = self.session.model
        if not model or model in self._ps_shown_for:
            return
        try:
            if not self.client.is_resident(model):
                return
            entries = self.client.ps()
        except OllamaError:
            return
        self._ps_shown_for.add(model)
        if entries:
            console.print(ps_table(entries))

    def effective_think(self) -> Optional[Any]:
        return self.session.think

    def _payload(self, messages: list[dict[str, Any]], *, stream: bool = True) -> dict[str, Any]:
        options = dict(self.session.parameters)
        return build_chat_payload(self.session.model, messages, stream=stream,
                                  fmt=self.session.format, think=self.effective_think(),
                                  keepalive=self.session.keepalive or self.args.keepalive,
                                  options=options)

    # ── generation ──────────────────────────────────────────────────────────

    def _generate(self) -> GenerationResult:
        if self._preload_thread is not None and self._preload_thread.is_alive():
            # The user sent a message before the background preload finished;
            # wait for our own load call to land (chat() would otherwise
            # trigger a redundant concurrent load) so ollama ps -- printed
            # right as the preload thread finishes -- shows up immediately
            # instead of only after this response is fully generated.
            if console.is_terminal:
                console.print(f"[dim]waiting for {escape(self.session.model)} "
                              f"to finish loading…[/dim]")
            self._preload_thread.join()
        renderer = self._renderer()
        marker = f"[ai]{escape(self.session.model)} ›[/ai]"
        if renderer.mode != "raw" and not renderer.live_render:
            console.print(marker)
        try:
            result = renderer.stream(self.client.chat(self._payload(self.session.message_payloads()),
                                                      timeout=self.timeout))
        except OllamaError as exc:
            result = GenerationResult(error=str(exc))
            console.print(f"[err]{escape(str(exc))}[/err]")
        if result.content:
            self.session.add_message("assistant", result.content,
                                     thinking=result.thinking or None,
                                     cancelled=True if result.cancelled else None,
                                     tool_calls=result.tool_calls or None)
        elif result.cancelled:
            self.session.add_message("assistant", "", cancelled=True)
        if result.stats:
            self.session.record_turn(result.stats)
            self.session.stats["last_prompt_tokens"] = result.stats.get("prompt_tokens")
        if not self.no_save:
            self.store.save(self.session)
        if console.is_terminal and not result.error:
            self._maybe_show_ps()
        return result

    def send(self, text: str) -> GenerationResult:
        text = text.strip()
        if not text and not self.pending_images:
            return GenerationResult()
        if console.is_terminal:
            console.print(user_message_box(text, console))
        augmented, included, warnings = expand_file_mentions(text, max_kb=self.max_file_kb)
        for warning in warnings:
            console.print(f"[warn]{escape(warning)}[/warn]")
        images = self.pending_images
        self.pending_images = []
        self.session.add_message("user", text, images=images or None,
                                 files=included or None)
        if not self.no_save:
            self.store.save(self.session)
        if not self.session.title:
            self._maybe_title(text)
        return self._generate()

    def _maybe_title(self, first_text: str) -> None:
        title = ""
        if self.auto_title and self.session.model:
            try:
                prompt = ("Give a short title (max 8 words, no quotes) for a chat that "
                          f"starts with:\n\n{first_text[:1500]}")
                response = self.client.chat_once(build_chat_payload(
                    self.session.model, [{"role": "user", "content": prompt}],
                    stream=False, options={"num_predict": 24, "temperature": 0.2}))
                title = (response.get("message") or {}).get("content", "").strip().strip('"')
            except OllamaError:
                title = ""
        self.session.title = shortcut(title or naive_title(first_text), 60)

    # ── command dispatch ────────────────────────────────────────────────────

    def run(self) -> int:
        self._preload_model()
        self.refresh_models()
        self._pt_session = self._build_prompt_session()
        self._banner()
        self._context_warning()
        try:
            while True:
                prompt_text = f"{shortcut(self.session.model, 24)} › "
                try:
                    line = self._read_line(prompt_text)
                except KeyboardInterrupt:
                    console.print("[dim]use /bye (or Ctrl+D) to exit[/dim]")
                    continue
                except EOFError:
                    console.print()
                    break
                line = line.strip()
                if not line and not self.pending_images:
                    continue
                if line.startswith("/"):
                    if not self._dispatch(line):
                        break
                    continue
                try:
                    result = self.send(line)
                except KeyboardInterrupt:
                    console.print("[warn]cancelled[/warn]")
                    continue
                if result.cancelled:
                    continue
                if self.config.get("auto_compact") == "auto":
                    self._maybe_auto_compact()
        finally:
            if self.unload_on_exit:
                self.unload_model()
        if not self.no_save:
            self.store.save(self.session)
            console.print(f"[dim]session {self.session.id} saved · {self.store.dir}[/dim]")
        else:
            console.print("[dim]--no-save: session not persisted[/dim]")
        return 0

    def _banner(self) -> None:
        model_count = len(self.model_names)
        lines = [
            f"[key]model[/key]   {escape(self.session.model)}",
            f"[key]session[/key] {self.session.id} · {escape(self.session.display_title())}",
            f"[key]host[/key]    {escape(self.client.base)}",
            f"[key]render[/key]  {self.render_mode}"
            + ("" if self.math else " · math off")
            + f" · thinking {self.thinking_mode}"
            + f" · {model_count} local models",
        ]
        if self.session.messages:
            lines.append(f"[key]context[/key] {len(self.session.messages)} messages")
        lines.append("[dim]/help for commands · Ctrl+D to exit · Ctrl+T thinking[/dim]")
        console.print(Panel("\n".join(lines), title="ollamarun", border_style="cyan",
                            title_align="left"))

    def _context_warning(self) -> None:
        if not self.session.messages:
            return
        try:
            show = self.client.show(self.session.model)
        except OllamaError:
            return
        total = model_context_length(show, self.session.parameters)
        used = self.session.stats.get("last_prompt_tokens")
        if total and used and used / total >= 0.8:
            console.print(f"[warn]context {used}/{total} tokens "
                          f"({used * 100 // total}%) — /summarize to compact[/warn]")

    def _maybe_auto_compact(self) -> None:
        try:
            show = self.client.show(self.session.model)
        except OllamaError:
            return
        total = model_context_length(show, self.session.parameters)
        used = self.session.stats.get("last_prompt_tokens")
        if total and used and used / total >= 0.85 and len(self.session.messages) > 4:
            console.print("[warn]context almost full — summarizing older messages[/warn]")
            cmd_summarize(self, "auto")

    def _dispatch(self, line: str) -> bool:
        name, _, argument = line.partition(" ")
        handler = COMMAND_HANDLERS.get(name)
        if handler is None:
            console.print(f"[err]unknown command {escape(name)}[/err] — /help")
            return True
        try:
            outcome = handler(self, argument.strip())
        except KeyboardInterrupt:
            console.print("[warn]cancelled[/warn]")
            return True
        return True if outcome is None else bool(outcome)


# ── commands ────────────────────────────────────────────────────────────────

def cmd_help(repl: Repl, argument: str) -> None:
    if argument == "shortcuts":
        table = Table(title="shortcuts", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("key")
        table.add_column("action")
        for key, action in (("Ctrl+D", "exit"),
                            ("Ctrl+C", "cancel input / generation"),
                            ("Ctrl+T", "expand thinking & open it in the pager "
                                       "(again to collapse)"),
                            ("Ctrl+G", "$OLLAMA_EDITOR for the current prompt"),
                            ("Ctrl+J", "insert newline"),
                            ("Ctrl+R", "search prompt history"),
                            ("Up/Down", "history navigation"),
                            ("Tab", "complete commands, ids, models, @files"),
                            ("@path", "inline a file into the prompt")):
            table.add_row(key, action)
        console.print(table)
        return
    table = Table(title="commands", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("command")
    table.add_column("usage", style="dim")
    table.add_column("description")
    for command, (usage, description) in COMMAND_INFO.items():
        table.add_row(escape(command), escape(usage), description)
    console.print(table)


def cmd_set(repl: Repl, argument: str) -> None:
    try:
        parts = shlex.split(argument)
    except ValueError as exc:
        console.print(f"[err]{exc}[/err]")
        return
    if not parts:
        cmd_params(repl, "")
        return
    kind = parts[0]
    if kind == "parameter":
        if len(parts) < 2:
            console.print("[err]usage: /set parameter NAME [VALUE][/err]")
            return
        if len(parts) == 2:
            repl.session.parameters.pop(parts[1], None)
            console.print(f"[ok]cleared[/ok] parameter {escape(parts[1])}")
        else:
            repl.session.parameters[parts[1]] = coerce_value(" ".join(parts[2:]))
            console.print(f"[ok]set[/ok] {escape(parts[1])} = "
                          f"{escape(json.dumps(repl.session.parameters[parts[1]]))}")
            repl.session.touch()
    elif kind == "system":
        text = argument[len("system"):].strip()
        repl.session.system = text or None
        console.print("[ok]system prompt set[/ok]" if text else "[ok]system prompt cleared[/ok]")
        repl.session.touch()
    else:
        console.print("[err]usage: /set parameter NAME VALUE | /set system TEXT[/err]")


def cmd_params(repl: Repl, argument: str) -> None:
    table = Table(title="session parameters", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("parameter")
    table.add_column("value")
    if repl.session.parameters:
        for name, value in repl.session.parameters.items():
            table.add_row(name, escape(json.dumps(value)))
    else:
        table.add_row("[dim]none[/dim]", "")
    table.add_row("think", escape(str(repl.session.think)))
    table.add_row("keep_alive", escape(str(repl.session.keepalive or "")))
    table.add_row("format", escape(str(repl.session.format or "")))
    table.add_row("system", shortcut(repl.session.system or "", 60))
    console.print(table)


def cmd_show(repl: Repl, argument: str) -> None:
    topic = (argument or "info").split()[0]
    try:
        data = repl.client.show(repl.session.model)
    except OllamaError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    if topic in {"info"}:
        details = data.get("details") or {}
        table = Table(title=f"model · {repl.session.model}", box=box.ROUNDED,
                      header_style="bold magenta")
        table.add_column("field")
        table.add_column("value")
        for field_name, value in (
            ("family", details.get("family")),
            ("parameter size", details.get("parameter_size")),
            ("quantization", details.get("quantization_level")),
            ("format", details.get("format")),
            ("capabilities", ", ".join(data.get("capabilities") or [])),
            ("context length", model_context_length(data, repl.session.parameters)),
            ("modified", data.get("modified_at")),
        ):
            table.add_row(field_name, escape(str(value if value is not None else "")))
        console.print(table)
        cmd_params(repl, "")
        return
    key_map = {"license": "license", "modelfile": "modelfile", "parameters": "parameters",
               "system": "system", "template": "template"}
    if topic not in key_map:
        console.print("[err]usage: /show [info|license|modelfile|parameters|system|template][/err]")
        return
    value = data.get(key_map[topic]) or "(empty)"
    if topic in {"modelfile", "template"}:
        from rich.syntax import Syntax
        console.print(Panel(Syntax(str(value), "dockerfile" if topic == "modelfile" else "text"),
                            title=topic, border_style="dim", title_align="left"))
    else:
        console.print(Panel(escape(str(value)), title=topic, border_style="dim",
                            title_align="left"))


def ps_table(entries: list[dict[str, Any]]) -> Table:
    table = Table(title="ollama ps", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("model")
    table.add_column("size")
    table.add_column("VRAM")
    table.add_column("quant")
    table.add_column("until")
    for entry in entries:
        details = entry.get("details") or {}
        table.add_row(
            escape(str(entry.get("name") or entry.get("model") or "?")),
            human_bytes(entry.get("size")),
            human_bytes(entry.get("size_vram")),
            escape(str(details.get("quantization_level") or "?")),
            short_stamp(entry.get("expires_at")) or "?")
    return table


def cmd_ps(repl: Repl, argument: str) -> None:
    try:
        entries = repl.client.ps()
    except OllamaError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    if not entries:
        console.print("[dim]no models currently loaded[/dim]")
        return
    console.print(ps_table(entries))


def resolve_num_ctx(client: OllamaClient, model: str, requested: int) -> tuple[int, Optional[int]]:
    """Caps `requested` to the model's max context length (from /api/show).

    Returns (value_to_use, max_ctx_or_None). Never raises: an unreachable
    server just means the max is unknown and the request passes through.
    """
    try:
        show_data = client.show(model)
    except OllamaError:
        return requested, None
    max_ctx = model_context_length(show_data, {})
    if max_ctx and requested > max_ctx:
        return max_ctx, max_ctx
    return requested, max_ctx


def cmd_num_ctx(repl: Repl, argument: str) -> None:
    argument = argument.strip()
    if not argument:
        current = repl.session.parameters.get("num_ctx")
        try:
            max_ctx = model_context_length(repl.client.show(repl.session.model), {})
        except OllamaError:
            max_ctx = None
        bits = [f"num_ctx = {current if current is not None else '(model default)'}"]
        if max_ctx:
            bits.append(f"model max {max_ctx}")
        console.print(f"[key]context size[/key] " + " · ".join(bits))
        return
    try:
        requested = int(argument)
    except ValueError:
        console.print("[err]usage: /num-ctx N[/err]")
        return
    if requested <= 0:
        console.print("[err]--num-ctx must be a positive integer[/err]")
        return
    applied, max_ctx = resolve_num_ctx(repl.client, repl.session.model, requested)
    if max_ctx and requested > max_ctx:
        console.print(f"[warn]{requested} exceeds {escape(repl.session.model)}'s max context "
                      f"({max_ctx}) — capped to {max_ctx}[/warn]")
    repl.session.parameters["num_ctx"] = applied
    repl.session.touch()
    console.print(f"[ok]num_ctx set to {applied}[/ok]")


def _switch_session(repl: Repl, session: Session) -> None:
    current_model = repl.session.model
    target_model = session.model
    if target_model and current_model and target_model != current_model:
        if bool(repl.session.messages):
            console.print(f"[dim]session {session.id} was last used with[/dim] "
                          f"[key]{escape(target_model)}[/key]")
            answer = Prompt.ask(
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
    console.print(f"[ok]loaded[/ok] {session.id} · {escape(session.display_title())} "
                  f"· model {escape(session.model)}")
    repl.refresh_models()


def cmd_load(repl: Repl, argument: str) -> None:
    if not argument:
        session = pick_session(repl.store)
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
    if not repl.client.model_exists(name):
        console.print(f"[err]no session or model named {escape(argument)}[/err]")
        return
    if name == repl.session.model:
        console.print("[dim]already using that model[/dim]")
        return
    if repl.session.messages:
        keep = Confirm.ask(f"switch to [key]{escape(name)}[/key] and keep the "
                           f"{len(repl.session.messages)}-message context?", default=False)
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
    if repl.client.model_exists(name) and not Confirm.ask(
            f"[warn]{escape(name)} already exists — overwrite?[/warn]", default=False):
        return
    console.print(f"[dim]creating {escape(name)} from {escape(repl.session.model)}…[/dim]")

    def progress(status: str) -> None:
        console.print(f"[dim]  {escape(status)}[/dim]")

    try:
        repl.client.create_model(name, repl.session.model,
                                 system=repl.session.system,
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
                                  parameters=parameters)
    console.print(f"[ok]new session[/ok] {repl.session.id}")


def cmd_sessions(repl: Repl, argument: str) -> None:
    session = pick_session(repl.store, filter_text=argument)
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


def cmd_model(repl: Repl, argument: str) -> None:
    if argument == "--list":
        try:
            models = repl.client.tags()
        except OllamaError as exc:
            console.print(f"[err]{escape(str(exc))}[/err]")
            return
        table = Table(title="models", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("name")
        table.add_column("size", justify="right")
        table.add_column("modified")
        for model in sorted(models, key=lambda item: item.get("modified_at") or "",
                            reverse=True):
            table.add_row(escape(model.get("name", "?")), human_bytes(model.get("size")),
                          short_stamp(model.get("modified_at")))
        console.print(table)
        return
    name = argument.strip() or pick_model(repl.client, repl.session.model)
    if not name:
        return
    if not repl.client.model_exists(name):
        console.print(f"[err]model {escape(name)} not found[/err]")
        return
    if name == repl.session.model:
        console.print("[dim]already using that model[/dim]")
        return
    if repl.session.messages and not Confirm.ask(
            f"switch to [key]{escape(name)}[/key] and keep context?", default=False):
        repl.session.messages = []
    repl.unload_model()
    repl.session.model = name
    repl.session.touch()
    repl.refresh_models()
    console.print(f"[ok]model[/ok] {escape(name)}")


def cmd_think(repl: Repl, argument: str) -> None:
    if not argument:
        console.print(f"think: [key]{repl.session.think if repl.session.think is not None else 'default'}[/key]")
        return
    try:
        repl.session.think = parse_think(argument)
    except ValueError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    repl.session.touch()
    console.print(f"[ok]think[/ok] {repl.session.think}")


def cmd_thinking(repl: Repl, argument: str) -> None:
    """Display preference for the model's thinking (not the model's think flag)."""
    topic = argument.split()[0].lower() if argument.split() else ""
    if topic in {"full", "show-full", "last"}:
        message = _last_assistant(repl)
        thinking = (message or {}).get("thinking")
        if not thinking:
            console.print("[dim]the last response has no stored thinking[/dim]")
            return
        panel = thinking_panel(thinking, f"thinking · {human_chars(len(thinking))}")
        if console.is_terminal:
            with console.pager(styles=True):
                console.print(panel)
        else:
            console.print(panel)
        return
    if topic in {"", "status"}:
        console.print(f"thinking display: [key]{repl.thinking_mode}[/key] "
                      "(collapsed|expanded|hidden)")
        return
    if topic in {"expand", "expanded", "show", "on"}:
        repl.thinking_mode = "expanded"
    elif topic in {"collapse", "collapsed"}:
        repl.thinking_mode = "collapsed"
    elif topic in {"hide", "hidden", "off"}:
        repl.thinking_mode = "hidden"
    elif topic == "toggle":
        repl.thinking_mode = "collapsed" if repl.thinking_mode == "expanded" else "expanded"
    else:
        console.print("[err]usage: /thinking [show|collapse|hide|toggle|full][/err]")
        return
    console.print(f"[ok]thinking[/ok] {repl.thinking_mode}")


def cmd_system(repl: Repl, argument: str) -> None:
    if not argument:
        if repl.session.system:
            console.print(Panel(escape(repl.session.system), title="system",
                                border_style="dim", title_align="left"))
        else:
            console.print("[dim]no system prompt[/dim]")
        return
    if argument == "clear":
        repl.session.system = None
        repl.session.touch()
        console.print("[ok]system prompt cleared[/ok]")
        return
    repl.session.system = argument
    repl.session.touch()
    console.print("[ok]system prompt set[/ok]")


def cmd_undo(repl: Repl, argument: str) -> None:
    if not repl.session.messages:
        console.print("[dim]nothing to undo[/dim]")
        return
    while repl.session.messages and repl.session.messages[-1].get("role") == "assistant":
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
    edited = open_in_editor(original)
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
    branch.tags = list(repl.session.tags)
    branch.stats = dict(repl.session.stats)
    if not repl.no_save:
        repl.store.save(repl.session)
        repl.store.save(branch)
    repl.session = branch
    console.print(f"[ok]branched[/ok] {branch.id}")


def cmd_context(repl: Repl, argument: str) -> None:
    messages = repl.session.messages
    roles: dict[str, int] = {}
    for message in messages:
        roles[message.get("role", "?")] = roles.get(message.get("role", "?"), 0) + 1
    used = repl.session.stats.get("last_prompt_tokens")
    total = None
    try:
        total = model_context_length(repl.client.show(repl.session.model),
                                     repl.session.parameters)
    except OllamaError:
        pass
    table = Table(title="context", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("messages", str(len(messages)))
    for role, count in sorted(roles.items()):
        table.add_row(f"  {role}", str(count))
    table.add_row("last prompt tokens", str(used if used is not None else "?"))
    table.add_row("context length", str(total if total else "unknown"))
    if used and total:
        pct = used * 100 / total
        color = "err" if pct >= 90 else "warn" if pct >= 70 else "ok"
        table.add_row("usage", f"[{color}]{pct:.0f}%[/{color}]")
    console.print(table)


def cmd_summarize(repl: Repl, argument: str) -> None:
    messages = repl.session.messages
    if len(messages) <= 3:
        console.print("[dim]not enough messages to summarize[/dim]")
        return
    keep_tail = messages[-2:] if len(messages) % 2 == 0 else messages[-1:]
    older = messages[:len(messages) - len(keep_tail)]
    if not older:
        console.print("[dim]nothing to summarize[/dim]")
        return
    if not argument and not Confirm.ask(f"summarize {len(older)} older messages?",
                                        default=True):
        return
    transcript_lines = []
    for message in older:
        role = message.get("role", "?")
        transcript_lines.append(f"{role}: {message.get('content', '')}")
    prompt = ("Summarize the following conversation concisely for continued use as "
              "context. Keep decisions, facts, names, numbers, code identifiers and open "
              "questions. Output only the summary.\n\n" + "\n\n".join(transcript_lines))
    console.print("[dim]summarizing…[/dim]")
    try:
        response = repl.client.chat_once(build_chat_payload(
            repl.session.model, [{"role": "user", "content": prompt}],
            stream=False, think=False))
        summary = (response.get("message") or {}).get("content", "").strip()
    except OllamaError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    if not summary:
        console.print("[err]empty summary; keeping messages[/err]")
        return
    repl.session.messages = [
        {"role": "user", "ts": now_iso(),
         "content": "[Summary of earlier conversation]\n\n" + summary},
        {"role": "assistant", "ts": now_iso(),
         "content": "Understood — continuing from that summary."},
        *keep_tail,
    ]
    repl.session.touch()
    repl.session.stats["last_prompt_tokens"] = None
    if not repl.no_save:
        repl.store.save(repl.session)
    console.print(f"[ok]compacted[/ok] {len(older)} messages → summary")


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
    try:
        tokens = shlex.split(argument)
    except ValueError as exc:
        console.print(f"[err]{exc}[/err]")
        return
    for token in tokens:
        path = Path(os.path.expanduser(token))
        try:
            repl.pending_images.append(encode_image(path, repl.image_storage))
            console.print(f"[ok]attached[/ok] {escape(path.name)}")
        except (OSError, ValueError) as exc:
            console.print(f"[err]{escape(str(exc))}[/err]")


def cmd_render(repl: Repl, argument: str) -> None:
    if not argument:
        console.print(f"render: [key]{repl.render_mode}[/key]")
        return
    if argument not in {"markdown", "plain", "raw", "auto"}:
        console.print("[err]usage: /render [markdown|plain|raw|auto][/err]")
        return
    repl.render_mode = argument
    console.print(f"[ok]render[/ok] {argument}")


def cmd_math(repl: Repl, argument: str) -> None:
    if not argument:
        console.print(f"math: [key]{'on' if repl.math else 'off'}[/key]")
        return
    repl.math = parse_bool(argument, repl.math)
    console.print(f"[ok]math {'on' if repl.math else 'off'}[/ok]")


def cmd_pager(repl: Repl, argument: str) -> None:
    if not argument:
        console.print(f"pager: [key]{'on' if repl.pager else 'off'}[/key]")
        return
    repl.pager = parse_bool(argument, repl.pager)
    console.print(f"[ok]pager {'on' if repl.pager else 'off'}[/ok]")


def cmd_tps(repl: Repl, argument: str) -> None:
    if not argument:
        console.print(f"tps footer: [key]{'on' if repl.tps_footer else 'off'}[/key]")
        return
    repl.tps_footer = parse_bool(argument, repl.tps_footer)
    console.print(f"[ok]tps footer[/ok] {'on' if repl.tps_footer else 'off'}")


def _last_assistant(repl: Repl) -> Optional[dict[str, Any]]:
    for message in reversed(repl.session.messages):
        if message.get("role") == "assistant" and message.get("content"):
            return message
    return None


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
    if not HAS_PYPERCLIP:
        console.print("[err]pyperclip is not installed[/err]")
        return
    try:
        pyperclip.copy(content)
    except Exception as exc:  # pyperclip raises platform-specific errors
        console.print(f"[err]clipboard failed: {escape(str(exc))}[/err]")
        return
    console.print(f"[ok]copied[/ok] {len(content)} characters")


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


def cmd_stats(repl: Repl, argument: str) -> None:
    stats = repl.session.stats
    table = Table(title="stats", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("session", repl.session.id)
    table.add_row("model", repl.session.model)
    table.add_row("turns", str(stats.get("turns", 0)))
    table.add_row("messages", str(len(repl.session.messages)))
    table.add_row("prompt tokens", str(stats.get("prompt_tokens", 0)))
    table.add_row("completion tokens", str(stats.get("completion_tokens", 0)))
    table.add_row("generation time", human_duration(stats.get("duration_s")))
    if stats.get("last_eval_tps"):
        table.add_row("last speed", f"{stats['last_eval_tps']:.1f} tok/s")
    if stats.get("last_prompt_tokens"):
        table.add_row("last prompt", f"{stats['last_prompt_tokens']} tok")
    console.print(table)


def cmd_bye(repl: Repl, argument: str) -> bool:
    return False


COMMAND_HANDLERS: dict[str, Callable[[Repl, str], Any]] = {
    "/help": cmd_help, "/?": cmd_help,
    "/set": cmd_set,
    "/show": cmd_show,
    "/load": cmd_load,
    "/save": cmd_save,
    "/clear": cmd_clear,
    "/new": cmd_new,
    "/sessions": cmd_sessions,
    "/title": cmd_title,
    "/tag": cmd_tag,
    "/model": cmd_model,
    "/ps": cmd_ps,
    "/num-ctx": cmd_num_ctx,
    "/params": cmd_params,
    "/think": cmd_think,
    "/thinking": cmd_thinking,
    "/system": cmd_system,
    "/undo": cmd_undo,
    "/retry": cmd_retry,
    "/edit": cmd_edit,
    "/branch": cmd_branch,
    "/context": cmd_context,
    "/summarize": cmd_summarize,
    "/image": cmd_image,
    "/render": cmd_render,
    "/math": cmd_math,
    "/pager": cmd_pager,
    "/tps": cmd_tps,
    "/copy": cmd_copy,
    "/export": cmd_export,
    "/stats": cmd_stats,
    "/bye": cmd_bye, "/exit": cmd_bye,
}


