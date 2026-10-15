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
import subprocess
import threading
import time
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
    Application,
    Completer,
    Completion,
    FileHistory,
    FormattedTextControl,
    KeyBindings,
    Layout,
    PromptSession,
    PromptStyle,
    Window,
    console,
    patch_stdout,
    pyperclip,
    tool_color,
)
from .files import expand_file_mentions
from .helpers import (
    coerce_value,
    human_bytes,
    human_chars,
    human_duration,
    is_local_host,
    naive_title,
    new_session_id,
    now_iso,
    parse_bool,
    parse_think,
    remote_node,
    short_stamp,
    shortcut,
)
from .images import encode_image
from .ctxfit import (
    FitRequest,
    NUM_CTX_HELP,
    machine_state_for,
    max_gpu_context,
    parse_num_ctx_arg,
)
from .interactive import (
    GPU_LAYERS_USAGE,
    _CODE_BLOCK_RE,
    disk_free_line,
    ensure_model,
    gpu_layers_label,
    model_block_count,
    model_context_length,
    open_in_editor,
    parse_gpu_layers_spec,
    pick_model,
    pick_session,
    pull_with_progress,
    resolve_gpu_layers,
)
from .renderer import (GenerationResult, Renderer, thinking_panel,
                       tool_result_panel, user_message_box)
from .session import Session, SessionStore, export_html, export_json, export_markdown
from .tools import (
    DEFAULT_SHELL_ALLOW,
    ToolContext,
    ToolRule,
    approval_decision,
    call_detail,
    call_fingerprint,
    default_registry,
    execute_tool,
    find_tts_remote,
    matching_rule,
    normalize_arguments,
    resolve_tool_alias,
    rule_label,
    suggest_rules,
    tool_schemas as build_tool_schemas,
)


COMMAND_INFO: dict[str, tuple[str, str]] = {
    "/help, /?": ("[topic]", "show commands (topic: shortcuts)"),
    "/set": ("parameter NAME VALUE | system TEXT", "session parameters / system prompt"),
    "/show": ("[info|license|modelfile|parameters|system|template]", "model details"),
    "/load": ("[MODEL|SESSION]", "load a saved session or switch model"),
    "/save": ("NAME", "create an Ollama model from this session"),
    "/pull": ("[MODEL]", "pull a model onto this host (updates if present)"),
    "/clear": ("", "forget the context, keep settings, start a new session"),
    "/new": ("", "start a brand new session"),
    "/sessions": ("[filter]", "browse and switch saved sessions"),
    "/title": ("[TEXT|--auto]", "show or set the session title"),
    "/tag": ("[list|add TAG…|rm TAG]", "session tags"),
    "/model": ("[NAME|--list]", "switch model"),
    "/ps": ("", "show resident models (ollama ps)"),
    "/num-ctx": ("[N | --max-gpu [--probe]]",
                 "context window: N, or the largest that keeps the GPU placement"),
    "/gpu-layers": ("[N|N%|+N|-N|all|cpu|auto]", "GPU layers for this model"),
    "/params": ("", "show session parameters"),
    "/tools": ("[on|off|policy MODE|enable|disable NAME…|allow|rules]", "tool harness, approval rules and per-tool control"),
    "/tts": ("[on|off|voice NAME|lang CODE|tools on|off|voices]", "auto-speak via tts-remote"),
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


class ApprovalPicker:
    """Checkbox state machine behind the tool-approval prompt.

    Rows are the suggested approval rules (plus the exact-call grant). Space
    toggles the row under the cursor, enter runs the call storing what is
    ticked, `a` switches to auto-allowing every later call and `d`/`q`/escape
    deny. Pure key handling, so tests can drive it without a terminal.
    """

    def __init__(self, labels: list[str]):
        self.labels = labels
        self.cursor = 0
        self.ticked: set[int] = set()
        self.action: Optional[str] = None

    def handle(self, key: str) -> bool:
        """Process one key press; True once the picker is resolved."""
        if key in {"up", "k", "c-p"}:
            self.cursor = (self.cursor - 1) % max(1, len(self.labels))
        elif key in {"down", "j", "c-n"}:
            self.cursor = (self.cursor + 1) % max(1, len(self.labels))
        elif key == "space":
            self.ticked ^= {self.cursor}
        elif key in {"enter", "c-j"}:
            self.action = "run"
            return True
        elif key == "a":
            self.action = "auto"
            return True
        elif key in {"d", "q", "escape"}:
            self.action = "deny"
            return True
        return False

    def render(self) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        for index, label in enumerate(self.labels):
            pointer = "❯ " if index == self.cursor else "  "
            marker = "[x]" if index in self.ticked else "[ ]"
            style = "class:pick-cursor" if index == self.cursor else ""
            lines.append((style, f"{pointer}{marker} {label}\n"))
        lines.append(("class:pick-hint",
                      " space toggle · enter run · a auto-allow all · d deny"))
        return lines


def _run_approval_picker_text(picker: ApprovalPicker) -> None:
    """Plain prompt fallback when prompt_toolkit is unavailable."""
    choices = " · ".join(f"[{index + 1}] {label}"
                         for index, label in enumerate(picker.labels))
    answer = Prompt.ask(
        f"[y] run once only · {choices} · [a] auto-allow all · [n] deny",
        default="y").strip().lower()
    if answer in {"a", "auto"}:
        picker.action = "auto"
    elif answer.isdigit() and 1 <= int(answer) <= len(picker.labels):
        picker.action = "run"
        picker.ticked = {int(answer) - 1}
    elif answer in {"y", "yes", ""}:
        picker.action = "run"
    else:
        picker.action = "deny"


def _run_approval_picker(picker: ApprovalPicker, *, input_source: Any = None,
                         output: Any = None) -> None:
    """Run the checkbox picker on the terminal, filling in `picker.action`.

    `input_source`/`output` are the prompt_toolkit test seam: pass a pipe
    input and a DummyOutput to drive the picker headlessly.
    """
    if not HAS_PROMPT_TOOLKIT or Application is None:
        _run_approval_picker_text(picker)
        return
    control = FormattedTextControl(text=picker.render, focusable=True)
    key_bindings = KeyBindings()

    def bind(key: str) -> None:
        @key_bindings.add(key)
        def _handler(event: Any) -> None:
            if picker.handle(key):
                event.app.exit()

    for key in ("up", "k", "c-p", "down", "j", "c-n", "space", "enter",
                "c-j", "a", "d", "q", "escape"):
        bind(key)
    app = Application(layout=Layout(Window(control)), key_bindings=key_bindings,
                      style=PromptStyle.from_dict({"pick-cursor": "reverse",
                                                   "pick-hint": "fg:ansibrightblack"}),
                      full_screen=False, input=input_source, output=output)
    app.run()


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
        self.render_mode = args.render or ("raw" if getattr(args, "raw", False)
                                           else config.get("render", "markdown"))
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
        if getattr(args, "no_tools", False):
            self.tools_enabled = False
        elif getattr(args, "tools", False):
            self.tools_enabled = True
        else:
            self.tools_enabled = bool(config.get("tools", True))
        self.auto_approve = bool(getattr(args, "yes", False))
        self.tools_policy = str(config.get("tools_policy") or "allowlist").lower()
        if self.tools_policy not in {"auto", "allowlist", "ask"}:
            self.tools_policy = "allowlist"
        try:
            self.tools_max_rounds = max(1, int(config.get("tools_max_rounds") or 8))
        except (TypeError, ValueError):
            self.tools_max_rounds = 8
        self.shell_allow = set(DEFAULT_SHELL_ALLOW) | {
            str(item) for item in (config.get("tools_shell_allow") or []) if item}
        try:
            tool_timeout = float(config.get("tools_timeout") or 120)
        except (TypeError, ValueError):
            tool_timeout = 120.0
        try:
            tool_max_output = int(config.get("tools_max_output") or 4000)
        except (TypeError, ValueError):
            tool_max_output = 4000
        self.tool_ctx = ToolContext(cwd=os.getcwd(), timeout=tool_timeout,
                                    max_output=tool_max_output,
                                    tts_remote=config.get("tts_remote") or None)
        tts_option: Optional[bool]
        if not config.get("tts_tools", True):
            tts_option = False
        elif config.get("tts_remote") and config["tts_remote"] != "tts-remote":
            tts_option = True  # explicit path: register, the handler reports if missing
        else:
            tts_option = None  # auto-detect a tts-remote binary
        self.tool_registry = default_registry(tts=tts_option)
        self._tool_support_checked = False
        self._format_tools_warned = False
        if getattr(args, "no_speak", False):
            self.tts_auto = False
        elif getattr(args, "speak", False):
            self.tts_auto = True
        elif session.tts_auto is not None:
            self.tts_auto = bool(session.tts_auto)
        else:
            self.tts_auto = bool(config.get("tts_auto", False))
        self.tts_voice = session.tts_voice or config.get("tts_voice") or None
        self.tts_lang = session.tts_lang or config.get("tts_lang") or None
        self._pt_session: Optional[Any] = None
        self._ps_shown_for: set[str] = set()
        self._preload_thread: Optional[threading.Thread] = None
        self._block_counts: dict[str, Optional[int]] = {}

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
                                 key_bindings=bindings, complete_while_typing=True,
                                 rprompt=self._loading_rprompt,
                                 refresh_interval=0.1)
        except Exception:
            return None

    _SPINNER_FRAMES = "⠋⠙⠹⠸⠴⠦⠧⠇"

    def _loading_rprompt(self) -> Optional[str]:
        """Right-side prompt text (same line as the input, right-aligned):
        an animated "waiting for model" spinner for as long as the
        background preload is still running, driven by the prompt
        session's own refresh_interval -- this is what makes it show up
        without blocking the prompt, unlike a separate rich.Live view
        fighting prompt_toolkit for the terminal. `rprompt` (not
        `bottom_toolbar`, which anchors to the terminal's last row and
        leaves a large blank gap) keeps this on the prompt's own line."""
        thread = self._preload_thread
        if thread is None or not thread.is_alive():
            return None
        frame = self._SPINNER_FRAMES[int(time.monotonic() * 10) % len(self._SPINNER_FRAMES)]
        return f"{frame} waiting for {self.session.model}… "

    def _read_line(self, message: str) -> str:
        if self._pt_session is not None:
            if patch_stdout is not None:
                with patch_stdout(raw=True):
                    return self._pt_session.prompt(message)
            return self._pt_session.prompt(message)
        return input(message)

    # ── rendering ───────────────────────────────────────────────────────────

    def _renderer(self, *, after_tools: bool = False) -> Renderer:
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
                        model=self.session.model, after_tools=after_tools)

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
        already warm by the time the first message is sent, without
        blocking the prompt from appearing -- see _loading_rprompt() for
        the "waiting for model" indicator shown while this runs, and
        _read_line()'s use of patch_stdout() for how `ollama ps` (printed
        below, once loading finishes) safely interleaves with an active
        prompt instead of corrupting it."""
        model = self.session.model
        if not model:
            return
        keepalive = self.session.keepalive or self.args.keepalive

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

    def _model_block_count(self) -> Optional[int]:
        """Cached `<arch>.block_count` for the session model (None if unknown)."""
        model = self.session.model
        if not model:
            return None
        if model not in self._block_counts:
            try:
                show = self.client.show(model)
            except OllamaError:
                show = {}
            self._block_counts[model] = model_block_count(show)
        return self._block_counts[model]

    def gpu_layers_value(self) -> Optional[int]:
        """Resolved num_gpu for the session's gpu-layers spec (None = automatic)."""
        if not self.session.gpu_layers:
            return None
        try:
            return resolve_gpu_layers(self.session.gpu_layers, self._model_block_count())
        except (ValueError, RuntimeError):
            return None

    def gpu_layers_summary(self) -> str:
        return gpu_layers_label(self.session.gpu_layers, self._model_block_count())

    def _payload(self, messages: list[dict[str, Any]], *, stream: bool = True,
                 with_tools: Optional[bool] = None) -> dict[str, Any]:
        if with_tools is None:
            with_tools = self.tools_enabled
        tools = None
        if with_tools and self.session.format:
            if not self._format_tools_warned:
                self._format_tools_warned = True
                console.print("[warn]--format and tools cannot be combined; tools "
                              "disabled for this session[/warn]")
            with_tools = False
        if with_tools:
            tools = build_tool_schemas(self.tool_registry,
                                       set(self.session.tools_disabled)) or None
        options = dict(self.session.parameters)
        gpu_layers = self.gpu_layers_value()
        if gpu_layers is not None:
            options["num_gpu"] = gpu_layers
        return build_chat_payload(self.session.model, messages, stream=stream,
                                  fmt=self.session.format, think=self.effective_think(),
                                  keepalive=self.session.keepalive or self.args.keepalive,
                                  options=options, tools=tools)

    # ── tool harness ────────────────────────────────────────────────────────

    def active_tool_schemas(self) -> list[dict[str, Any]]:
        if not self.tools_enabled:
            return []
        return build_tool_schemas(self.tool_registry, set(self.session.tools_disabled))

    def _check_tool_support(self) -> None:
        if not self.tools_enabled or self._tool_support_checked:
            return
        self._tool_support_checked = True
        try:
            capabilities = self.client.capabilities(self.session.model)
        except Exception:  # capability probing must never break a turn
            return
        if capabilities and "tools" not in capabilities:
            console.print(f"[warn]{escape(self.session.model)} does not advertise tool "
                          f"support — tool calls may not work[/warn]")

    def _session_rules(self) -> list[ToolRule]:
        rules: list[ToolRule] = []
        for entry in self.session.tool_rules or []:
            rule = ToolRule.from_dict(entry)
            if rule is not None:
                rules.append(rule)
        return rules

    def _approve_tool_call(self, spec: Any, args: dict[str, Any], fingerprint: str,
                           *, allow_prompt: bool = True) -> tuple[bool, str]:
        """Decide whether a tool call runs; a denial is a normal result."""
        if self.auto_approve:
            return True, "auto (--yes)"
        decision = approval_decision(spec, args, policy=self.tools_policy,
                                     shell_allow=self.shell_allow)
        if decision.allowed:
            return True, "safe list"
        if fingerprint in (self.session.tool_allow or []):
            return True, "previously allowed"
        rule = matching_rule(self._session_rules(), spec.name, args,
                             cwd=self.tool_ctx.cwd, shell_allow=self.shell_allow)
        if rule is not None:
            label = rule_label(rule)
            console.print(f"[dim]allowed by rule[/dim] [key]{escape(label)}[/key]")
            return True, f"rule: {label}"
        if not (allow_prompt and console.is_terminal):
            return False, f"not auto-approved ({decision.reason}) and this run is unattended"
        console.print(f"[warn]{escape(spec.name)} needs confirmation: "
                      f"{escape(decision.reason)}[/warn]")
        rows: list[tuple[str, Optional[ToolRule], str]] = [
            ("call", None, "allow exactly this call")]
        for suggestion in suggest_rules(spec.name, args, cwd=self.tool_ctx.cwd):
            rows.append(("rule", suggestion, rule_label(suggestion)))
        picker = ApprovalPicker([label for _kind, _rule, label in rows])
        _run_approval_picker(picker)
        if picker.action == "auto":
            self.auto_approve = True
            self.session.touch()
            return True, "auto-allow everything (careful)"
        if picker.action != "run":
            return False, "denied by the user"
        remembered_call = False
        stored_rules: list[dict[str, Any]] = []
        stored: list[str] = []
        for index in sorted(picker.ticked):
            kind, rule, _label = rows[index]
            if kind == "call":
                remembered_call = True
            elif rule is not None:
                stored_rules.append(rule.to_dict())
                stored.append(rule_label(rule))
        if remembered_call:
            self.session.tool_allow = list(self.session.tool_allow or []) + [fingerprint]
            stored.insert(0, "exactly this call")
        if stored_rules:
            self.session.tool_rules = list(self.session.tool_rules or []) + stored_rules
        if stored:
            self.session.touch()
            console.print("[dim]stored rule(s):[/dim] "
                          + escape(" · ".join(stored)))
            return True, "allowed (remembered)"
        return True, "confirmed"

    def _execute_tool_calls(self, calls: list[dict[str, Any]], *, quiet: bool = False,
                            allow_prompt: bool = True) -> bool:
        """Run the model's tool calls; returns True when interrupted."""
        disabled = set(self.session.tools_disabled)
        available = sorted(set(self.tool_registry) - disabled)
        interrupted = False
        for call in calls:
            function = (call.get("function") or {}) if isinstance(call, dict) else {}
            name = str(function.get("name") or "?")
            if interrupted:
                self.session.add_message("tool", "interrupted by user", tool_name=name)
                continue
            args = normalize_arguments(function.get("arguments"))
            if args is None:
                text = ("invalid tool arguments (not a JSON object): "
                        + str(function.get("arguments"))[:200])
                self.session.add_message("tool", text, tool_name=name)
                if not quiet:
                    console.print(tool_result_panel(name, text=text, ok=False))
                continue
            spec = self.tool_registry.get(name) if name not in disabled else None
            alias_note = ""
            if spec is None:
                alias = resolve_tool_alias(name)
                if alias != name and alias not in disabled:
                    spec = self.tool_registry.get(alias)
                    if spec is not None:
                        alias_note = f"(note: {name!r} is called {alias!r} here)\n"
                        name = alias
            if spec is None:
                text = f"unknown or disabled tool: {name}"
                if available:
                    text += "; available: " + ", ".join(available)
                self.session.add_message("tool", text, tool_name=name)
                if not quiet:
                    console.print(tool_result_panel(name, text=text, ok=False))
                continue
            fingerprint = call_fingerprint(name, args)
            detail = call_detail(name, args)
            try:
                run, note = self._approve_tool_call(spec, args, fingerprint,
                                                    allow_prompt=allow_prompt)
            except KeyboardInterrupt:
                run, note = False, "interrupted by user"
                interrupted = True
            if not run:
                text = alias_note + f"tool not run: {note}"
                self.session.add_message("tool", text, tool_name=name)
                if not quiet:
                    console.print(tool_result_panel(name, detail=detail, text=text,
                                                    ok=False))
                continue
            if not quiet:
                color = tool_color(name)
                console.print(f"[dim]running[/dim] [{color}]{escape(name)}[/{color}]"
                              f"[dim]…[/dim]")
            try:
                outcome = execute_tool(spec, args, self.tool_ctx)
            except KeyboardInterrupt:
                text = "interrupted by user"
                self.session.add_message("tool", text, tool_name=name)
                if not quiet:
                    console.print(tool_result_panel(name, detail=detail, text=text,
                                                    ok=False))
                interrupted = True
                continue
            self.session.add_message("tool", alias_note + outcome.text,
                                     tool_name=name)
            if not quiet:
                console.print(tool_result_panel(name, detail=detail,
                                                text=alias_note + outcome.text,
                                                ok=outcome.ok, duration=outcome.duration))
        if not self.no_save:
            self.store.save(self.session)
        return interrupted

    # ── tts ─────────────────────────────────────────────────────────────────

    def speak_text(self, text: str) -> bool:
        """Queue `text` for tts-remote playback; never blocks or raises."""
        if not text or not text.strip():
            return False
        binary = find_tts_remote(self.tool_ctx)
        if not binary:
            return False
        command = [binary, "say", "-q", "--markdown"]
        if self.tts_voice:
            command += ["--voice", str(self.tts_voice)]
        if self.tts_lang:
            command += ["--lang", str(self.tts_lang)]
        try:
            proc = subprocess.run(command, input=text, capture_output=True,
                                  text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            console.print(f"[warn]tts: {escape(str(exc))}[/warn]")
            return False
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            console.print(f"[warn]tts: {escape(detail or 'tts-remote failed')}[/warn]")
            return False
        if console.is_terminal:
            console.print(f"[dim]speaking · {len(text)} chars[/dim]")
        return True

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
            if console.is_terminal:
                self._maybe_show_ps()
        self._check_tool_support()
        result = GenerationResult()
        tool_rounds = 0
        while True:
            renderer = self._renderer(after_tools=bool(tool_rounds))
            marker = f"[ai]{escape(self.session.model)} ›[/ai]"
            if renderer.mode != "raw" and not renderer.live_render:
                console.print(marker)
            with_tools = self.tools_enabled and tool_rounds < self.tools_max_rounds
            try:
                result = renderer.stream(self.client.chat(
                    self._payload(self.session.message_payloads(with_tools=with_tools),
                                  with_tools=with_tools),
                    timeout=self.timeout))
            except OllamaError as exc:
                result = GenerationResult(error=str(exc))
                console.print(f"[err]{escape(str(exc))}[/err]")
            if result.content or result.tool_calls:
                self.session.add_message("assistant", result.content,
                                         thinking=result.thinking or None,
                                         cancelled=True if result.cancelled else None,
                                         tool_calls=result.tool_calls or None)
            elif result.cancelled:
                self.session.add_message("assistant", "", cancelled=True)
            if result.stats:
                self.session.record_turn(result.stats, count_turn=(tool_rounds == 0))
                self.session.stats["last_prompt_tokens"] = result.stats.get("prompt_tokens")
                self.session.stats["last_completion_tokens"] = result.stats.get("completion_tokens")
            if not self.no_save:
                self.store.save(self.session)
            if result.error or result.cancelled or not result.tool_calls \
                    or not self.tools_enabled:
                break
            tool_rounds += 1
            if tool_rounds > self.tools_max_rounds:
                console.print(f"[warn]tool-call limit reached "
                              f"({self.tools_max_rounds}) — not running more tools[/warn]")
                for call in result.tool_calls:
                    function = (call.get("function") or {}) if isinstance(call, dict) else {}
                    self.session.add_message(
                        "tool", "not executed: tool-call limit reached",
                        tool_name=str(function.get("name") or "?"))
                if not self.no_save:
                    self.store.save(self.session)
                break
            if self._execute_tool_calls(result.tool_calls,
                                        quiet=(self.render_mode == "raw")):
                console.print("[warn]tool execution interrupted; stopping this turn[/warn]")
                break
        if self.tts_auto and result.content and not result.cancelled and not result.error:
            self.speak_text(result.content)
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
            f"[key]host[/key]    {escape(self.client.base)}"
            + (f" · {escape(self.session.remote)}" if self.session.remote else ""),
            f"[key]render[/key]  {self.render_mode}"
            + ("" if self.math else " · math off")
            + f" · thinking {self.thinking_mode}"
            + (" · tts auto" if self.tts_auto else "")
            + f" · {model_count} local models",
        ]
        if self.session.messages:
            lines.append(f"[key]context[/key] {len(self.session.messages)} messages")
        lines.append("[dim]/help for commands · Ctrl+D to exit · Ctrl+T thinking[/dim]")
        console.print(Panel("\n".join(lines), title="ollamarun", border_style="cyan",
                            title_align="left"))

    def context_used(self) -> Optional[int]:
        """Tokens that the next request will carry: the last prompt plus the
        output just generated (both are re-sent; `prompt_eval_count` alone
        omits the assistant reply that was appended after it was counted)."""
        prompt = self.session.stats.get("last_prompt_tokens")
        completion = self.session.stats.get("last_completion_tokens")
        if prompt is None and completion is None:
            return None
        return (prompt or 0) + (completion or 0)

    def _context_warning(self) -> None:
        if not self.session.messages:
            return
        try:
            show = self.client.show(self.session.model)
        except OllamaError:
            return
        total = model_context_length(show, self.session.parameters)
        used = self.context_used()
        if total and used and used / total >= 0.8:
            console.print(f"[warn]context {used}/{total} tokens "
                          f"({used * 100 // total}%) — /summarize to compact[/warn]")

    def _maybe_auto_compact(self) -> None:
        try:
            show = self.client.show(self.session.model)
        except OllamaError:
            return
        total = model_context_length(show, self.session.parameters)
        used = self.context_used()
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
                            ("Ctrl+T", "expand or collapse the thinking panel "
                                       "while a response streams"),
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
    table.add_row("gpu layers", escape(repl.gpu_layers_summary()))
    table.add_row("format", escape(str(repl.session.format or "")))
    table.add_row("system", shortcut(repl.session.system or "", 60))
    console.print(table)


def _rule_rows(repl: Repl) -> list[tuple[str, str, str, str]]:
    """(kind, pattern, base, created) for every stored approval, call-first."""
    rows: list[tuple[str, str, str, str]] = []
    for fingerprint in repl.session.tool_allow or []:
        rows.append(("call", fingerprint, "", ""))
    for entry in repl.session.tool_rules or []:
        rule = ToolRule.from_dict(entry)
        if rule is not None:
            rows.append((rule.kind, rule.pattern, rule.base, rule.created))
    return rows


def _cmd_tools_rules_rm(repl: Repl, parts: list[str]) -> None:
    indices = sorted({int(part) - 1 for part in parts if part.isdigit()})
    if not indices:
        console.print("[err]usage: /tools rules rm N…[/err]")
        return
    allow = list(repl.session.tool_allow or [])
    stored = [entry for entry in (repl.session.tool_rules or [])
              if ToolRule.from_dict(entry) is not None]
    merged: list[tuple[str, int]] = [("call", index) for index in range(len(allow))]
    merged += [("rule", index) for index in range(len(stored))]
    drop = {index for index in indices if 0 <= index < len(merged)}
    if not drop:
        console.print("[warn]no rule with that number[/warn]")
        return
    repl.session.tool_allow = [allow[index] for position, (kind, index)
                               in enumerate(merged)
                               if position not in drop and kind == "call"]
    repl.session.tool_rules = [stored[index] for position, (kind, index)
                               in enumerate(merged)
                               if position not in drop and kind == "rule"]
    repl.session.touch()
    console.print(f"[ok]removed[/ok] {len(drop)} rule(s)")


def _cmd_tools_rules_add(repl: Repl, parts: list[str]) -> None:
    kind = parts[0].lower() if parts else ""
    spec = parts[1:]
    if kind == "shell" and spec:
        pattern = " ".join(spec)
        try:
            tokens = shlex.split(pattern)
        except ValueError as exc:
            console.print(f"[err]unparsable pattern: {exc}[/err]")
            return
        if not tokens or any(char in pattern for char in "$`><(){}&|;"):
            console.print("[err]pattern contains shell metacharacters[/err]")
            return
        if any(token == ":*" for token in tokens[:-1]):
            console.print("[err]':*' is only valid at the end[/err]")
            return
        rule = ToolRule(kind="shell", pattern=pattern, created=now_iso())
    elif kind == "path" and len(spec) >= 2:
        base = Path(spec[0]).expanduser()
        if not base.is_dir():
            console.print(f"[err]not a directory: {escape(str(base))}[/err]")
            return
        pattern = " ".join(spec[1:])
        if pattern.startswith("/") or ".." in Path(pattern).parts:
            console.print("[err]path patterns are relative to BASE and cannot "
                          "escape it[/err]")
            return
        rule = ToolRule(kind="path", pattern=pattern,
                        base=os.path.realpath(base), created=now_iso())
    elif kind == "python" and len(spec) == 1:
        roots = sorted({root.strip() for root in spec[0].split(",") if root.strip()})
        if not roots or any(not root.isidentifier() for root in roots):
            console.print("[err]usage: /tools rules add python mod1,mod2[/err]")
            return
        rule = ToolRule(kind="python", pattern=",".join(roots), created=now_iso())
    elif kind == "call" and spec:
        rule = ToolRule(kind="call", pattern=" ".join(spec), created=now_iso())
    else:
        console.print("[err]usage: /tools rules add shell 'PREFIX :*' | "
                      "path BASE PATTERN | python mod1,mod2 | call FINGERPRINT[/err]")
        return
    repl.session.tool_rules = list(repl.session.tool_rules or []) + [rule.to_dict()]
    repl.session.touch()
    console.print(f"[ok]added[/ok] [key]{escape(rule_label(rule))}[/key]")


def _cmd_tools_rules(repl: Repl, parts: list[str]) -> None:
    topic = parts[0].lower() if parts else ""
    if topic in {"rm", "remove", "delete"}:
        _cmd_tools_rules_rm(repl, parts[1:])
        return
    if topic == "add":
        _cmd_tools_rules_add(repl, parts[1:])
        return
    rows = _rule_rows(repl)
    if not rows:
        console.print("[dim]no stored approval rules — tick a pattern in the "
                      "confirmation picker or add one with /tools rules add[/dim]")
        return
    table = Table(title="approval rules", box=box.ROUNDED,
                  header_style="bold magenta", row_styles=["", "dim"])
    table.add_column("#", justify="right")
    table.add_column("kind")
    table.add_column("rule")
    table.add_column("created", style="dim")
    for index, (kind, pattern, base, created) in enumerate(rows, 1):
        if kind == "call":
            label = "exact: " + shortcut(pattern, 60)
        else:
            label = rule_label(ToolRule(kind=kind, pattern=pattern, base=base))
        table.add_row(str(index), kind, escape(label),
                      shortcut(created, 16) if created else "")
    console.print(table)
    console.print("[dim]/tools rules rm N… removes one · changes apply to this "
                  "session[/dim]")


def cmd_tools(repl: Repl, argument: str) -> None:
    parts = argument.split()
    topic = parts[0].lower() if parts else ""
    if not topic or topic in {"status", "list"}:
        table = Table(title="tools", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("tool")
        table.add_column("state")
        table.add_column("approval", style="dim")
        disabled = set(repl.session.tools_disabled)
        for name in sorted(repl.tool_registry):
            spec = repl.tool_registry[name]
            if not repl.tools_enabled:
                state = "[dim]off[/dim]"
            elif name in disabled:
                state = "[warn]disabled[/warn]"
            else:
                state = "[ok]on[/ok]"
            if spec.read_only:
                approval = "auto (read-only)"
            elif spec.always_safe:
                approval = "auto"
            elif name in {"run_shell", "run_python"}:
                approval = "safe list, rules, else asks"
            else:
                approval = "rules, else asks"
            table.add_row(name, state, approval)
        console.print(table)
        policy = "auto (--yes)" if repl.auto_approve else repl.tools_policy
        console.print(f"[key]policy[/key] {policy} · [key]rules[/key] "
                      f"{len(_rule_rows(repl))} (/tools rules) · [key]max rounds[/key] "
                      f"{repl.tools_max_rounds} · [key]timeout[/key] "
                      f"{repl.tool_ctx.timeout:.0f}s · [key]output cap[/key] "
                      f"{repl.tool_ctx.max_output} chars")
        return
    if topic == "on":
        repl.tools_enabled = True
        console.print("[ok]tools on[/ok]")
        return
    if topic == "off":
        repl.tools_enabled = False
        console.print("[ok]tools off[/ok]")
        return
    if topic == "policy":
        mode = parts[1].lower() if len(parts) > 1 else ""
        if mode not in {"auto", "allowlist", "ask"}:
            console.print("[err]usage: /tools policy auto|allowlist|ask[/err]")
            return
        repl.tools_policy = mode
        console.print(f"[ok]policy[/ok] {mode}")
        return
    if topic in {"enable", "disable"}:
        names = parts[1:]
        if not names:
            console.print(f"[err]usage: /tools {topic} NAME…[/err]")
            return
        disabled = list(repl.session.tools_disabled)
        for name in names:
            if name not in repl.tool_registry:
                console.print(f"[err]unknown tool {escape(name)}[/err]")
                continue
            if topic == "disable" and name not in disabled:
                disabled.append(name)
            if topic == "enable" and name in disabled:
                disabled.remove(name)
        repl.session.tools_disabled = disabled
        repl.session.touch()
        console.print(f"[ok]{topic}d[/ok] active: "
                      + (", ".join(sorted(set(repl.tool_registry) - set(disabled)))
                         or "[dim]none[/dim]"))
        return
    if topic == "allow":
        console.print("safe shell commands: " + ", ".join(sorted(repl.shell_allow)))
        return
    if topic == "rules":
        _cmd_tools_rules(repl, parts[1:])
        return
    console.print("[err]usage: /tools [on|off|policy auto|allowlist|ask|"
                  "enable NAME…|disable NAME…|allow|rules][/err]")


def cmd_tts(repl: Repl, argument: str) -> None:
    parts = argument.split()
    topic = parts[0].lower() if parts else ""
    binary = find_tts_remote(repl.tool_ctx)
    if not topic or topic == "status":
        table = Table(title="tts-remote", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("setting")
        table.add_column("value")
        table.add_row("binary", escape(binary) if binary else "[err]not found[/err]")
        table.add_row("auto-speak", "[ok]on[/ok]" if repl.tts_auto else "off")
        table.add_row("voice", escape(repl.tts_voice or "(default)"))
        table.add_row("language", escape(repl.tts_lang or "(default)"))
        states = []
        for name in ("speak", "list_voices"):
            if name not in repl.tool_registry:
                states.append(f"{name}: not installed")
            elif name in repl.session.tools_disabled:
                states.append(f"{name}: disabled")
            else:
                states.append(f"{name}: on")
        table.add_row("model access", escape(", ".join(states)))
        console.print(table)
        return
    if topic in {"on", "off"}:
        repl.tts_auto = topic == "on"
        repl.session.tts_auto = repl.tts_auto
        repl.session.touch()
        console.print(f"[ok]auto-speak[/ok] {'on' if repl.tts_auto else 'off'}")
        return
    if topic in {"voice", "lang"}:
        value = " ".join(parts[1:]).strip()
        if topic == "voice":
            repl.tts_voice = value or None
            repl.session.tts_voice = repl.tts_voice
        else:
            repl.tts_lang = value or None
            repl.session.tts_lang = repl.tts_lang
        repl.session.touch()
        console.print(f"[ok]{topic}[/ok] {escape(value or '(default)')}")
        return
    if topic == "tools":
        action = parts[1].lower() if len(parts) > 1 else ""
        if action not in {"on", "off"}:
            console.print("[err]usage: /tts tools on|off[/err]")
            return
        if not {"speak", "list_voices"} & set(repl.tool_registry):
            console.print("[warn]tts-remote not found; nothing to toggle[/warn]")
            return
        disabled = list(repl.session.tools_disabled)
        for name in ("speak", "list_voices"):
            if name not in repl.tool_registry:
                continue
            if action == "off" and name not in disabled:
                disabled.append(name)
            if action == "on" and name in disabled:
                disabled.remove(name)
        repl.session.tools_disabled = disabled
        repl.session.touch()
        console.print(f"[ok]tts tools {action}[/ok]")
        return
    if topic == "voices":
        spec = repl.tool_registry.get("list_voices")
        if spec is None:
            console.print("[err]tts-remote not found[/err]")
            return
        outcome = execute_tool(spec, {}, repl.tool_ctx)
        console.print(Panel(escape(outcome.text), title="tts voices",
                            border_style="dim" if outcome.ok else "red",
                            title_align="left"))
        return
    console.print("[err]usage: /tts [on|off|voice NAME|lang CODE|tools on|off|"
                  "voices][/err]")


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
        console.print("[dim]--max-gpu picks the largest context that still fits 100% "
                      "on the GPU (--probe verifies by loading)[/dim]")
        return
    tokens = argument.split()
    if tokens[0] in {"--help", "-h", "help"}:
        console.print(NUM_CTX_HELP)
        return
    kind, value = parse_num_ctx_arg(tokens[0])
    if kind == "max-gpu":
        probe = False
        for token in tokens[1:]:
            flag = token.lower()
            if flag in {"--probe", "probe", "-p"}:
                probe = True
            elif flag in {"--help", "-h", "help"}:
                console.print(NUM_CTX_HELP)
                return
            else:
                console.print("[err]usage: /num-ctx N | --max-gpu [--probe][/err]")
                return
        ssh_target = remote_node(repl.session.remote, repl.config) \
            if repl.session.remote else None
        ssh_user = repl.config.get("remote_ssh_user")
        base = getattr(repl.client, "base", "") or ""
        host_label = repl.session.remote or (None if is_local_host(base) else base)
        request = FitRequest(
            client=repl.client, model=repl.session.model, config=repl.config,
            host_label=host_label, ssh_target=ssh_target, ssh_user=ssh_user,
            pinned_spec=repl.session.gpu_layers,
            num_gpu=repl.session.parameters.get("num_gpu"),
            base_options={key: val for key, val in (repl.session.parameters or {}).items()
                          if key != "num_ctx"},
            keepalive=repl.session.keepalive or getattr(repl.args, "keepalive", None),
            refresh_gpu=lambda: machine_state_for(base, ssh_target=ssh_target,
                                                  ssh_user=ssh_user),
        )
        ctx = max_gpu_context(request, probe=probe)
        if ctx is None:
            return
        repl.session.parameters["num_ctx"] = ctx
        repl.session.touch()
        note = " [dim]· model is loaded at that context[/dim]" if probe else ""
        console.print(f"[ok]num_ctx set to {ctx}[/ok]{note}")
        return
    if kind == "error":
        console.print("[err]usage: /num-ctx N | --max-gpu [--probe][/err]")
        return
    requested = int(value or 0)
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


def cmd_gpu_layers(repl: Repl, argument: str) -> None:
    argument = argument.strip()
    if not argument:
        console.print(f"[key]gpu layers[/key] {escape(repl.gpu_layers_summary())}")
        console.print(f"[dim]{GPU_LAYERS_USAGE}[/dim]")
        return
    try:
        kind, number = parse_gpu_layers_spec(argument)
    except ValueError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    if kind == "auto":
        new_spec: Optional[str] = None
    elif kind == "cpu":
        new_spec = "cpu"
    elif kind == "all":
        new_spec = "all"
    else:
        current = repl.gpu_layers_value()
        try:
            resolved = resolve_gpu_layers(argument, repl._model_block_count(), current)
        except (ValueError, RuntimeError) as exc:
            console.print(f"[err]{escape(str(exc))}[/err]")
            return
        # Deltas are normalized to an absolute value; percentages stay relative
        # so switching models re-resolves them against the new layer count.
        new_spec = f"{number}%" if kind == "pct" else str(resolved)
    if new_spec == repl.session.gpu_layers:
        console.print(f"[dim]already set[/dim] — {escape(repl.gpu_layers_summary())}")
        return
    repl.session.gpu_layers = new_spec
    repl.session.touch()
    console.print(f"[ok]gpu layers[/ok] {escape(repl.gpu_layers_summary())}")
    if repl.unload_model():
        console.print("[dim]the model will reload with the new setting[/dim]")


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
    if session.tts_auto is not None:
        repl.tts_auto = bool(session.tts_auto)
    if session.tts_voice:
        repl.tts_voice = session.tts_voice
    if session.tts_lang:
        repl.tts_lang = session.tts_lang
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
    ssh_target = remote_node(repl.session.remote, repl.config) if repl.session.remote else None
    name = ensure_model(repl.client, name, repl.config, ssh_target=ssh_target,
                        host_label=repl.session.remote)
    if not name:
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


def cmd_pull(repl: Repl, argument: str) -> None:
    name = argument.strip() or pick_model(repl.client, None)
    if not name:
        return
    ssh_target = remote_node(repl.session.remote, repl.config) if repl.session.remote else None
    summary = disk_free_line(repl.client, repl.config, ssh_target=ssh_target,
                             host_label=repl.session.remote)
    console.print(f"[dim]{escape(summary)}[/dim]")
    if repl.client.model_exists(name):
        console.print(f"[dim]{escape(name)} is installed — checking for updates[/dim]")
    try:
        pull_with_progress(repl.client, name, quiet=not console.is_terminal)
    except OllamaError as exc:
        console.print(f"[err]pull failed: {escape(str(exc))}[/err]")
        return
    repl.refresh_models()
    console.print(f"[ok]pulled[/ok] {escape(name)}")


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
    ssh_target = remote_node(repl.session.remote, repl.config) if repl.session.remote else None
    chosen = ensure_model(repl.client, name, repl.config, ssh_target=ssh_target,
                          host_label=repl.session.remote)
    if not chosen:
        return
    name = chosen
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
        console.print(thinking_panel(thinking, f"thinking · {human_chars(len(thinking))}"))
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


def cmd_context(repl: Repl, argument: str) -> None:
    messages = repl.session.messages
    roles: dict[str, int] = {}
    for message in messages:
        roles[message.get("role", "?")] = roles.get(message.get("role", "?"), 0) + 1
    prompt = repl.session.stats.get("last_prompt_tokens")
    completion = repl.session.stats.get("last_completion_tokens")
    used = repl.context_used()
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
    table.add_row("last prompt tokens", str(prompt if prompt is not None else "?"))
    table.add_row("last completion tokens",
                  str(completion if completion is not None else "?"))
    table.add_row("context used", str(used if used is not None else "?"))
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
    repl.session.stats["last_completion_tokens"] = None
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
    if stats.get("last_completion_tokens"):
        table.add_row("last completion", f"{stats['last_completion_tokens']} tok")
    used = repl.context_used()
    if used:
        table.add_row("last context", f"{used} tok")
    console.print(table)


def cmd_bye(repl: Repl, argument: str) -> bool:
    return False


COMMAND_HANDLERS: dict[str, Callable[[Repl, str], Any]] = {
    "/help": cmd_help, "/?": cmd_help,
    "/set": cmd_set,
    "/show": cmd_show,
    "/load": cmd_load,
    "/save": cmd_save,
    "/pull": cmd_pull,
    "/clear": cmd_clear,
    "/new": cmd_new,
    "/sessions": cmd_sessions,
    "/title": cmd_title,
    "/tag": cmd_tag,
    "/model": cmd_model,
    "/ps": cmd_ps,
    "/num-ctx": cmd_num_ctx,
    "/gpu-layers": cmd_gpu_layers,
    "/params": cmd_params,
    "/tools": cmd_tools,
    "/tts": cmd_tts,
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


