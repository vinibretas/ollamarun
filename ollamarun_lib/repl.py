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
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable, Optional

from rich import box
from rich.markup import escape
from rich.panel import Panel
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
    patch_stdout,
    pyperclip,
    tool_color,
)
from .files import expand_file_mentions
from .helpers import (
    CancelToken,
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
    resolve_show_stats,
    short_stamp,
    shortcut,
)
from .images import encode_image
from .prompts import default_system_prompt, summarize_prompt, title_prompt
from .ctxfit import (
    FitRequest,
    NUM_CTX_HELP,
    machine_state_for,
    max_cpu_context,
    max_gpu_context,
    parse_fill_percent,
    parse_num_ctx_arg,
)
from .interactive import (
    GPU_LAYERS_USAGE,
    MODEL_SORT_KEYS,
    _CODE_BLOCK_RE,
    disk_free_line,
    ensure_model,
    fetch_context_lengths,
    gpu_layers_label,
    model_block_count,
    model_context_length,
    model_table,
    open_in_editor,
    parse_gpu_layers_spec,
    parse_model_sort,
    pick_model,
    pick_session,
    pull_with_progress,
    resolve_gpu_layers,
    sort_models,
)
from .events import TOOL_CALL, TOOL_RESULT, TURN_ERROR, RuntimeEvent
from .renderer import (GenerationResult, HeadlessRenderer, Renderer, StreamSink,
                       thinking_panel, tool_call_panel, tool_result_panel,
                       user_message_box)
from .runtime import (AgentRuntime, ApprovalDecision, ApprovalRequest, TurnHooks,
                      TurnSettings, approval_notice, build_payload,
                      context_length as resolve_context_length,
                      context_report as build_context_report,
                      context_used as session_context_used,
                      model_show as read_model_show,
                      resident_context_length,
                      session_report as build_session_report)
from .session import Session, SessionStore, export_html, export_json, export_markdown
from .tools import (
    DEFAULT_SHELL_ALLOW,
    REQUIREMENT_HINTS,
    ToolContext,
    ToolDefinition,
    ToolRule,
    ToolSpec,
    call_detail,
    call_fingerprint,
    default_registry,
    execute_tool,
    find_tts_remote,
    policy_check,
    resolve_tool_alias,
    rule_label,
    run_tts_say,
    speech_hosts,
    suggest_rules,
    tool_definitions,
    tool_schemas as build_tool_schemas,
)
from .ui import (ConsoleUi, StatusSnapshot, Ui,
                 _prompt_approval)  # noqa: F401  (compatibility re-export)


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
    "/model, /models": ("[NAME|--list|--sort KEY]",
                        "switch model, list or sort the installed models"),
    "/ps": ("", "show resident models (ollama ps)"),
    "/num-ctx": ("[N | --max-gpu [--probe] | --max-cpu N [--probe]]",
                 "context window: N, the largest that keeps the GPU, or N% of DRAM"),
    "/gpu-layers": ("[N|N%|+N|-N|all|cpu|auto]", "GPU layers for this model"),
    "/params": ("", "show session parameters"),
    "/tools": ("[on|off|policy MODE|enable|disable NAME…|NAME on|off|NAME help|NAME call ARGS|allow|rules|help]",
               "tool harness, per-tool control and help"),
    "/tts": ("[on|off|voice NAME|lang CODE|tools on|off|voices]", "auto-speak via tts-remote"),
    "/think": ("[on|off|low|medium|high]", "thinking mode"),
    "/thinking": ("[show|collapse|hide|toggle|full]", "thinking display (Ctrl+T while streaming)"),
    "/system": ("[TEXT|clear|off]", "show or set the system prompt (built-in default when unset)"),
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
    "/queue": ("[list|clear|rm N]", "messages and commands typed while busy"),
    "/bye, /exit": ("", "leave"),
}


#: `/tools` subcommands (the first token), in help/table order.
TOOLS_SUBCOMMANDS: tuple[str, ...] = (
    "on", "off", "policy", "enable", "disable", "allow", "rules", "help",
    "list", "status")

#: actions after a tool name: `/tools NAME <action>`.
TOOLS_TOOL_ACTIONS: tuple[str, ...] = ("on", "off", "help", "call")

#: `/tools policy` modes (same values `approval_decision` knows).
TOOLS_POLICIES: tuple[str, ...] = ("auto", "allowlist", "ask")

#: `/tools help` reference rows: syntax -> what it does.
TOOLS_HELP_ROWS: tuple[tuple[str, str], ...] = (
    ("(none) · list · status", "table of tools with state and approval mode"),
    ("on | off", "enable or disable the whole tool harness"),
    ("policy auto|allowlist|ask", "global approval policy"),
    ("enable NAME… | disable NAME…", "toggle several tools at once"),
    ("NAME on | off", "toggle one tool, e.g. `/tools web_search off`"),
    ("NAME help", "description, parameters (required/optional), aliases"),
    ("NAME call ARGS", 'run it yourself: call {"query": "x"} · call query=x'),
    ("help [NAME]", "this reference, or one tool's page"),
    ("allow", "safe shell command allowlist"),
    ("rules [add|rm]", "stored approval rules"),
)


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
        return [key for key in MODEL_SORT_KEYS if key.startswith(fragment)]
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


@dataclass
class QueuedItem:
    """One thing typed while a turn was running: prompt text or a `/command`."""

    kind: str  # "message" | "command"
    text: str


class MessageQueue:
    """What the user typed while a turn was running, in typing order.

    Messages are handed to the model at the next sub-task boundary -- the loop
    re-sends the conversation before every `/api/chat` round, so a message
    queued during a tool call is in context for the round that reacts to the
    tool result, and one queued while an answer streams is delivered in a
    follow-up round right after it. Commands only run once the turn is over
    (`/clear`, `/undo`, `/load` … would otherwise rewrite the conversation the
    running turn is built on). Delivery stops at the first command so the
    order the user typed things in is never rearranged.
    """

    def __init__(self) -> None:
        self.items: list[QueuedItem] = []

    def add_message(self, text: str) -> "QueuedItem":
        item = QueuedItem("message", text)
        self.items.append(item)
        return item

    def add_command(self, line: str) -> "QueuedItem":
        item = QueuedItem("command", line)
        self.items.append(item)
        return item

    def has_messages(self) -> bool:
        """True when the head of the queue is a message (so a boundary may take it)."""
        return bool(self.items) and self.items[0].kind == "message"

    def take_messages(self) -> list[str]:
        taken: list[str] = []
        while self.items and self.items[0].kind == "message":
            taken.append(self.items.pop(0).text)
        return taken

    def pop_next(self) -> Optional["QueuedItem"]:
        return self.items.pop(0) if self.items else None

    def remove(self, index: int) -> Optional["QueuedItem"]:
        """Drop item `index` (1-based, as `/queue rm N` counts them)."""
        if 1 <= index <= len(self.items):
            return self.items.pop(index - 1)
        return None

    def clear(self) -> int:
        count = len(self.items)
        self.items = []
        return count

    def describe(self) -> list[str]:
        return [f"{index}. {'you' if item.kind == 'message' else 'cmd'} ▸ {item.text}"
                for index, item in enumerate(self.items, 1)]

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)


class Repl:
    def __init__(self, args: argparse.Namespace, config: dict[str, Any],
                 client: OllamaClient, store: SessionStore, session: Session,
                 artifacts: Any = None):
        self.args = args
        self.config = config
        self.client = client
        self.store = store
        self.session = session
        #: The turn-bound artifact publisher (`artifacts.TurnArtifacts`) when
        #: the host runs a store; None for the CLI, so `publish_artifact` is
        #: absent and file tools never register anything.
        self.artifacts = artifacts
        self.model_names: list[str] = []
        try:
            self.model_sort = parse_model_sort(getattr(args, "sort", None) or "modified")
        except ValueError:
            self.model_sort = "modified"
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
        self.show_stats = resolve_show_stats(
            bool(args.verbose), config.get("show_stats", True), console.is_terminal)
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
        self.tool_ctx = ToolContext(
            cwd=os.getcwd(), timeout=tool_timeout, max_output=tool_max_output,
            tts_remote=config.get("tts_remote") or None,
            client=client, store=store,
            ssh_target=remote_node(session.remote, config) if session.remote else None,
            ssh_user=config.get("remote_ssh_user"),
            chem_inventory=config.get("chem_inventory") or None,
            context_report=self.context_report,
            session_report=self.session_report,
            artifacts=artifacts)
        tts_option: Optional[bool]
        if not config.get("tts_tools", True):
            tts_option = False
        elif config.get("tts_remote") and config["tts_remote"] != "tts-remote":
            tts_option = True  # explicit path: register, the handler reports if missing
        else:
            tts_option = None  # auto-detect a tts-remote binary
        chem_option: Optional[bool]
        if not config.get("chem_tools", True):
            chem_option = False
        elif config.get("chem_inventory") and config["chem_inventory"] != "chemInventoryAPI":
            chem_option = True  # explicit path: register, the handler reports if missing
        else:
            chem_option = None  # auto-detect the chemInventoryAPI CLI
        self.tool_registry = default_registry(
            tts=tts_option, chem=chem_option, artifacts=artifacts is not None)
        self._tool_support_seen: set[str] = set()
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
        # Front end (console prompts by default; the Textual app swaps this),
        # the queue of things typed while a turn runs, and the status-bar state.
        self.ui: Ui = ConsoleUi()
        self.queue = MessageQueue()
        self.cancel_token = CancelToken()
        # The tool context carries the turn's token so stopping a turn kills
        # the subprocess-backed tool that is running (and refuses to start
        # the next one); `tool_ctx` is built before the token, so wire it here.
        self.tool_ctx.cancel = self.cancel_token
        self.current_renderer: Optional[Renderer] = None
        self.state = "ready"
        self.state_detail = ""
        self._state_since = time.monotonic()
        self._turn_started: Optional[float] = None
        self._last_tps: Optional[float] = None
        self._turn_summary = ""
        self._context_totals: dict[tuple[str, Any], Optional[int]] = {}
        self._context_provisional: set[tuple[str, Any]] = set()
        self._show_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._tool_round = 0
        # The Textual front end sets `tui_mode` and `stream_sink`; the inline
        # REPL leaves both alone (rich.Live keeps rendering the stream).
        self.tui_mode = False
        self.stream_sink: Optional[StreamSink] = None
        # The serve front end sets `headless` + `event_sink`: rendering turns
        # event-only (no Live, no stdout, no terminal prompts) so a turn can
        # run with no terminal at all.
        self.headless = False
        self.event_sink: Optional[Callable[[RuntimeEvent], None]] = None

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

    def _renderer(self, *, after_tools: bool = False) -> Any:
        if self.headless:
            renderer = HeadlessRenderer(on_event=self.event_sink,
                                        thinking_mode=self.thinking_mode,
                                        after_tools=after_tools)
            renderer.context_used = self.context_used()
            return renderer
        mode = self.render_mode
        if mode == "auto":
            mode = "markdown" if console.is_terminal else "plain"
        if not console.is_terminal and mode == "markdown":
            mode = "plain"
        renderer = Renderer(console, mode=mode, math=self.math,
                            nowordwrap=self.args.nowordwrap,
                            thinking_mode=self.thinking_mode,
                            tps_footer=self.tps_footer,
                            context_used=self.context_used(),
                            show_stats=self.show_stats,
                            max_render_chars=self.max_render_chars,
                            pager=self.pager and not self.tui_mode,
                            model=self.session.model, after_tools=after_tools)
        renderer = renderer.with_sink(self.stream_sink)
        if renderer.tps_footer and (renderer.live_render or renderer.sink_render):
            # The footer's ctx share needs the window size: one cached
            # /api/show per model, fetched only when the footer can show.
            renderer.context_total = self.context_length()
        return renderer

    def sync_live_renderer(self) -> None:
        """Push display knobs changed mid-turn (`/thinking`, `/tps`) to the
        renderer that is streaming right now."""
        renderer = self.current_renderer
        if renderer is None:
            return
        renderer.thinking_mode = self.thinking_mode
        renderer.tps_footer = self.tps_footer

    def toggle_thinking(self) -> str:
        self.thinking_mode = "collapsed" if self.thinking_mode == "expanded" else "expanded"
        self.sync_live_renderer()
        return self.thinking_mode

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
            self._block_counts[model] = model_block_count(self.model_show(model))
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

    def effective_system(self) -> Optional[str]:
        """What chat requests carry: the session's text when set, the built-in
        default otherwise (composed for the current tool state), or None when
        the session disabled system prompts with an empty string."""
        if self.session.system is not None:
            return self.session.system or None
        return default_system_prompt(with_tools=self.tools_enabled)

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
        return build_payload(self.session, messages, stream=stream, tools=tools,
                             system=self.effective_system(),
                             think=self.effective_think(),
                             keepalive=self.session.keepalive or self.args.keepalive,
                             gpu_layers=self.gpu_layers_value())

    # ── tool harness ────────────────────────────────────────────────────────

    def active_tool_schemas(self) -> list[dict[str, Any]]:
        if not self.tools_enabled:
            return []
        return build_tool_schemas(self.tool_registry, set(self.session.tools_disabled))

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
        check = policy_check(spec, args, policy=self.tools_policy,
                             shell_allow=self.shell_allow,
                             fingerprints=self.session.tool_allow or [],
                             rules=self._session_rules(), fingerprint=fingerprint,
                             cwd=self.tool_ctx.cwd)
        if check.run:
            if check.rule is not None and not self.headless:
                console.print(f"[dim]allowed by rule[/dim] "
                              f"[key]{escape(rule_label(check.rule))}[/key]")
            return True, check.note
        if not (allow_prompt and self.ui.interactive):
            return False, (f"not auto-approved ({check.reason}) and this run is "
                           "unattended")
        suggestions = suggest_rules(spec.name, args, cwd=self.tool_ctx.cwd)
        detail = call_detail(spec.name, args)
        self._emit_event(approval_notice(
            ApprovalRequest(spec=spec, args=args, fingerprint=fingerprint,
                            detail=detail, allow_prompt=allow_prompt),
            reason=check.reason, suggestions=suggestions))
        if not self.headless:
            color = tool_color(spec.name)
            console.print(f"[{color}]{escape(spec.name)}[/{color}][warn] needs "
                          f"confirmation: {escape(check.reason)}[/warn]")
        self.set_state("approval", spec.name)
        action, rule = self.ui.approval(
            suggestions,
            name=spec.name, detail=detail,
            reason=check.reason)
        if action == "auto":
            self.auto_approve = True
            self.session.touch()
            return True, "auto-allow everything (careful)"
        if action == "deny":
            return False, "denied by the user"
        if action == "remember":
            self.session.tool_allow = list(self.session.tool_allow or []) + [fingerprint]
            self.session.touch()
            if not self.headless:
                console.print("[dim]stored rule:[/dim] exactly this call")
            return True, "allowed (remembered)"
        if action == "rule" and rule is not None:
            self.session.tool_rules = list(self.session.tool_rules or []) + [rule.to_dict()]
            self.session.touch()
            if not self.headless:
                console.print(f"[dim]stored rule:[/dim] {escape(rule_label(rule))}")
            return True, "allowed (remembered)"
        return True, "confirmed"

    def _execute_tool_calls(self, calls: list[dict[str, Any]], *, quiet: bool = False,
                            allow_prompt: bool = True) -> bool:
        """Run the model's tool calls (thin wrapper over `AgentRuntime`)."""
        hooks = self._turn_hooks(allow_prompt=allow_prompt, quiet_tools=quiet)
        runtime = AgentRuntime(hooks)
        runtime.tool_round = self._tool_round
        return runtime.execute_tool_calls(calls, quiet=quiet,
                                          allow_prompt=allow_prompt)

    # ── tts ─────────────────────────────────────────────────────────────────

    def speak_text(self, text: str) -> bool:
        """Queue `text` for tts-remote playback; never blocks or raises."""
        if not text or not text.strip():
            return False
        binary = find_tts_remote(self.tool_ctx)
        if not binary:
            return False
        try:
            proc = run_tts_say(binary, text, voice=self.tts_voice or None,
                               lang=self.tts_lang or None, markdown=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            console.print(f"[warn]tts: {escape(str(exc))}[/warn]")
            return False
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            console.print(f"[warn]tts: {escape(detail or 'tts-remote failed')}[/warn]")
            return False
        if console.is_terminal:
            hosts = speech_hosts(proc.stdout)
            suffix = f" · {hosts}" if hosts else ""
            console.print(f"[dim]speaking · {len(text)} chars{suffix}[/dim]")
        return True

    # ── generation ──────────────────────────────────────────────────────────

    # ── status ──────────────────────────────────────────────────────────────

    def set_state(self, state: str, detail: str = "") -> None:
        """Record what the front end should show (also used by the TUI)."""
        self.state = state
        self.state_detail = detail
        self._state_since = time.monotonic()

    def turn_elapsed(self) -> Optional[float]:
        if self._turn_started is None:
            return None
        return time.monotonic() - self._turn_started

    def current_tps(self) -> Optional[float]:
        """Tokens/s while a response streams, or the speed of the last one."""
        renderer = self.current_renderer
        if renderer is not None:
            rate = renderer.estimator.rate()
            if rate:
                return rate
        return self._last_tps

    def current_tps_avg(self) -> Optional[float]:
        """Cumulative average of the running round (None when idle)."""
        renderer = self.current_renderer
        return renderer.estimator.average() if renderer is not None else None

    def _resident_context_length(self) -> Optional[int]:
        """The window the loaded runner actually uses, from `/api/ps`."""
        return resident_context_length(self.client, self.session)

    def model_show(self, model: Optional[str] = None) -> dict[str, Any]:
        """Raw `/api/show` for a model, cached per (host, model) for the run.

        `context_length()`, `_model_block_count()`, `context_report()` and
        `session_report()` all read from this one cache, so a burst of calls
        makes a single request; a failure is cached as `{}`.
        """
        return read_model_show(self.client, self.session, self._show_cache, model)

    def context_length(self, *, refresh: bool = False) -> Optional[int]:
        """The window the model actually runs with, cached per model/num_ctx.

        Priority: the session's `num_ctx` (what the next request asks for and
        would reload at), the loaded runner's `/api/ps` context, then the
        model's own default from `/api/show`. A show-only answer is
        *provisional* -- the model may not be loaded yet when the footer
        first asks -- so it is re-checked against `/api/ps` until the loaded
        window shows up.
        """
        return resolve_context_length(self.client, self.session, self._show_cache,
                                      self._context_totals, self._context_provisional,
                                      refresh=refresh)

    def _context_total(self) -> Optional[int]:
        """The cached window size for the current model/num_ctx, if known."""
        return self._context_totals.get(
            (self.session.model, self.session.parameters.get("num_ctx")))

    def context_report(self) -> dict[str, Any]:
        """Plain data for the `context_status` tool (window + usage)."""
        return build_context_report(self.client, self.session, self._show_cache,
                                    self._context_totals, self._context_provisional)

    def session_report(self) -> dict[str, Any]:
        """Plain data for the `session_info` tool (identity, settings, tools)."""
        return build_session_report(
            session=self.session, show=self.model_show(), base=self.client.base,
            cwd=self.tool_ctx.cwd,
            keepalive=self.session.keepalive or getattr(self.args, "keepalive", None),
            gpu_layers=self.gpu_layers_summary(),
            thinking=self.thinking_mode, render=self.render_mode,
            tools_enabled=self.tools_enabled, auto_approve=self.auto_approve,
            tools_policy=self.tools_policy, tool_round=self._tool_round,
            tools_max_rounds=self.tools_max_rounds)

    def live_context_used(self) -> Optional[int]:
        """`context_used()` plus the chunks streamed in the running round.

        The status bar (and the footer, through the renderer) count the live
        stream on top of the last round's totals, so the ctx counter moves
        with every token instead of stepping once per turn.
        """
        used = self.context_used()
        renderer = self.current_renderer
        if used is None or renderer is None:
            return used
        return used + renderer.estimator.tokens

    def status_snapshot(self) -> StatusSnapshot:
        tools = "off" if not self.tools_enabled else (
            "auto" if self.auto_approve else self.tools_policy)
        return StatusSnapshot(
            state=self.state,
            detail=self.state_detail,
            model=self.session.model,
            remote=self.session.remote,
            thinking=self.thinking_mode,
            tools=tools,
            render=self.render_mode,
            context_used=self.live_context_used(),
            context_total=self._context_total(),
            queued=len(self.queue),
            tps=self.current_tps(),
            tps_avg=self.current_tps_avg(),
            elapsed=self.turn_elapsed(),
        )

    # ── queue ───────────────────────────────────────────────────────────────

    def queue_message(self, text: str) -> None:
        self.queue.add_message(text.strip())

    def queue_command(self, line: str) -> None:
        self.queue.add_command(line.strip())

    def _deliver_queued_messages(self, texts: list[str]) -> None:
        """Append messages queued mid-turn as user turns (the next round re-sends
        the whole conversation, so the model sees them without losing its work)."""
        for text in texts:
            if console.is_terminal:
                console.print(user_message_box(text, console))
            console.print(f"[dim]delivered from queue[/dim]")
            self.session.add_message("user", text)
        if texts and not self.no_save:
            self.store.save(self.session)
        self.ui.notify("queued_delivered", count=len(texts))

    def drain_boundary_messages(self) -> list[str]:
        """Take the queued messages a sub-task boundary is allowed to deliver."""
        if not self.queue.has_messages():
            return []
        texts = self.queue.take_messages()
        self._deliver_queued_messages(texts)
        return texts

    # ── generation ──────────────────────────────────────────────────────────

    def _generate(self) -> GenerationResult:
        """Run one turn through `AgentRuntime` (the loop lives in runtime.py)."""
        if self._preload_thread is not None and self._preload_thread.is_alive():
            # The user sent a message before the background preload finished;
            # wait for our own load call to land (chat() would otherwise
            # trigger a redundant concurrent load) so ollama ps -- printed
            # right as the preload thread finishes -- shows up immediately
            # instead of only after this response is fully generated.
            if console.is_terminal and not self.headless:
                console.print(f"[dim]waiting for {escape(self.session.model)} "
                              f"to finish loading…[/dim]")
            self._preload_thread.join()
            if console.is_terminal and not self.headless:
                self._maybe_show_ps()
        self._turn_started = time.monotonic()
        self.set_state("generating")
        self.ui.notify("turn_start")
        try:
            result = AgentRuntime(self._turn_hooks()).run()
        finally:
            self._turn_started = None
            self.current_renderer = None
        if self.tts_auto and result.content and not result.cancelled and not result.error:
            self.speak_text(result.content)
        if console.is_terminal and not self.headless and not result.error:
            self._maybe_show_ps()
        return result

    def _present_round(self, payload: dict[str, Any], after_tools: bool,
                       cancel: Optional[CancelToken]) -> GenerationResult:
        """Stream one round: build a renderer, print the marker, consume chunks."""
        renderer = self._renderer(after_tools=after_tools)
        self.current_renderer = renderer
        marker = f"[ai]{escape(self.session.model)} ›[/ai]"
        if not self.headless and renderer.mode != "raw" and not renderer.render_frames:
            console.print(marker)
        try:
            result = renderer.stream(
                self.client.chat(payload, timeout=self.timeout, cancel=cancel),
                cancel=cancel, sink=self.stream_sink)
        except OllamaError as exc:
            result = GenerationResult(error=str(exc))
        finally:
            self._last_tps = renderer.estimator.rate() or self._last_tps
            self.current_renderer = None
        return result

    # ── runtime seam ────────────────────────────────────────────────────────

    def _set_tool_round(self, round_number: int) -> None:
        self._tool_round = round_number

    def _emit_event(self, event: RuntimeEvent) -> None:
        """Hand one runtime event to the front end, when one is listening."""
        if self.event_sink is not None:
            self.event_sink(event)

    def _turn_hooks(self, *, presenter: Optional[Callable[..., GenerationResult]] = None,
                    allow_prompt: bool = True,
                    quiet_tools: Optional[bool] = None) -> TurnHooks:
        """Build the per-turn seam from the REPL's live state.

        Built fresh for every turn (and for every direct `_execute_tool_calls`
        call) so a test or front end that swaps `client`/`ui`/`stream_sink`
        between turns is honoured, and so mid-turn `/tools` and `/render`
        changes are still read through the callables below.
        """
        if quiet_tools is None:
            quiet = lambda: self.headless or self.render_mode == "raw"
        else:
            quiet = lambda: bool(quiet_tools)
        settings = TurnSettings(
            no_save=self.no_save,
            timeout=self.timeout,
            allow_prompt=allow_prompt,
            quiet_tools=quiet,
            tools_enabled=lambda: self.tools_enabled,
            tools_max_rounds=lambda: self.tools_max_rounds,
        )
        return TurnHooks(
            client=self.client,
            store=self.store,
            session=self.session,
            registry=self.tool_registry,
            tool_ctx=self.tool_ctx,
            settings=settings,
            cancel=self.cancel_token,
            payload_for=lambda messages, with_tools: self._payload(
                messages, with_tools=with_tools),
            present_round=presenter or self._present_round,
            approver=self._approve_request,
            execute_tools=self._execute_tool_calls,
            tool_support_checked=self._tool_support_seen,
            on_event=self._event_printer(quiet=quiet),
            on_warning=self._warning_printer(),
            drain_boundary=self.drain_boundary_messages,
            has_queued=lambda: self.queue.has_messages(),
            set_tool_round=self._set_tool_round,
            set_state=self.set_state,
            notify=self.ui.notify,
        )

    def _event_printer(self, *, quiet: Callable[[], bool]
                       ) -> Callable[[RuntimeEvent], None]:
        """Forward every event to the sink, and print the console panels.

        The panels are exactly the ones the pre-runtime loop printed: the
        "running…" line, tool-result panels (skipping the silent
        interrupted-tool notices), and the one-line error. `quiet` is a
        callable so `/render raw` mid-turn takes effect on the next event.
        """
        def handle(event: RuntimeEvent) -> None:
            if self.event_sink is not None:
                self.event_sink(event)
            if quiet():
                return
            data = event.data
            name = str(data.get("name") or "tool")
            if event.type == TOOL_CALL:
                color = tool_color(name)
                console.print(f"[dim]running[/dim] [{color}]{escape(name)}[/{color}]"
                              f"[dim]…[/dim]")
            elif event.type == TOOL_RESULT and not data.get("silent"):
                if data.get("detail") is not None:
                    console.print(tool_result_panel(
                        name, detail=str(data.get("detail") or ""),
                        text=str(data.get("text") or ""),
                        ok=bool(data.get("ok")),
                        duration=data.get("duration") or None,
                        note=str(data.get("note") or "")))
                else:
                    console.print(tool_result_panel(
                        name, text=str(data.get("text") or ""),
                        ok=bool(data.get("ok"))))
            elif event.type == TURN_ERROR and not self.headless:
                console.print(f"[err]{escape(str(data.get('message') or ''))}[/err]")
        return handle

    def _warning_printer(self) -> Callable[[str], None]:
        def warn(message: str) -> None:
            if not self.headless:
                console.print(f"[warn]{escape(message)}[/warn]")
        return warn

    def _approve_request(self, request: ApprovalRequest) -> ApprovalDecision:
        """Adapter: the runtime asks, `_approve_tool_call` answers."""
        run, note = self._approve_tool_call(request.spec, request.args,
                                            request.fingerprint,
                                            allow_prompt=request.allow_prompt)
        return ApprovalDecision(run=run, note=note,
                                action=self._approval_action(note))

    @staticmethod
    def _approval_action(note: str) -> str:
        note = note or ""
        if note.startswith("auto"):
            return "auto"
        if note.startswith("denied") or note.startswith("not auto-approved"):
            return "deny"
        if note.startswith("interrupted"):
            return "interrupt"
        if note.startswith("allowed (remembered)"):
            return "remember"
        if note.startswith("rule:"):
            return "rule"
        return "run"

    def _add_user_message(self, text: str) -> None:
        """Record a user turn: bubble, `@file` expansion, session save, title."""
        if console.is_terminal and not self.headless:
            console.print(user_message_box(text, console))
        if self.headless:
            # A remote turn must not probe/inline local files through `@path`
            # mentions or spend an extra model call on an auto title.
            expanded = text
            included: list[str] = []
            warnings: list[str] = []
        else:
            expanded, included, warnings = expand_file_mentions(
                text, max_kb=self.max_file_kb)
        for warning in warnings:
            if not self.headless:
                console.print(f"[warn]{escape(warning)}[/warn]")
        images = self.pending_images
        self.pending_images = []
        # The bubble shows what the user typed; the stored (and sent) message
        # carries the inlined <file> blocks.
        self.session.add_message("user", expanded, images=images or None,
                                 files=included or None)
        if not self.no_save:
            self.store.save(self.session)
        if not self.session.title and not self.headless:
            self._maybe_title(text)

    def send(self, text: str) -> GenerationResult:
        text = text.strip()
        if not text and not self.pending_images:
            return GenerationResult()
        self.cancel_token.reset()
        self._add_user_message(text)
        return self._generate()

    def _maybe_title(self, first_text: str) -> None:
        title = ""
        prompt = None
        if self.auto_title and self.session.model:
            prompt = title_prompt(first_text)
        if prompt:
            try:
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
                    self.cancel_token.reset()
                    result = self.send(line)
                except KeyboardInterrupt:
                    console.print("[warn]cancelled[/warn]")
                    continue
                self.set_state("ready")
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
        return session_context_used(self.session)

    def context_warning(self) -> Optional[str]:
        """The 80%-full notice, or None; the caller decides where it goes.

        The REPL prints it; the serve controller raises it as a
        `turn.warning` event at the start of the next turn.
        """
        if not self.session.messages:
            return None
        total = self.context_length()
        used = self.context_used()
        if total and used and used / total >= 0.8:
            return (f"context {used}/{total} tokens "
                    f"({used * 100 // total}%) — /summarize to compact")
        return None

    def _context_warning(self) -> None:
        warning = self.context_warning()
        if warning:
            console.print(f"[warn]{escape(warning)}[/warn]")

    def maybe_auto_compact(self) -> Optional[str]:
        """Compact at 85% when config `auto_compact` is "auto"; the summary.

        Headless-safe: returns the summary text (or None when nothing was
        done) and prints only for a terminal front end.
        """
        total = self.context_length()
        used = self.context_used()
        if total and used and used / total >= 0.85 and len(self.session.messages) > 4:
            if not self.headless:
                console.print("[warn]context almost full — summarizing older "
                              "messages[/warn]")
            return self.summarize(ask=False)
        return None

    def _maybe_auto_compact(self) -> None:
        self.maybe_auto_compact()

    def summarize(self, *, ask: bool = True) -> Optional[str]:
        """Compact older messages into a summary; returns it (None = no-op).

        The headless serve controller calls this directly (no printing, no
        confirmation); `/summarize` keeps the interactive prompt.
        """
        quiet = self.headless
        messages = self.session.messages
        if len(messages) <= 3:
            if not quiet:
                console.print("[dim]not enough messages to summarize[/dim]")
            return None
        keep_tail = messages[-2:] if len(messages) % 2 == 0 else messages[-1:]
        older = messages[:len(messages) - len(keep_tail)]
        if not older:
            if not quiet:
                console.print("[dim]nothing to summarize[/dim]")
            return None
        if ask and not self.ui.confirm(f"summarize {len(older)} older messages?",
                                       default=True):
            return None
        transcript_lines = []
        for message in older:
            role = message.get("role", "?")
            transcript_lines.append(f"{role}: {message.get('content', '')}")
        prompt = summarize_prompt("\n\n".join(transcript_lines))
        if prompt is None:
            if not quiet:
                console.print("[err]prompt file defs/prompts/summarize.md is missing[/err]")
            return None
        if not quiet:
            console.print("[dim]summarizing…[/dim]")
        try:
            response = self.client.chat_once(build_chat_payload(
                self.session.model, [{"role": "user", "content": prompt}],
                stream=False, think=False))
            summary = (response.get("message") or {}).get("content", "").strip()
        except OllamaError as exc:
            if not quiet:
                console.print(f"[err]{escape(str(exc))}[/err]")
            return None
        if not summary:
            if not quiet:
                console.print("[err]empty summary; keeping messages[/err]")
            return None
        self.session.messages = [
            {"role": "user", "ts": now_iso(),
             "content": "[Summary of earlier conversation]\n\n" + summary},
            {"role": "assistant", "ts": now_iso(),
             "content": "Understood — continuing from that summary."},
            *keep_tail,
        ]
        self.session.touch()
        self.session.stats["last_prompt_tokens"] = None
        self.session.stats["last_completion_tokens"] = None
        if not self.no_save:
            self.store.save(self.session)
        if not quiet:
            console.print(f"[ok]compacted[/ok] {len(older)} messages → summary")
        return summary

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
    if argument == "tools":
        _cmd_tools_help_table()
        return
    if argument == "shortcuts":
        table = Table(title="shortcuts", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("key")
        table.add_column("action")
        entries = [
            ("Ctrl+D", "exit"),
            ("Ctrl+C", "cancel input / generation"),
            ("Ctrl+T", "expand or collapse the thinking panel while a "
                       "response streams"),
            ("Ctrl+G", "$OLLAMA_EDITOR for the current prompt"),
            ("Ctrl+J", "insert newline"),
            ("Ctrl+R", "search prompt history"),
            ("Up/Down", "history navigation"),
            ("Tab", "complete commands, ids, models, @files"),
            ("@path", "inline a file into the prompt"),
        ]
        if repl.tui_mode:
            entries += [
                ("Enter", "send — queues the message while a turn is running"),
                ("Ctrl+B", "send what is typed right now (a running tool finishes "
                           "first)"),
                ("Esc Esc", "stop the turn (first Esc arms it)"),
                ("Ctrl+Q", "quit (twice while busy or with a queued message)"),
                ("Ctrl+L", "clear the transcript pane"),
                ("PgUp/PgDn", "scroll the transcript"),
                ("/queue", "show, remove or drop queued messages and commands"),
            ]
        for key, action in entries:
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
        _apply_system(repl, argument[len("system"):].strip())
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
    table.add_row("system", escape(system_label(repl)))
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
        console.print("[dim]no stored approval rules — pick a pattern number in "
                      "the confirmation prompt or add one with /tools rules add[/dim]")
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


def _find_tool(repl: Repl, name: str
               ) -> tuple[Optional[ToolSpec], Optional[ToolDefinition]]:
    """(registry spec, definition) for a tool name or alias; each may be None.

    The definition exists even when the tool is not installed (its `requires`
    is unmet), so a help page can still explain what it would do.
    """
    key = str(name or "").strip().lower().replace("-", "_")
    canonical = resolve_tool_alias(key)
    spec = repl.tool_registry.get(canonical)
    definition = next((item for item in tool_definitions()
                       if item.name == canonical), None)
    return spec, definition


def _apply_tool_toggle(repl: Repl, name: str, on: bool) -> None:
    """Record NAME on/off in the session (the one place that edits it)."""
    disabled = list(repl.session.tools_disabled)
    if on:
        if name in disabled:
            disabled.remove(name)
    elif name not in disabled:
        disabled.append(name)
    repl.session.tools_disabled = disabled
    repl.session.touch()


def _missing_requirement(definition: Optional[ToolDefinition]) -> str:
    needs = definition.requires if definition is not None else None
    return REQUIREMENT_HINTS.get(needs, needs or "?")


def _tool_state(repl: Repl, spec: Optional[ToolSpec],
                definition: Optional[ToolDefinition]) -> str:
    """The state label a tool's help header and toggle share."""
    if spec is None:
        missing = escape(_missing_requirement(definition))
        return f"[err]not installed[/err] — needs {missing}"
    if not repl.tools_enabled:
        return "[dim]harness off[/dim]"
    if spec.name in repl.session.tools_disabled:
        return "[warn]disabled for the model[/warn]"
    return "[ok]on[/ok]"


def _tool_approval_line(spec: ToolSpec) -> str:
    """How a call to SPEC is approved, in the same words as the /tools table."""
    if spec.read_only:
        return "auto — read-only, never asks"
    if spec.always_safe:
        return "auto — runs without confirmation"
    if spec.name == "run_shell":
        return ("safe-command allowlist and stored rules, else asks "
                "(/tools allow, /tools rules)")
    if spec.name == "run_python":
        return "AST-safe snippets and stored rules, else asks (/tools rules)"
    return "stored rules, else asks (/tools rules)"


def _print_tool_help(repl: Repl, spec: Optional[ToolSpec],
                     definition: Optional[ToolDefinition]) -> None:
    """One tool's page: state, description, parameters and approval."""
    source = spec if spec is not None else definition
    if source is None:
        return
    canonical = source.name
    aliases = spec.aliases if spec is not None else definition.aliases
    color = tool_color(canonical)
    header = (f"[{color}]{escape(canonical)}[/{color}] — "
              f"{_tool_state(repl, spec, definition)}")
    if aliases:
        header += " · aliases: " + escape(", ".join(aliases))
    console.print(header)
    console.print(escape(source.description))
    parameters = source.parameters or {}
    properties = parameters.get("properties") or {}
    required = [key for key in (parameters.get("required") or [])]
    order = list(properties)
    signature = ", ".join(key + ("" if key in required else "?") for key in order)
    console.print(f"[key]model calls it with[/key] {canonical}({signature})"
                  + ("" if order else " — no arguments"))
    if order:
        table = Table(box=box.SIMPLE, header_style="bold", pad_edge=False)
        table.add_column("parameter", style=color)
        table.add_column("type", style="dim")
        table.add_column("", style="dim")
        table.add_column("description")
        for key in order:
            prop = properties.get(key) or {}
            table.add_row(escape(key), escape(str(prop.get("type") or "any")),
                          "required" if key in required else "optional",
                          escape(str(prop.get("description") or "")))
        console.print(table)
    if spec is not None:
        console.print(f"[key]approval[/key] {_tool_approval_line(spec)}")
    console.print(f"[dim]/tools {escape(canonical)} "
                  f"{escape('[on|off|help|call ARGS]')}[/dim]")


def _tool_call_args(spec: ToolSpec, raw: str) -> dict[str, Any]:
    """Arguments for `/tools NAME call`: JSON, key=value pairs, or one bare
    value when exactly one parameter is required (`call notes.txt`).

    Everything is checked against the tool's schema: unknown or missing
    parameters are refused with the valid names spelled out.
    """
    text = raw.strip()
    parameters = spec.parameters or {}
    properties = parameters.get("properties") or {}
    required = [key for key in (parameters.get("required") or [])]
    if not text:
        args: dict[str, Any] = {}
    elif text.startswith("{"):
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"invalid JSON arguments: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("JSON arguments must be an object")
        args = value
    else:
        try:
            tokens = shlex.split(text)
        except ValueError as exc:
            raise ValueError(f"unparsable arguments: {exc}") from exc
        keyed = [token for token in tokens if "=" in token]
        if keyed and len(keyed) == len(tokens):
            args = {}
            for token in tokens:
                key, _, value = token.partition("=")
                if not key:
                    raise ValueError(f"missing parameter name in {token!r}")
                args[key] = coerce_value(value)
        elif not keyed and len(required) == 1:
            args = {required[0]: " ".join(tokens)}
        else:
            raise ValueError(
                'usage: /tools NAME call \'{"parameter": value}\' | parameter=value …'
                + (f" | VALUE (for {required[0]})" if len(required) == 1 else ""))
    unknown = [key for key in args if key not in properties]
    if unknown:
        takes = (f"{spec.name} takes: " + ", ".join(properties)
                 if properties else f"{spec.name} takes no arguments")
        raise ValueError(f"unknown parameter(s) {', '.join(unknown)} — {takes}")
    missing = [key for key in required if key not in args]
    if missing:
        raise ValueError(f"missing required parameter(s): {', '.join(missing)}"
                         f" (/tools {spec.name} help)")
    return args


def _cmd_tools_call(repl: Repl, spec: ToolSpec, raw: str) -> None:
    """Run one tool exactly like the model would: approve, execute, render.

    The result is shown but never added to the conversation: a manual call
    must not change what the model sees (or how much context it carries).
    """
    try:
        args = _tool_call_args(spec, raw)
    except ValueError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    detail = call_detail(spec.name, args)
    console.print(tool_call_panel(spec.name, args))
    if spec.name in repl.session.tools_disabled:
        console.print(f"[warn]{escape(spec.name)} is disabled for the model "
                      f"— this manual call does not change that[/warn]")
    try:
        run, note = repl._approve_tool_call(spec, args,
                                            call_fingerprint(spec.name, args))
    except KeyboardInterrupt:
        console.print("[warn]cancelled[/warn]")
        return
    if not run:
        console.print(tool_result_panel(spec.name, detail=detail,
                                        text=f"not run: {note}", ok=False))
        return
    repl.set_state("tool", detail or spec.name)
    repl.ui.notify("tool_start", name=spec.name, detail=detail)
    started = time.monotonic()
    try:
        outcome = execute_tool(spec, args, repl.tool_ctx)
    except KeyboardInterrupt:
        console.print(tool_result_panel(spec.name, detail=detail,
                                        text="interrupted by user", ok=False))
        repl.set_state("ready")
        return
    repl.ui.notify("tool_done", name=spec.name, ok=outcome.ok,
                   duration=outcome.duration or (time.monotonic() - started))
    console.print(tool_result_panel(spec.name, detail=detail, text=outcome.text,
                                    ok=outcome.ok, duration=outcome.duration,
                                    note=outcome.title_note))
    repl.set_state("ready")


def _cmd_tools_named(repl: Repl, name: str, parts: list[str]) -> None:
    """`/tools NAME [on|off|help|call ARGS]` for one tool (aliases resolve)."""
    spec, definition = _find_tool(repl, name)
    if spec is None and definition is None:
        console.print(f"[err]unknown tool {escape(name)}[/err]")
        suggestions = get_close_matches(name.lower(),
                                        tool_name_candidates(repl, ""), n=3)
        if suggestions:
            console.print("[dim]did you mean: " + ", ".join(suggestions) + "?[/dim]")
        console.print("[dim]/tools lists every tool[/dim]")
        return
    canonical = spec.name if spec is not None else definition.name
    action = parts[0].lower() if parts else ""
    if action in {"", "help", "info", "?"}:
        _print_tool_help(repl, spec, definition)
        return
    if action in {"on", "off", "enable", "disable"}:
        if spec is None:
            console.print(f"[err]{escape(canonical)} is not installed[/err] — "
                          f"needs {escape(_missing_requirement(definition))}")
            return
        on = action in {"on", "enable"}
        _apply_tool_toggle(repl, spec.name, on)
        console.print(f"[ok]{escape(spec.name)} {'on' if on else 'off'}[/ok]")
        if not repl.tools_enabled:
            console.print("[dim]note: the tool harness is off globally — "
                          "/tools on to bring it back[/dim]")
        return
    if action == "call":
        if spec is None:
            console.print(f"[err]{escape(canonical)} is not installed[/err] — "
                          f"needs {escape(_missing_requirement(definition))}")
            return
        _cmd_tools_call(repl, spec, " ".join(parts[1:]))
        return
    console.print(f"[err]usage: /tools {escape(name)} "
                  f"{escape('[on|off|help|call ARGS]')}[/err]")


def _cmd_tools_help_table() -> None:
    """`/tools help`: every form the command accepts, in one table."""
    table = Table(title="/tools", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("form")
    table.add_column("what it does")
    for form, what in TOOLS_HELP_ROWS:
        table.add_row(escape(form), what)
    console.print(table)
    console.print("[dim]NAME is any tool name or alias (Tab completes them) · "
                  "/tools NAME help shows its parameters[/dim]")


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
            color = tool_color(name)
            table.add_row(f"[{color}]{escape(name)}[/{color}]", state, approval)
        console.print(table)
        policy = "auto (--yes)" if repl.auto_approve else repl.tools_policy
        console.print(f"[key]policy[/key] {policy} · [key]rules[/key] "
                      f"{len(_rule_rows(repl))} (/tools rules) · [key]max rounds[/key] "
                      f"{repl.tools_max_rounds} · [key]timeout[/key] "
                      f"{repl.tool_ctx.timeout:.0f}s · [key]output cap[/key] "
                      f"{repl.tool_ctx.max_output} chars")
        console.print("[dim]/tools NAME help shows one tool's parameters · "
                      "/tools help explains every form[/dim]")
        return
    if topic == "help":
        if len(parts) > 1:
            _cmd_tools_named(repl, parts[1], parts[2:])
        else:
            _cmd_tools_help_table()
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
        for name in names:
            spec, definition = _find_tool(repl, name)
            if spec is None:
                hint = (f" — {escape(definition.name)} is not installed"
                        if definition is not None else "")
                console.print(f"[err]unknown tool {escape(name)}[/err]{hint}")
                continue
            _apply_tool_toggle(repl, spec.name, topic == "enable")
        console.print(f"[ok]{topic}d[/ok] active: "
                      + (", ".join(sorted(set(repl.tool_registry)
                                         - set(repl.session.tools_disabled)))
                         or "[dim]none[/dim]"))
        if not repl.tools_enabled:
            console.print("[dim]note: the tool harness is off globally — /tools on[/dim]")
        return
    if topic == "allow":
        console.print("safe shell commands: " + ", ".join(sorted(repl.shell_allow)))
        return
    if topic == "rules":
        _cmd_tools_rules(repl, parts[1:])
        return
    # any other first token is a tool name or an alias for one
    spec, definition = _find_tool(repl, parts[0])
    if spec is not None or definition is not None:
        _cmd_tools_named(repl, parts[0], parts[1:])
        return
    console.print(f"[err]unknown tool or subcommand {escape(parts[0])}[/err]")
    suggestions = get_close_matches(
        parts[0].lower(),
        list(TOOLS_SUBCOMMANDS) + tool_name_candidates(repl, ""), n=3)
    if suggestions:
        console.print("[dim]did you mean: " + ", ".join(suggestions) + "?[/dim]")
    console.print("[err]usage: /tools "
                  + escape("[on|off|policy MODE|enable|disable NAME…|"
                           "NAME on|off|NAME help|NAME call ARGS|allow|rules|help]")
                  + "[/err]")


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
    usage = "usage: /num-ctx N | --max-gpu [--probe] | --max-cpu N [--probe]"
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
                      "on the GPU; --max-cpu N fills N% of the DRAM available to the "
                      "model (--probe verifies by loading)[/dim]")
        return
    tokens = argument.split()
    if tokens[0] in {"--help", "-h", "help"}:
        console.print(NUM_CTX_HELP)
        return

    def fit_request() -> FitRequest:
        ssh_target = remote_node(repl.session.remote, repl.config) \
            if repl.session.remote else None
        ssh_user = repl.config.get("remote_ssh_user")
        base = getattr(repl.client, "base", "") or ""
        host_label = repl.session.remote or (None if is_local_host(base) else base)
        return FitRequest(
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

    def apply(ctx: Optional[int], *, probe: bool) -> None:
        if ctx is None:
            return
        repl.session.parameters["num_ctx"] = ctx
        repl.session.touch()
        note = " [dim]· model is loaded at that context[/dim]" if probe else ""
        console.print(f"[ok]num_ctx set to {ctx}[/ok]{note}")

    head = tokens[0].lower()
    rest = tokens[1:]
    max_cpu: Optional[str] = None
    if head in {"--max-cpu", "max-cpu"}:
        if not rest:
            console.print(f"[err]{usage} — --max-cpu needs a percentage[/err]")
            return
        max_cpu, rest = rest[0], rest[1:]
    elif head.startswith("--max-cpu="):
        max_cpu = tokens[0].split("=", 1)[1]
    if max_cpu is not None:
        percent = parse_fill_percent(max_cpu)
        if percent is None:
            console.print("[err]--max-cpu takes a percentage between 1 and 100[/err]")
            return
        probe = False
        for token in rest:
            flag = token.lower()
            if flag in {"--probe", "probe", "-p"}:
                probe = True
            elif flag in {"--help", "-h", "help"}:
                console.print(NUM_CTX_HELP)
                return
            else:
                console.print(f"[err]{usage}[/err]")
                return
        apply(max_cpu_context(fit_request(), percent=percent, probe=probe), probe=probe)
        return

    kind, value = parse_num_ctx_arg(tokens[0])
    if kind == "max-gpu":
        probe = False
        for token in rest:
            flag = token.lower()
            if flag in {"--probe", "probe", "-p"}:
                probe = True
            elif flag in {"--help", "-h", "help"}:
                console.print(NUM_CTX_HELP)
                return
            else:
                console.print(f"[err]{usage}[/err]")
                return
        apply(max_gpu_context(fit_request(), probe=probe), probe=probe)
        return
    if kind == "error":
        console.print(f"[err]{usage}[/err]")
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


def _pick_model(repl: Repl, current: Optional[str] = None) -> Optional[str]:
    """The model picker ordered by the session's sort; a sort chosen inside
    the picker sticks for the rest of the session."""
    return pick_model(repl.client, current, ui=repl.ui, sort=repl.model_sort,
                      on_sort=lambda key: setattr(repl, "model_sort", key))


def _print_model_list(repl: Repl, sort: str) -> None:
    try:
        models = repl.client.tags()
    except OllamaError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    contexts = fetch_context_lengths(repl.client, models) if sort == "context" else {}
    console.print(model_table(sort_models(models, sort, contexts=contexts),
                              sort=sort, contexts=contexts))


def cmd_pull(repl: Repl, argument: str) -> None:
    name = argument.strip() or _pick_model(repl)
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


def cmd_model(repl: Repl, argument: str) -> None:
    try:
        tokens = shlex.split(argument)
    except ValueError as exc:
        console.print(f"[err]unparsable arguments: {escape(str(exc))}[/err]")
        return
    wants_list = False
    sort: Optional[str] = None
    name_parts: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--sort" or token.startswith("--sort="):
            value = token.split("=", 1)[1] if "=" in token else (
                tokens[index + 1] if index + 1 < len(tokens) else "")
            if not value:
                console.print("[err]usage: /model --sort "
                              + "|".join(MODEL_SORT_KEYS) + "[/err]")
                return
            try:
                sort = parse_model_sort(value)
            except ValueError as exc:
                console.print(f"[err]{escape(str(exc))}[/err]")
                return
            index += 1 if "=" in token else 2
            continue
        if token == "--list":
            wants_list = True
        else:
            name_parts.append(token)
        index += 1
    if sort is not None:
        repl.model_sort = sort
    if wants_list:
        _print_model_list(repl, repl.model_sort)
        return
    name = " ".join(name_parts).strip() or _pick_model(repl, repl.session.model)
    if not name:
        return
    ssh_target = remote_node(repl.session.remote, repl.config) if repl.session.remote else None
    chosen = ensure_model(repl.client, name, repl.config, ssh_target=ssh_target,
                          host_label=repl.session.remote, ui=repl.ui)
    if not chosen:
        return
    name = chosen
    if name == repl.session.model:
        console.print("[dim]already using that model[/dim]")
        return
    if repl.session.messages and not repl.ui.confirm(
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


def _apply_system(repl: Repl, text: str) -> None:
    """One place for the /system and /set system semantics.

    None means "use the built-in default", an empty string means "send no
    system prompt at all", any other text is the session's own prompt.
    """
    if text in ("off", "none"):
        repl.session.system = ""
        console.print("[ok]system prompt disabled[/ok]")
    elif text in ("", "clear", "default", "reset"):
        repl.session.system = None
        console.print("[ok]system prompt reset to the built-in default[/ok]")
    else:
        repl.session.system = text
        console.print("[ok]system prompt set[/ok]")
    repl.session.touch()


def _show_system(repl: Repl) -> None:
    if repl.session.system:
        console.print(Panel(escape(repl.session.system), title="system",
                            border_style="dim", title_align="left"))
        return
    if repl.session.system == "":
        console.print("[dim]no system prompt (disabled; /system clear re-enables "
                      "the built-in default)[/dim]")
        return
    text = repl.effective_system() or ""
    if not text:
        console.print("[warn]the built-in system prompt file is missing "
                      "(defs/prompts/system.md)[/warn]")
        return
    console.print(Panel(escape(text), title="system (built-in default)",
                        border_style="dim", title_align="left"))


def system_label(repl: Repl) -> str:
    """One-line system-prompt state for the parameter/session tables."""
    if repl.session.system:
        return shortcut(repl.session.system, 60)
    if repl.session.system == "":
        return "disabled"
    text = repl.effective_system()
    return f"built-in default ({len(text)} chars)" if text else "none"


def cmd_system(repl: Repl, argument: str) -> None:
    if not argument:
        _show_system(repl)
        return
    _apply_system(repl, argument.strip())


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


def cmd_context(repl: Repl, argument: str) -> None:
    messages = repl.session.messages
    roles: dict[str, int] = {}
    for message in messages:
        roles[message.get("role", "?")] = roles.get(message.get("role", "?"), 0) + 1
    prompt = repl.session.stats.get("last_prompt_tokens")
    completion = repl.session.stats.get("last_completion_tokens")
    used = repl.context_used()
    total = repl.context_length()
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
    repl.summarize(ask=not argument)


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


def cmd_queue(repl: Repl, argument: str) -> None:
    """Show or edit what was typed while a turn was running."""
    parts = argument.split()
    topic = parts[0].lower() if parts else "list"
    if topic in {"list", ""}:
        if not repl.queue:
            console.print("[dim]queue is empty[/dim]")
            return
        table = Table(title="queued", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("#", justify="right")
        table.add_column("kind")
        table.add_column("text")
        for index, item in enumerate(repl.queue.items, 1):
            table.add_row(str(index),
                          "you" if item.kind == "message" else "cmd",
                          escape(shortcut(item.text, 72)))
        console.print(table)
        console.print("[dim]messages are delivered at the next sub-task boundary · "
                      "commands run when the turn ends[/dim]")
        return
    if topic in {"clear", "flush", "drop"}:
        count = repl.queue.clear()
        console.print(f"[ok]dropped {count} queued item(s)[/ok]")
        repl.ui.notify("queue_changed")
        return
    if topic in {"rm", "remove", "del"}:
        if len(parts) < 2 or not parts[1].isdigit():
            console.print("[err]usage: /queue rm N[/err]")
            return
        item = repl.queue.remove(int(parts[1]))
        if item is None:
            console.print("[err]no such queued item[/err]")
            return
        console.print(f"[ok]dropped[/ok] {escape(shortcut(item.text, 72))}")
        repl.ui.notify("queue_changed")
        return
    console.print("[err]usage: /queue [list|clear|rm N][/err]")


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
    "/model": cmd_model, "/models": cmd_model,
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
    "/queue": cmd_queue,
    "/bye": cmd_bye, "/exit": cmd_bye,
}


