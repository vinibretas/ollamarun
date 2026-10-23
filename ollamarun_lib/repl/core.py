"""The `Repl`: one interactive session and its turn plumbing.

The slash commands live in `commands/`, completion in `completion.py`,
the typed queue in `queue.py`; this module owns the session state, the
request payload, the approval gate, the turn hooks and the read loop.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
from typing import Any, Callable, Optional
from rich.markup import escape
from rich.panel import Panel
from ..client import OllamaClient, OllamaError, build_chat_payload
from ..constants import (
    DEFAULT_TIMEOUT,
    HAS_PROMPT_TOOLKIT,
    HISTORY_PATH,
    THINKING_MODES,
    FileHistory,
    KeyBindings,
    PromptSession,
    console,
    err_console,
    patch_stdout,
    tool_color,
)
from ..files import expand_file_mentions
from ..helpers import (
    CancelToken,
    naive_title,
    now_iso,
    remote_node,
    resolve_show_stats,
    shortcut,
)
from ..prompts import default_system_prompt, load_prompt, summarize_prompt, title_prompt
from ..interactive import (
    gpu_layers_label,
    model_block_count,
    open_in_editor,
    parse_model_sort,
    resolve_gpu_layers,
)
from ..events import ROUND_STARTED, TOOL_CALL, TOOL_RESULT, TURN_ERROR, RuntimeEvent
from ..renderer import (
    GenerationResult,
    HeadlessRenderer,
    Renderer,
    StreamSink,
    tool_result_panel,
    user_message_box,
)
from ..runtime import (
    AgentRuntime,
    ApprovalDecision,
    ApprovalRequest,
    TurnHooks,
    TurnSettings,
    approval_notice,
    build_payload,
    context_length as resolve_context_length,
    context_report as build_context_report,
    context_used as session_context_used,
    model_show as read_model_show,
    resident_context_length,
    session_report as build_session_report,
)
from ..session import Session, SessionStore
from ..tools import (
    DEFAULT_SHELL_ALLOW,
    MAX_PDF_VISION_PAGES,
    ToolContext,
    ToolRule,
    call_detail,
    default_registry,
    find_tts_remote,
    policy_check,
    rule_label,
    run_tts_say,
    speech_hosts,
    suggest_rules,
    tool_schemas as build_tool_schemas,
)
from ..ui import ConsoleUi, StatusSnapshot, Ui
from .completion import SlashCompleter
from .queue import MessageQueue
from .view import ps_table
from .commands.registry import COMMAND_HANDLERS


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
        try:
            pdf_vision_pages = int(config.get("pdf_vision_pages")
                                   or MAX_PDF_VISION_PAGES)
        except (TypeError, ValueError):
            pdf_vision_pages = MAX_PDF_VISION_PAGES
        self.tool_ctx = ToolContext(
            cwd=os.getcwd(), timeout=tool_timeout, max_output=tool_max_output,
            tts_remote=config.get("tts_remote") or None,
            client=client, store=store,
            ssh_target=remote_node(session.remote, config) if session.remote else None,
            ssh_user=config.get("remote_ssh_user"),
            chem_inventory=config.get("chem_inventory") or None,
            context_report=self.context_report,
            session_report=self.session_report,
            artifacts=artifacts,
            current_model=lambda: self.session.model,
            model_show=self.model_show,
            pdf_vision=bool(config.get("pdf_vision", True)),
            pdf_vision_switch=bool(config.get("pdf_vision_switch", True)),
            pdf_vision_model=config.get("pdf_vision_model") or None,
            pdf_vision_pages=pdf_vision_pages,
            vision_prompt=load_prompt("pdf_vision.md") or "")
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
        # The background session host (`session_host.py`) runs this Repl with
        # no terminal: `host_mode` keeps @file expansion, titles and the user
        # bubble (as events), `prompt_provider` reads the next line from the
        # attached client, `turn_runner` routes turns through its TurnManager,
        # and `switch_guard` refuses /load into a session another host owns.
        self.host_mode = False
        self.prompt_provider: Optional[Callable[[str], str]] = None
        self.turn_runner: Optional[Callable[["Repl"], Any]] = None
        self.switch_guard: Optional[Callable[[Session], bool]] = None
        self.settings_changed: Optional[Callable[[], None]] = None
        self._pending_user_event: Optional[dict[str, Any]] = None

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

        @bindings.add("c-q")
        def _ignore_stop(event) -> None:  # pragma: no cover - interactive
            # Ctrl+Q stops a streaming generation; at the idle prompt it is a
            # no-op (and must not become prompt_toolkit's quoted-insert).
            return

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
        if self.prompt_provider is not None:
            return self.prompt_provider(message)
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
                            model=self.session.model, after_tools=after_tools,
                            on_thinking_toggle=self._renderer_thinking_toggled)
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

    def _renderer_thinking_toggled(self, mode: str) -> None:
        """A Ctrl+T toggle: make it the mode later rounds/turns start with.

        The renderer's own field only covers the round that is streaming;
        `_renderer()` seeds every new one from `self.thinking_mode`, which is
        what `/thinking` changes too.
        """
        if mode in THINKING_MODES:
            self.thinking_mode = mode

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
            (err_console if quiet else console).print(
                f"[warn]could not unload {escape(target)}: "
                f"{escape(str(exc))}[/warn]")
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
                try:
                    self.client.load(model, keepalive=keepalive)
                except OllamaError:
                    return
                if console.is_terminal:
                    self._maybe_show_ps()
            finally:
                if self.settings_changed is not None:
                    try:
                        self.settings_changed()
                    except Exception:
                        pass

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

    def tools_active(self) -> bool:
        """Whether requests actually carry tools.

        The toggle (`--tools`/`/tools on`, config `tools`) is the intent; a
        response format (`--format`, restored with a resumed session) vetoes
        it, because Ollama's function calling and a forced output shape do not
        combine. Every reader -- the system prompt, the payload, the runtime
        settings, `/tools`, `session_info` -- goes through this one predicate
        so the harness can never look on while it is stepping aside.
        """
        return bool(self.tools_enabled and not self.session.format)

    def effective_system(self) -> Optional[str]:
        """What chat requests carry: the session's text when set, the built-in
        default otherwise (composed for the current tool state), or None when
        the session disabled system prompts with an empty string."""
        if self.session.system is not None:
            return self.session.system or None
        return default_system_prompt(with_tools=self.tools_active())

    def _payload(self, messages: list[dict[str, Any]], *, stream: bool = True,
                 with_tools: Optional[bool] = None) -> dict[str, Any]:
        if with_tools is None:
            with_tools = self.tools_active()
        if self.tools_enabled and self.session.format:
            if (not self._format_tools_warned and not self.headless
                    and not getattr(self.args, "json", False)):
                self._format_tools_warned = True
                console.print("[warn]--format and tools cannot be combined; tools "
                              "disabled for this session[/warn]")
            with_tools = False
        tools = None
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
        if not self.tools_active():
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
            tools_enabled=self.tools_active(), auto_approve=self.auto_approve,
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
        tools = "off" if not self.tools_active() else (
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
            console.print("[dim]delivered from queue[/dim]")
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
        """Run one turn; a session host supplies its own turn runner.

        The host wrapper keeps the whole agent loop (and its replay buffer)
        in the detached process; the foreground path is unchanged.
        """
        if self.turn_runner is not None:
            return self.turn_runner(self)
        return self._generate_local()

    def _generate_local(self) -> GenerationResult:
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
        if self.host_mode and self.event_sink is not None:
            # A remote front end needs the per-round render settings before
            # the first delta: which model, whether tools ran, and the ctx
            # counters for the footer. Local rendering does not need this.
            self._emit_event(RuntimeEvent(ROUND_STARTED, {
                "round": self._tool_round,
                "after_tools": bool(after_tools),
                "model": self.session.model,
                "context_used": self.context_used(),
                "context_total": self.context_length(),
            }))
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
            tools_enabled=lambda: self.tools_active(),
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
        """Record a user turn: bubble, `@file` expansion, session save, title.

        A background host is "headless" for rendering but still local: it
        expands `@file` mentions, titles the session and hands the bubble to
        the attached client as a `message.user` event instead of printing it.
        A serve/API turn is the one that must not touch local files.
        """
        host = self.host_mode
        if console.is_terminal and not self.headless and not host:
            console.print(user_message_box(text, console))
        if self.headless and not host:
            # A remote API turn must not probe/inline local files through
            # `@path` mentions or spend an extra model call on a title.
            expanded = text
            included: list[str] = []
            warnings: list[str] = []
        else:
            expanded, included, warnings = expand_file_mentions(
                text, max_kb=self.max_file_kb)
        if host:
            self._pending_user_event = {
                "text": text,
                "warnings": list(warnings),
                "files": list(included),
                "images": [image.get("name") or image.get("path")
                           for image in self.pending_images],
            }
        else:
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
        if not self.session.title and (host or not self.headless):
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
        self._pt_session = None if self.host_mode else self._build_prompt_session()
        if self.host_mode:
            if self.session.system:
                console.print(f"[dim]system prompt: "
                              f"{escape(shortcut(self.session.system, 80))}[/dim]")
            elif self.session.system == "":
                console.print("[dim]system prompt disabled[/dim]")
            elif console.is_terminal:
                console.print("[dim]system prompt: built-in default[/dim]")
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
        if self.host_mode:
            lines.append(f"[key]background[/key] pid {os.getpid()} · survives "
                         "detach")
            lines.append("[dim]/help for commands · Ctrl+D closes the session · "
                         "Ctrl+C twice detaches · Ctrl+T thinking[/dim]")
        else:
            lines.append("[dim]/help for commands · Ctrl+D to exit · "
                         "Ctrl+T thinking[/dim]")
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


