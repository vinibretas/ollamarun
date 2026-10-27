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
from ..constants import (DEFAULT_TIMEOUT, HAS_PROMPT_TOOLKIT, HISTORY_PATH,
                         THINKING_MODES, FileHistory, KeyBindings,
                         PromptSession, console, err_console, patch_stdout,
                         tool_color)
from ..ctxfit import wait_for_ps
from ..events import (ROUND_STARTED, TOOL_CALL, TOOL_RESULT, TURN_ERROR,
                      RuntimeEvent)
from ..files import expand_file_mentions
from ..helpers import (CancelToken, HardwareSample, canonical_model_name,
                       human_tokens, local_hardware_sample,
                       local_runner_processes, local_service_env, naive_title,
                       now_iso, remote_node, resolve_show_stats,
                       service_context_length, shortcut, source_color,
                       ssh_hardware_sample, ssh_service_env, usage_color)
from ..interactive import (gpu_layers_label, model_block_count, open_in_editor,
                           parse_model_sort, resolve_gpu_layers)
from ..prompts import (default_system_prompt, load_prompt, summarize_prompt,
                       title_prompt)
from ..renderer import (GenerationResult, HeadlessRenderer, Renderer,
                        StreamSink, render_transcript, tool_result_panel,
                        user_message_box)
from ..runtime import (AgentRuntime, ApprovalDecision, ApprovalRequest,
                       TurnHooks, TurnSettings, approval_notice, build_payload)
from ..runtime import context_length as resolve_context_length
from ..runtime import context_report as build_context_report
from ..runtime import context_used as session_context_used
from ..runtime import model_show as read_model_show
from ..runtime import resident_context_length
from ..runtime import session_report as build_session_report
from ..session import Session, SessionStore
from ..tools import (DEFAULT_SHELL_ALLOW, MAX_PDF_VISION_PAGES, ToolContext,
                     ToolRule, call_detail, default_registry, find_tts_remote,
                     policy_check, rule_label, run_tts_say, speech_hosts,
                     suggest_rules)
from ..tools import tool_schemas as build_tool_schemas
from ..ui import ConsoleUi, StatusSnapshot, Ui
from ..waitstage import (StageTracker, load_pending_line, parse_show_blobs)
from .commands.registry import COMMAND_HANDLERS
from .completion import SlashCompleter
from .queue import MessageQueue
from .view import ps_table


#: Sentinel for "this cached probe has not run yet" (None is a valid answer).
_UNSET = object()

#: Capability colours on the banner's model line: the load-bearing features
#: get an identity, everything else stays dim.
_CAP_STYLES = {"tools": "ok", "thinking": "ai", "vision": "info"}


def _join_bits(bits: list[str]) -> str:
    """Styled banner fragments joined by a dim separator."""
    return " [dim]·[/dim] ".join(bits)


def _plural(count: int, word: str) -> str:
    """`1 message` / `2 messages`."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


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
        # The waiting-stage sampler behind the "waiting for model" line
        # (waitstage.py), and the loaded config the next request will change:
        # `/num-ctx` and `/gpu-layers` only store the new value, so this is
        # what tells "resident at 4,096, reloads at 8,192 on send" apart from
        # a plain prefill.
        self.wait_stage: Optional[StageTracker] = None
        self.load_pending: Optional[dict[str, Any]] = None
        # The pending reload is snapshotted when a wait starts: the idle hint
        # is cleared as soon as the request is sent, but the sampler still
        # needs the from -> to transition to name the reload.
        self._stage_pending: Optional[dict[str, Any]] = None
        self._stage_model_size: Any = _UNSET
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
        self._context_sources: dict[tuple[str, Any], str] = {}
        self._context_provisional: set[tuple[str, Any]] = set()
        # The ollama service environment measured at most once per run
        # (`server_default_context()`); `_UNSET` means "not probed yet".
        self._server_context: Any = _UNSET
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

    # ── waiting-stage sampler ───────────────────────────────────────────────

    def wait_stage_enabled(self) -> bool:
        """Whether the "waiting" line may sample the machine.

        Headless *server* turns (`ollamarun serve`) never do: a turn behind
        the HTTP API must not spawn probes. The background session host is
        headless too, but it is an interactive front end, so it keeps them.
        """
        if not self.config.get("wait_stages", True):
            return False
        if not (self.tui_mode or self.host_mode or console.is_terminal):
            return False  # nobody is watching the line: no probes
        return self.host_mode or not self.headless

    def _stage_interval(self) -> float:
        """Sampling cadence: cheap locally, one ssh per sample remotely."""
        key = "wait_stage_remote_seconds" if self.tool_ctx.ssh_target \
            else "wait_stage_sample_seconds"
        try:
            return float(self.config.get(key, 1.0) or 1.0)
        except (TypeError, ValueError):
            return 1.0

    def _stage_probe(self, with_apps: bool) -> tuple[HardwareSample, list[Any]]:
        """One hardware + runner-table reading, from the machine answering.

        Remote hosts pay one ssh round trip (the same one for both halves);
        locally it is `/proc` reads plus nvidia-smi.
        """
        remote = self.tool_ctx.ssh_target
        if remote:
            got = ssh_hardware_sample(remote, user=self.tool_ctx.ssh_user,
                                      with_apps=with_apps)
            return got if got is not None else (HardwareSample(), [])
        return (local_hardware_sample(with_apps=with_apps),
                local_runner_processes())

    def _stage_blobs(self) -> list[str]:
        """Blob names this model's runner is started with (for matching it)."""
        model = self.session.model
        if not model:
            return []
        try:
            show = self.model_show(model)
        except OllamaError:
            return []
        return parse_show_blobs(str(show.get("modelfile") or ""))

    def _stage_requested_ctx(self) -> Optional[int]:
        value = self.session.parameters.get("num_ctx")
        return int(value) if isinstance(value, int) else None

    def _stage_model_size_value(self) -> Optional[int]:
        if self._stage_model_size is not _UNSET:
            return self._stage_model_size
        model = self.session.model
        wanted = canonical_model_name(model)
        total: Optional[int] = None
        try:
            entries = self.client.tags()
        except OllamaError:
            entries = []
        for entry in entries or []:
            name = str(entry.get("name") or entry.get("model") or "")
            if name and canonical_model_name(name) == wanted:
                size = entry.get("size")
                total = int(size) if isinstance(size, int) else None
                break
        self._stage_model_size = total
        return total

    def _stage_changed(self) -> None:
        if self.settings_changed is not None:
            try:
                self.settings_changed()
            except Exception:
                pass

    def start_wait_stage(self, *, inflight: bool = False,
                         warming: bool = False) -> None:
        """Start (or re-target) the waiting-stage sampler for this wait."""
        if not self.wait_stage_enabled():
            return
        tracker = self.wait_stage
        if tracker is not None and tracker.is_running():
            if inflight:
                tracker.set_inflight(True)
            return
        # Snapshot the reload note before `_present_round` clears it: the
        # sampler keeps saying "reloading ... 4,096 -> 8,192" after the idle
        # hint has served its purpose.
        self._stage_pending = (dict(self.load_pending)
                               if self.load_pending else None)
        tracker = StageTracker(
            self.session.model,
            ps=self.client.ps,
            probe=self._stage_probe,
            blobs=self._stage_blobs,
            pending=lambda: self._stage_pending,
            requested_ctx=self._stage_requested_ctx,
            context_used=self.context_used,
            model_size=self._stage_model_size_value,
            interval=self._stage_interval(),
            on_change=self._stage_changed)
        self.wait_stage = tracker
        tracker.start(inflight=inflight, warming=warming)

    def stop_wait_stage(self, *, wait: bool = True) -> None:
        """Stop sampling (safe to call twice, or when nothing started)."""
        tracker = self.wait_stage
        if tracker is not None:
            tracker.stop(join=wait)
        self._stage_pending = None

    def wait_stage_text(self) -> Optional[str]:
        """The one line describing what the wait is actually doing."""
        tracker = self.wait_stage
        if tracker is not None:
            text = tracker.text()
            if text:
                return text
        return load_pending_line(self.load_pending or {})

    def wait_stage_active(self) -> bool:
        """Whether a wait is running (an idle pending note is not one)."""
        tracker = self.wait_stage
        return bool(tracker is not None and tracker.text())

    def mark_load_pending(self, kind: str) -> None:
        """Note that the resident runner's config differs from the session's.

        `/num-ctx` and `/gpu-layers` only store their value — the runner keeps
        serving the old one until the next message — so this is what makes
        that visible ("resident at 4,096 · reloads at 8,192 on send") and what
        lets a reload be told apart from a cold load once that message is
        sent. Nothing here touches the runner.
        """
        entry = None
        model = self.session.model
        try:
            for item in self.client.ps():
                name = str(item.get("name") or item.get("model") or "")
                if name and canonical_model_name(name) == canonical_model_name(model):
                    entry = item
                    break
        except OllamaError:
            entry = None
        if entry is None:
            self.load_pending = None
            return
        requested = self._stage_requested_ctx()
        if kind == "ctx":
            loaded = entry.get("context_length")
            if requested is None or loaded == requested:
                self.load_pending = None
            else:
                self.load_pending = {"kind": "ctx", "from": loaded,
                                     "to": requested}
            self._announce_pending()
            return
        if (self.load_pending or {}).get("kind") == "ctx":
            return  # the window mismatch is the more informative one
        self.load_pending = {"kind": "gpu", "from": None, "to": None}
        self._announce_pending()

    def _announce_pending(self) -> None:
        """Say what the next request will reload, right where it was decided."""
        note = load_pending_line(self.load_pending or {})
        if note:
            console.print(f"[dim]{note}[/dim]")
            self._stage_changed()

    def _loading_rprompt(self) -> Optional[str]:
        """Right-side prompt text (same line as the input, right-aligned).

        The animated "waiting for model" line, driven by the prompt session's
        own `refresh_interval` -- this is what makes it show up without
        blocking the prompt, unlike a separate rich.Live view fighting
        prompt_toolkit for the terminal. `rprompt` (not `bottom_toolbar`,
        which anchors to the terminal's last row and leaves a large blank
        gap) keeps this on the prompt's own line.

        The text comes from the wait-stage sampler, so it says *what* is
        happening (loading, waiting for room, reloading at a new window)
        instead of only that something is. `rprompt` is refreshed at 10 Hz
        while the sampler's own cadence is one sample per second. With no
        wait running, a pending reload still shows here -- a resident runner
        that the next message will reload -- without the spinner, since
        nothing is in flight yet.
        """
        tracker = self.wait_stage
        text = tracker.text() if tracker is not None else None
        if text:
            frame = self._SPINNER_FRAMES[
                int(time.monotonic() * 10) % len(self._SPINNER_FRAMES)]
            return f"{frame} {text} "
        thread = self._preload_thread
        if thread is not None and thread.is_alive():
            # Sampling off (or not sampled yet): still show that the preload
            # this REPL started is running.
            frame = self._SPINNER_FRAMES[
                int(time.monotonic() * 10) % len(self._SPINNER_FRAMES)]
            return f"{frame} waiting for {self.session.model}… "
        pending = load_pending_line(self.load_pending or {})
        if pending:
            return f"{pending} "
        return None

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
                                        after_tools=after_tools,
                                        on_first_token=self._on_first_token)
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
                            on_thinking_toggle=self._renderer_thinking_toggled,
                            stage_provider=self.wait_stage_text,
                            on_first_token=self._on_first_token)
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

    def load_options(self) -> dict[str, Any]:
        """The options a load should carry to match the next request.

        `num_ctx` and `num_gpu` are load-time properties, so whichever of
        them a change just touched also has to reach the loader; the resolved
        `/gpu-layers` spec wins over a raw `num_gpu` parameter, exactly as in
        the chat payload. The rest of the session's parameters ride along.
        """
        options = dict(self.session.parameters or {})
        gpu_layers = self.gpu_layers_value()
        if gpu_layers is not None:
            options["num_gpu"] = gpu_layers
        return options

    def reload_model(self, *, force: bool = False) -> Optional[int]:
        """Load the session's model with its current settings, right now.

        `num_ctx` and `num_gpu` only take effect at load time, so a change to
        either would otherwise sit in the session until the next message
        happens to reload the runner. This unloads whatever is resident (a
        reload would evict it anyway), loads again with `load_options()`, waits
        for `/api/ps` to report the window and refreshes the cached one, so
        the footer, `/context`, the banner and the warnings are right
        immediately. A model that is not loaded yet is loaded too, at the new
        settings, instead of leaving the change for the first message.

        `force` reloads even when the resident window already matches (a
        placement change with an unchanged `num_ctx`). Returns the window the
        runner reports afterwards, or None when it could not be read back.
        """
        model = self.session.model
        if not model:
            return None
        options = self.load_options()
        target = options.get("num_ctx")
        if not isinstance(target, int) or target <= 0:
            target = None
        resident = self._resident_context_length()
        if resident is not None and not force and target is not None \
                and resident == target:
            console.print(f"[dim]already loaded at {resident}[/dim]")
            return resident
        try:
            if resident is not None:
                self.client.unload(model)
                console.print(f"[dim]unloaded {escape(model)}[/dim]")
            console.print(f"[dim]loading {escape(model)}"
                          + (f" at {target:,} ctx" if target else "")
                          + "…[/dim]")
            self.client.load(model, keepalive=self.session.keepalive
                             or getattr(self.args, "keepalive", None),
                             options=options)
        except OllamaError as exc:
            console.print(f"[warn]could not load {escape(model)}: "
                          f"{escape(str(exc))}[/warn]")
            self._reset_window_cache()
            return None
        entry = wait_for_ps(self.client, model, target)
        window = entry.get("context_length") if entry else None
        if not isinstance(window, int) or window <= 0:
            window = None
        self._reset_window_cache()
        if window is None:
            console.print(f"[ok]loaded {escape(model)}[/ok] "
                          "[dim](window not reported)[/dim]")
        elif target is not None and window != target:
            console.print(f"[warn]the runner reports {window} tokens, "
                          f"not {target}[/warn]")
        else:
            console.print(f"[ok]loaded at {window}[/ok] "
                          "[dim](loaded runner)[/dim]")
        return window

    def _reset_window_cache(self) -> None:
        """Forget the cached windows after a load, then resolve the fresh one.

        Every cached answer for the model is stale once the runner changed; a
        re-resolve repopulates the current key (from `/api/ps` when the load
        landed) so the status bar keeps a denominator.
        """
        for key in [key for key in self._context_totals
                    if key[0] == self.session.model]:
            self._context_totals.pop(key, None)
            self._context_sources.pop(key, None)
            self._context_provisional.discard(key)
        self.context_length()

    def _preload_model(self) -> None:
        """Starts loading the session's model in the background so it's
        already warm by the time the first message is sent, without
        blocking the prompt from appearing -- see _loading_rprompt() for
        the "waiting for model" indicator shown while this runs, and
        _read_line()'s use of patch_stdout() for how `ollama ps` (printed
        below, once loading finishes) safely interleaves with an active
        prompt instead of corrupting it.

        The session's `num_ctx`/`num_gpu` ride along in the load options, so
        `--num-ctx`/`--gpu-layers` given at launch are in force from this
        first load instead of waiting for the first message to reload."""
        model = self.session.model
        if not model:
            return
        keepalive = self.session.keepalive or self.args.keepalive
        self.start_wait_stage(warming=True)

        def _load() -> None:
            try:
                try:
                    self.client.load(model, keepalive=keepalive,
                                     options=self.load_options())
                except OllamaError:
                    return
                # The load just applied the session's options, so a pending
                # window/placement change is no longer pending.
                self.load_pending = None
                if console.is_terminal:
                    self._maybe_show_ps()
            finally:
                # This push runs inside the load thread, where `is_alive()`
                # is still true; clear the handle first so the snapshot says
                # the preload is over instead of leaving the fallback spinner
                # up until the next turn.
                self._preload_thread = None
                self.stop_wait_stage()
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
        would reload at), the loaded runner's `/api/ps` context, the model's
        own Modelfile default, then the ollama service environment measured on
        the machine that is answering (see `server_default_context`). The
        architectural maximum from `/api/show` is never the window -- it is
        reported separately as `model_max`. Answers that `/api/ps` can still
        sharpen are provisional and re-checked until the runner shows up.
        """
        return resolve_context_length(self.client, self.session, self._show_cache,
                                      self._context_totals, self._context_sources,
                                      self._context_provisional,
                                      refresh=refresh,
                                      server_default=self.server_default_context)

    def server_default_context(self) -> Optional[tuple[int, str]]:
        """The context measured from the ollama service environment, or None.

        This is what a *cold* load will run with: the session `num_ctx` wins
        when set, a model's own `num_ctx` wins over the environment, and once
        a runner is resident `/api/ps` is the truth -- so this answers only
        when nothing else can. The probe runs at most once per run and is
        display-only: no config is ever written. `(length, "KEY=VALUE")`.
        """
        if self._server_context is _UNSET:
            self._server_context = self._probe_server_context()
        return self._server_context

    def _probe_server_context(self) -> Optional[tuple[int, str]]:
        """Read the ollama service env locally, or over ssh for `--remote`."""
        if self.headless and not self.host_mode:
            return None
        target = getattr(self.tool_ctx, "ssh_target", None)
        if target:
            env = ssh_service_env(target, user=self.tool_ctx.ssh_user,
                                  cancel=self.tool_ctx.cancel)
        else:
            env = local_service_env()
        return service_context_length(env)

    def _context_total(self) -> Optional[int]:
        """The window size for the current model/num_ctx, if known.

        The streaming renderer has usually resolved it already; when it has
        not (the turn's first status, before the footer renders), resolve on
        demand so the status bar carries the share on the first generation
        too instead of waiting for `/api/ps`.
        """
        key = (self.session.model, self.session.parameters.get("num_ctx"))
        if key not in self._context_totals:
            self.context_length()
        return self._context_totals.get(key)

    def context_report(self) -> dict[str, Any]:
        """Plain data for the `context_status` tool (window + usage)."""
        return build_context_report(self.client, self.session, self._show_cache,
                                    self._context_totals, self._context_sources,
                                    self._context_provisional,
                                    server_default=self.server_default_context)

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
        with every token instead of stepping once per turn. Before the first
        round has totals (the first generation) the base is 0, so the counter
        still ticks against the known window.
        """
        used = self.context_used()
        renderer = self.current_renderer
        if renderer is None:
            return used
        return (used or 0) + renderer.estimator.tokens

    def status_snapshot(self) -> StatusSnapshot:
        tools = "off" if not self.tools_active() else (
            "auto" if self.auto_approve else self.tools_policy)
        detail = self.state_detail
        if self.state in {"loading", "generating"}:
            # The TUI's "loading"/"generating" row says what the wait is
            # doing (loading, waiting for room, prefill), and empties again
            # once the first token arrives.
            detail = self.wait_stage_text() or detail
        return StatusSnapshot(
            state=self.state,
            detail=detail,
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
                detail = (self.wait_stage_text()
                          or f"waiting for {self.session.model} to finish loading")
                console.print(f"[dim]{escape(detail)}[/dim]")
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
        # The spinner before the first token covers a load and a prefill;
        # sample so it can say which one it is. A round is "inflight" from
        # here: the request is about to be sent.
        self.start_wait_stage(inflight=True)
        if self.load_pending is not None:
            # The note promised the reload "on send" -- this is the send, and
            # the sampler snapshotted it, so it keeps naming the transition
            # while the idle hint must not survive into the next prompt.
            self.load_pending = None
            self._stage_changed()
        try:
            result = renderer.stream(
                self.client.chat(payload, timeout=self.timeout, cancel=cancel),
                cancel=cancel, sink=self.stream_sink)
        except OllamaError as exc:
            result = GenerationResult(error=str(exc))
        finally:
            self.stop_wait_stage()
            self._last_tps = renderer.estimator.rate() or self._last_tps
            self.current_renderer = None
        return result

    def _on_first_token(self) -> None:
        """The runner answered: the wait is over and its config is in force.

        Called from the streaming thread, so it never joins the sampler (a
        probe in flight may be an ssh round trip); the sampler discards its
        own stale reading.
        """
        self.load_pending = None
        self.stop_wait_stage(wait=False)

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
        self._print_history()
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
        show = self.model_show()
        details = show.get("details") or {}
        model_bits = [f"[answer]{escape(self.session.model)}[/answer]"]
        facts = [str(bit) for bit in (details.get("parameter_size"),
                                      details.get("quantization_level")) if bit]
        if facts:
            model_bits.append(f"[dim]{escape(' · '.join(facts))}[/dim]")
        for cap in (show.get("capabilities") or []):
            name = str(cap)
            if not name or name == "completion":
                continue
            style = _CAP_STYLES.get(name, "dim")
            model_bits.append(f"[{style}]{escape(name)}[/{style}]")
        rows: list[tuple[str, str]] = [
            ("model", _join_bits(model_bits)),
            ("session", f"[dim]{self.session.id}[/dim] · "
                        f"{escape(self.session.display_title())}"),
            ("host", f"[dim]{escape(self.client.base)}[/dim]"
                     + (f" · [warn]remote {escape(self.session.remote)}[/warn]"
                        if self.session.remote else "")),
        ]
        report = self.context_report()
        rows.append(("window", self._banner_context(report)))
        usage = self._banner_usage(report)
        if usage:
            rows.append(("usage", usage))
        rows += [
            ("render", self._banner_render()),
            ("tools", self._banner_tools()),
            ("gpu", self._banner_gpu()),
        ]
        stats = self._banner_stats()
        if stats:
            rows.append(("stats", stats))
        if self.host_mode:
            rows.append(("background", f"[dim]pid {os.getpid()} · survives "
                                       "detach[/dim]"))
            rows.append(("", "[dim]/help for commands · Ctrl+D closes the "
                             "session · Ctrl+C twice detaches · Ctrl+T "
                             "thinking[/dim]"))
        else:
            rows.append(("", "[dim]/help for commands · Ctrl+D to exit · "
                             "Ctrl+T thinking[/dim]"))
        width = max(len(label) for label, _ in rows)
        lines = [f"[key]{label:<{width}}[/key] {value}" if label else value
                 for label, value in rows]
        console.print(Panel("\n".join(lines), title="ollamarun", border_style="cyan",
                            title_align="left"))

    def _banner_context(self, report: Optional[dict[str, Any]] = None) -> str:
        """The banner's context window: the measured value and its provenance.

        The value is whatever the model will actually run with -- the
        session's `num_ctx`, the loaded runner (`/api/ps`), the model's own
        default or the ollama service environment -- and the parenthesised
        text is that measured provenance. `model max` is the architectural
        maximum from `/api/show`, shown as a separate fact, never as the
        window.
        """
        report = report if report is not None else self.context_report()
        window = report.get("window")
        source = str(report.get("source") or "unknown")
        model_max = report.get("model_max")
        if window:
            style = source_color(source)
            head = (f"[bold]{window}[/bold] tok "
                    f"[{style}]({escape(source)})[/{style}]")
        else:
            head = f"[warn]{escape(source)}[/warn]"
        if model_max and model_max != window:
            head += f" · [dim]model max {model_max}[/dim]"
        return head

    def _banner_usage(self, report: Optional[dict[str, Any]] = None) -> str:
        """Message count and the share of the window in use; empty if unused."""
        report = report if report is not None else self.context_report()
        window = report.get("window")
        used = report.get("used")
        total = int((report.get("messages") or {}).get("total") or 0)
        if not total:
            return ""
        bits = [f"[dim]{_plural(total, 'message')}[/dim]"]
        if used:
            pct = used * 100 / window if window else None
            detail = f"{human_tokens(used)} used"
            if pct is not None:
                detail += f" ({pct:.0f}%)"
            color = usage_color(pct) if pct is not None else "dim"
            bits.append(f"[{color}]{detail}[/{color}]")
        return _join_bits(bits)

    def _banner_render(self) -> str:
        bits = [f"[info]{escape(self.render_mode)}[/info]"]
        if not self.math:
            bits.append("[warn]math off[/warn]")
        style = "info" if self.thinking_mode == "expanded" else "dim"
        bits.append(f"[{style}]thinking {escape(self.thinking_mode)}[/{style}]")
        if self.tts_auto:
            bits.append("[dim]tts auto[/dim]")
        bits.append(f"[dim]{len(self.model_names)} local models[/dim]")
        return _join_bits(bits)

    def _banner_tools(self) -> str:
        if not self.tools_active():
            return "[dim]off[/dim]"
        if self.auto_approve:
            mode = "[ok]auto (all allowed)[/ok]"
        else:
            mode = f"[info]{escape(self.tools_policy)}[/info]"
        return f"{mode} · [dim]{self.tools_max_rounds} rounds[/dim]"

    def _banner_gpu(self) -> str:
        summary = self.gpu_layers_summary()
        low = summary.lower()
        if "cpu" in low:
            style = "warn"
        elif low.startswith("auto"):
            style = "dim"
        else:
            style = "info"
        return f"[{style}]{escape(summary)}[/{style}]"

    def _banner_stats(self) -> str:
        """Turn/token totals for a resumed session; empty for a fresh one."""
        if not self.session.messages:
            return ""
        stats = self.session.stats or {}
        bits = [_plural(int(stats.get("turns") or 0), "turn")]
        if stats.get("prompt_tokens"):
            bits.append(f"{human_tokens(stats['prompt_tokens'])} prompt tok")
        if stats.get("completion_tokens"):
            bits.append(f"{human_tokens(stats['completion_tokens'])} completion tok")
        return f"[dim]{escape(' · '.join(bits))}[/dim]"

    def _print_history(self, *, force: bool = False) -> None:
        """Replay a resumed conversation so a session can be picked up cold.

        Called after the banner at launch (for `-c`/`--session`/`-r`) and
        whenever `/load`/`/sessions` switches the active session; a fresh or
        empty session prints nothing. A headless serve turn never reaches
        here, but stays silent if it somehow does.
        """
        if self.headless and not self.host_mode:
            return
        if not self.session.messages:
            return
        args = self.args
        resumed = force or (getattr(args, "continue_", None) is not None
                            or bool(getattr(args, "session", None))
                            or bool(getattr(args, "resume", False)))
        if not resumed:
            return
        console.print()
        render_transcript(self.session, console, show_stats=False)
        console.print()

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


