"""The agent turn runtime: one loop shared by every front end.

`AgentRuntime` owns the multi-round generation loop and tool execution that
used to live inside `Repl._generate`/`Repl._execute_tool_calls`. It never
prints, never touches a terminal and never knows which front end is running:
the host supplies a `TurnHooks` bundle with the seams it needs --

* `payload_for` / `present_round` -- how a round is built and displayed (the
  CLI builds a `rich` renderer there; the API server uses `HeadlessRenderer`),
* `approver` -- the policy + prompt question for a tool call, wrapped in
  `ApprovalRequest`/`ApprovalDecision`,
* `execute_tools` -- defaulting to `AgentRuntime.execute_tool_calls`, so a
  host can override it (and the CLI's boundary-delivery test can too),
* `on_event` -- the typed `RuntimeEvent` stream (see `events.py`) that a
  non-terminal front end turns into SSE/JSON.

The CLI keeps its old methods (`Repl._generate`, `Repl._execute_tool_calls`,
`Repl._payload`, `Repl.context_report`, `Repl.session_report`) as thin
wrappers so every existing caller and test keeps working; the server builds
the same hooks directly.

`build_payload()` and the context/session report helpers live here because
both front ends need exactly one implementation of them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .client import OllamaError, build_chat_payload
from .constants import DEFAULT_TIMEOUT
from .events import (APPROVAL_REQUIRED, APPROVAL_RESOLVED, MESSAGE_COMPLETED,
                     TOOL_CALL, TOOL_RESULT, TURN_CANCELLED, TURN_COMPLETED,
                     TURN_ERROR, TURN_STARTED, TURN_WARNING, RuntimeEvent)
from .helpers import CancelToken, canonical_model_name
from .interactive import model_context_length
from .renderer import GenerationResult
from .session import Session
from .tools import (ToolContext, ToolRule, ToolRunner, call_detail,
                    call_fingerprint, normalize_arguments, resolve_tool_alias)


def _always_true() -> bool:
    return True


def _always_false() -> bool:
    return False


def _default_rounds() -> int:
    return 8


@dataclass
class TurnSettings:
    """How one turn behaves; a front end supplies these per turn.

    `tools_enabled`/`tools_max_rounds` are callables so a command run while
    the turn streams (`/tools on|off`, an immediate TUI command) is honoured
    on the next round exactly as it was before the extraction;
    `quiet_tools` is a callable for the same reason (`/render raw` while a
    turn runs must silence the tool panels from the next event on).

    Event ownership: the runtime emits `tool.call` before an approval is
    requested and `approval.resolved` once the approver answers. The
    approver implementation owns `approval.required` -- only it knows
    whether the request actually parks for a human -- and
    `approval_notice()` builds that event.

    This is the *agent* turn only: stateless model turns (the OpenAI
    `/v1/chat/completions` surface) are a translation proxy that never
    reaches the agent loop, so there is no mode switch here.
    """

    no_save: bool = False
    timeout: float = DEFAULT_TIMEOUT
    allow_prompt: bool = True
    quiet_tools: Callable[[], bool] = _always_false
    tools_enabled: Callable[[], bool] = _always_true
    tools_max_rounds: Callable[[], int] = _default_rounds


@dataclass
class TurnRequest:
    """A structured turn request (the HTTP layer maps JSON onto this)."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    session_id: Optional[str] = None
    persist: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)
    system: Optional[str] = None


@dataclass
class TurnResult:
    """The terminal summary of a turn (Phase 2 builds the HTTP shape on it)."""

    id: str = ""
    session_id: str = ""
    status: str = "completed"           # completed | failed | cancelled
    final_message: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    rounds: int = 0
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ApprovalRequest:
    """One tool call asking to run, handed to the host's approver."""

    spec: Any
    args: dict[str, Any]
    fingerprint: str
    detail: str = ""
    allow_prompt: bool = True


@dataclass
class ApprovalDecision:
    """The approver's answer; `action` names how it was reached."""

    run: bool
    note: str = ""
    action: str = "run"                 # run|remember|rule|auto|deny|interrupt
    rule: Optional[ToolRule] = None


@dataclass
class TurnHooks:
    """Everything the runtime needs from its host, built fresh per turn.

    Volatile host state (the client, the `Ui`, the stream sink) is read when
    the hooks are built, so a test or front end that swaps `repl.client`/
    `repl.ui` between turns is honoured.
    """

    client: Any
    store: Any
    session: Session
    settings: TurnSettings
    payload_for: Callable[[list[dict[str, Any]], bool], dict[str, Any]]
    present_round: Callable[[dict[str, Any], bool, Any], GenerationResult]
    approver: Callable[[ApprovalRequest], ApprovalDecision]
    cancel: Optional[CancelToken] = None
    registry: dict[str, Any] = field(default_factory=dict)
    tool_ctx: Optional[ToolContext] = None
    #: Models the host has already probed for tool support; the runtime warns
    #: once per model, not once per turn. The host owns the set.
    tool_support_checked: Optional[set[str]] = None
    execute_tools: Optional[Callable[..., bool]] = None
    on_event: Optional[Callable[[RuntimeEvent], None]] = None
    on_warning: Optional[Callable[[str], None]] = None
    #: Called once per turn, right after `turn.started` (and the tool-support
    #: probe): a host's own turn-start work (the serve controller's context
    #: warning/compaction) emits through `on_event` and stays inside the
    #: turn's event window.
    on_started: Optional[Callable[[], None]] = None
    drain_boundary: Optional[Callable[[], None]] = None
    has_queued: Optional[Callable[[], bool]] = None
    set_tool_round: Optional[Callable[[int], None]] = None
    set_state: Optional[Callable[[str, str], None]] = None
    notify: Optional[Callable[..., None]] = None


class AgentRuntime:
    """One turn's loop and tool execution for a host-supplied set of hooks."""

    def __init__(self, hooks: TurnHooks) -> None:
        self.hooks = hooks
        self.tool_round = 0
        self.runner = ToolRunner()

    # ── plumbing ────────────────────────────────────────────────────────────

    def _emit(self, event_type: str, **data: Any) -> None:
        if self.hooks.on_event is not None:
            self.hooks.on_event(RuntimeEvent(event_type, data))

    def _warn(self, message: str) -> None:
        if self.hooks.on_warning is not None:
            self.hooks.on_warning(message)
        self._emit(TURN_WARNING, message=message)

    def _notify(self, event: str, **data: Any) -> None:
        if self.hooks.notify is not None:
            self.hooks.notify(event, **data)

    def _set_tool_round(self, round_number: int) -> None:
        self.tool_round = round_number
        if self.hooks.set_tool_round is not None:
            self.hooks.set_tool_round(round_number)

    def _cancelled(self) -> bool:
        return self.hooks.cancel is not None and self.hooks.cancel.cancelled

    def check_tool_support(self) -> None:
        """Warn (once per host, per model) when the model cannot call tools.

        The "seen" set lives in `TurnHooks.tool_support_checked` so the
        probe runs once per model across turns; the event is emitted after
        `turn.started` so the ordering contract holds.
        """
        hooks = self.hooks
        if not hooks.settings.tools_enabled():
            return
        model = hooks.session.model
        seen = hooks.tool_support_checked
        if seen is not None and model in seen:
            return
        try:
            capabilities = hooks.client.capabilities(model)
        except Exception:  # capability probing must never break a turn
            return
        if seen is not None:
            seen.add(model)
        if capabilities and "tools" not in capabilities:
            self._warn(f"{model} does not advertise tool support — "
                       "tool calls may not work")

    # ── the turn loop ───────────────────────────────────────────────────────

    def run(self) -> GenerationResult:
        """Run the multi-round turn; the host renders each round via hooks."""
        hooks = self.hooks
        session = hooks.session
        settings = hooks.settings
        self._set_tool_round(0)
        self._emit(TURN_STARTED, model=session.model, session_id=session.id)
        self.check_tool_support()
        if hooks.on_started is not None:
            hooks.on_started()
        result = GenerationResult()
        tool_rounds = 0
        while True:
            if hooks.drain_boundary is not None:
                hooks.drain_boundary()
            if self._cancelled():
                result.cancelled = True
                break
            with_tools = bool(settings.tools_enabled()) \
                and tool_rounds < settings.tools_max_rounds()
            payload = hooks.payload_for(
                session.message_payloads(with_tools=with_tools), with_tools)
            try:
                result = hooks.present_round(payload, bool(tool_rounds), hooks.cancel)
            except OllamaError as exc:
                result = GenerationResult(error=str(exc))
            self._emit(MESSAGE_COMPLETED, round=tool_rounds,
                       content=result.content, thinking=result.thinking or None,
                       tool_calls=len(result.tool_calls),
                       cancelled=True if result.cancelled else None,
                       error=result.error or None)
            if result.content or result.tool_calls:
                session.add_message("assistant", result.content,
                                    thinking=result.thinking or None,
                                    cancelled=True if result.cancelled else None,
                                    tool_calls=result.tool_calls or None)
            elif result.cancelled:
                session.add_message("assistant", "", cancelled=True)
            if result.stats:
                session.record_turn(result.stats, count_turn=(tool_rounds == 0))
                session.stats["last_prompt_tokens"] = result.stats.get("prompt_tokens")
                session.stats["last_completion_tokens"] = result.stats.get("completion_tokens")
            if not settings.no_save:
                hooks.store.save(session)
            if result.error or result.cancelled or not settings.tools_enabled():
                break
            if not result.tool_calls:
                # No tools asked for: only keep going when the user typed
                # something while this answer streamed (delivered next round).
                if hooks.has_queued is not None and hooks.has_queued():
                    continue
                break
            tool_rounds += 1
            if tool_rounds > settings.tools_max_rounds():
                self._warn(f"tool-call limit reached "
                           f"({settings.tools_max_rounds()}) — not running more tools")
                for call in result.tool_calls:
                    function = (call.get("function") or {}) if isinstance(call, dict) else {}
                    session.add_message("tool", "not executed: tool-call limit reached",
                                        tool_name=str(function.get("name") or "?"))
                if not settings.no_save:
                    hooks.store.save(session)
                break
            self._set_tool_round(tool_rounds)
            if hooks.execute_tools is not None:
                interrupted = hooks.execute_tools(
                    result.tool_calls, quiet=settings.quiet_tools(),
                    allow_prompt=settings.allow_prompt)
            else:
                interrupted = self.execute_tool_calls(
                    result.tool_calls, quiet=settings.quiet_tools(),
                    allow_prompt=settings.allow_prompt)
            if interrupted:
                self._warn("tool execution interrupted; stopping this turn")
                break
        self._finish(result, tool_rounds)
        return result

    def _finish(self, result: GenerationResult, rounds: int) -> None:
        if result.error:
            self._emit(TURN_ERROR, message=result.error, rounds=rounds)
        elif result.cancelled:
            reason = getattr(self.hooks.cancel, "reason", "user") or "user"
            self._emit(TURN_CANCELLED, reason=reason, rounds=rounds)
        else:
            self._emit(TURN_COMPLETED, content=result.content,
                       thinking=result.thinking or None,
                       tool_calls=len(result.tool_calls), rounds=rounds,
                       stats=result.stats or None)

    def to_turn_result(self, result: GenerationResult, *, session_id: str = "",
                       turn_id: str = "") -> TurnResult:
        """Map a finished `GenerationResult` onto the structured summary."""
        status = ("failed" if result.error else
                  "cancelled" if result.cancelled else "completed")
        return TurnResult(id=turn_id, session_id=session_id, status=status,
                          final_message={"role": "assistant",
                                         "content": result.content,
                                         "thinking": result.thinking or None,
                                         "tool_calls": result.tool_calls or None},
                          usage=dict(result.stats or {}),
                          rounds=self.tool_round, error=result.error)

    # ── tool execution ──────────────────────────────────────────────────────

    def execute_tool_calls(self, calls: list[dict[str, Any]], *,
                           quiet: bool = False, allow_prompt: bool = True) -> bool:
        """Run the model's tool calls; returns True when interrupted."""
        hooks = self.hooks
        session = hooks.session
        disabled = set(session.tools_disabled)
        available = sorted(set(hooks.registry) - disabled)
        interrupted = False
        for call in calls:
            function = (call.get("function") or {}) if isinstance(call, dict) else {}
            name = str(function.get("name") or "?")
            if interrupted or self._cancelled():
                # A tool already running is never killed by this path alone:
                # it keeps running and its result is still recorded, the model
                # is just not asked to react to it after a stop.
                interrupted = True
                session.add_message("tool", "interrupted by user", tool_name=name)
                self._emit(TOOL_RESULT, name=name, ok=False,
                           text="interrupted by user", cancelled=True)
                continue
            args = normalize_arguments(function.get("arguments"))
            if args is None:
                text = ("invalid tool arguments (not a JSON object): "
                        + str(function.get("arguments"))[:200])
                session.add_message("tool", text, tool_name=name)
                self._emit(TOOL_RESULT, name=name, ok=False, text=text)
                continue
            spec = hooks.registry.get(name) if name not in disabled else None
            alias_note = ""
            if spec is None:
                alias = resolve_tool_alias(name)
                if alias != name and alias not in disabled:
                    spec = hooks.registry.get(alias)
                    if spec is not None:
                        alias_note = f"(note: {name!r} is called {alias!r} here)\n"
                        name = alias
            if spec is None:
                text = f"unknown or disabled tool: {name}"
                if available:
                    text += "; available: " + ", ".join(available)
                session.add_message("tool", text, tool_name=name)
                self._emit(TOOL_RESULT, name=name, ok=False, text=text)
                continue
            fingerprint = call_fingerprint(name, args)
            detail = call_detail(name, args)
            self._emit(TOOL_CALL, name=name, args=args, detail=detail,
                       round=self.tool_round)
            request = ApprovalRequest(spec=spec, args=args, fingerprint=fingerprint,
                                      detail=detail, allow_prompt=allow_prompt)
            try:
                decision = hooks.approver(request)
            except KeyboardInterrupt:
                decision = ApprovalDecision(run=False, note="interrupted by user",
                                            action="interrupt")
                interrupted = True
            self._emit(APPROVAL_RESOLVED, name=name, action=decision.action,
                       note=decision.note, detail=detail)
            if not decision.run:
                text = alias_note + f"tool not run: {decision.note}"
                session.add_message("tool", text, tool_name=name)
                self._emit(TOOL_RESULT, name=name, ok=False, text=text,
                           denied=True, detail=detail)
                continue
            if hooks.set_state is not None:
                hooks.set_state("tool", detail or name)
            self._notify("tool_start", name=name, detail=detail)
            started = time.monotonic()
            try:
                outcome = self.runner.run(spec, args, hooks.tool_ctx)
            except KeyboardInterrupt:
                text = "interrupted by user"
                session.add_message("tool", text, tool_name=name)
                self._emit(TOOL_RESULT, name=name, ok=False, text=text,
                           cancelled=True, detail=detail)
                interrupted = True
                continue
            duration = outcome.duration if outcome.duration \
                else time.monotonic() - started
            self._notify("tool_done", name=name, ok=outcome.ok, duration=duration)
            session.add_message("tool", alias_note + outcome.text, tool_name=name,
                                tool_note=outcome.title_note or None)
            self._emit(TOOL_RESULT, name=name, ok=outcome.ok,
                       text=alias_note + outcome.text, duration=duration,
                       note=outcome.title_note or None, detail=detail,
                       cancelled=True if outcome.cancelled else None)
        if not hooks.settings.no_save:
            hooks.store.save(session)
        return interrupted


def approval_notice(request: ApprovalRequest, *, reason: str = "",
                    suggestions: Optional[list[Any]] = None,
                    approval_id: Optional[str] = None) -> RuntimeEvent:
    """The `approval.required` event for a call that is being asked about.

    Kept here so the console prompt, the TUI modal and the API approval all
    describe a pending call the same way; the API passes its own
    `approval_id` so a client can resolve the exact parked decision.
    """
    data: dict[str, Any] = {
        "name": request.spec.name,
        "args": request.args,
        "detail": request.detail,
        "fingerprint": request.fingerprint,
        "reason": reason,
        "suggestions": [item.to_dict() if hasattr(item, "to_dict") else item
                        for item in (suggestions or [])],
    }
    if approval_id:
        data["approval_id"] = approval_id
    return RuntimeEvent(APPROVAL_REQUIRED, data)


# ── payload + reports the server also needs ─────────────────────────────────


def build_payload(session: Session, messages: list[dict[str, Any]], *,
                  stream: bool = True, tools: Optional[list[dict[str, Any]]] = None,
                  system: Optional[str] = None, think: Optional[Any] = None,
                  keepalive: Optional[str] = None,
                  gpu_layers: Optional[int] = None) -> dict[str, Any]:
    """Assemble one `/api/chat` request from a session and its messages."""
    messages = list(messages)
    if system and (not messages or messages[0].get("role") != "system"):
        messages.insert(0, {"role": "system", "content": system})
    options = dict(session.parameters or {})
    if gpu_layers is not None:
        options["num_gpu"] = gpu_layers
    return build_chat_payload(session.model, messages, stream=stream,
                              fmt=session.format, think=think,
                              keepalive=keepalive or session.keepalive,
                              options=options, tools=tools)


def context_used(session: Session) -> Optional[int]:
    """Tokens the next request carries: last prompt plus the answer that followed."""
    prompt = session.stats.get("last_prompt_tokens")
    completion = session.stats.get("last_completion_tokens")
    if prompt is None and completion is None:
        return None
    return (prompt or 0) + (completion or 0)


def model_show(client: Any, session: Session, cache: dict[Any, dict[str, Any]],
               model: Optional[str] = None) -> dict[str, Any]:
    """Raw `/api/show` for a model, cached per (host, model) for the run."""
    name = model or session.model
    key = (client.base, name)
    if key not in cache:
        try:
            cache[key] = client.show(name)
        except OllamaError:
            cache[key] = {}
    return cache[key]


def resident_context_length(client: Any, session: Session) -> Optional[int]:
    """The window the loaded runner actually uses, from `/api/ps`."""
    model = session.model
    if not model:
        return None
    try:
        entries = client.ps()
    except OllamaError:
        return None
    wanted = canonical_model_name(model)
    for entry in entries:
        name = str(entry.get("name") or entry.get("model") or "")
        if canonical_model_name(name) != wanted:
            continue
        ctx = entry.get("context_length")
        if isinstance(ctx, int) and ctx > 0:
            return ctx
    return None


def context_length(client: Any, session: Session,
                   show_cache: dict[Any, dict[str, Any]],
                   totals: dict[Any, Optional[int]],
                   provisional: set[Any], *, refresh: bool = False) -> Optional[int]:
    """The effective window: session `num_ctx`, loaded runner, or model default."""
    key = (session.model, session.parameters.get("num_ctx"))
    if refresh or key not in totals:
        if refresh:
            show_cache.pop((client.base, session.model), None)
        total = model_context_length(model_show(client, session, show_cache,
                                                session.model),
                                     session.parameters)
        pending = False
        if not isinstance(session.parameters.get("num_ctx"), int):
            resident = resident_context_length(client, session)
            if resident is not None:
                total = resident
            else:
                pending = True
        totals[key] = total
        if pending:
            provisional.add(key)
        else:
            provisional.discard(key)
    elif key in provisional:
        resident = resident_context_length(client, session)
        if resident is not None:
            totals[key] = resident
            provisional.discard(key)
    return totals[key]


def context_report(client: Any, session: Session,
                   show_cache: dict[Any, dict[str, Any]],
                   totals: dict[Any, Optional[int]],
                   provisional: set[Any]) -> dict[str, Any]:
    """Plain data for the `context_status` tool (window + usage)."""
    parameters = session.parameters or {}
    num_ctx = parameters.get("num_ctx")
    key = (session.model, num_ctx)
    window = context_length(client, session, show_cache, totals, provisional)
    model_max = model_context_length(model_show(client, session, show_cache,
                                                session.model), {})
    pending = key in provisional
    if window is None:
        source = "unknown"
    elif isinstance(num_ctx, int) and window == num_ctx:
        source = "num_ctx"
    elif pending:
        source = "model default (runner not loaded yet)"
    elif model_max is not None and window == model_max:
        source = "model default"
    else:
        source = "loaded runner"
    roles: dict[str, int] = {}
    for message in session.messages:
        role = str(message.get("role") or "?")
        roles[role] = roles.get(role, 0) + 1
    return {
        "model": session.model,
        "base": client.base,
        "remote": session.remote,
        "window": window,
        "source": source,
        "model_max": model_max,
        "num_ctx": num_ctx,
        "prompt": session.stats.get("last_prompt_tokens"),
        "completion": session.stats.get("last_completion_tokens"),
        "used": context_used(session),
        "messages": {"total": len(session.messages), **roles},
    }


def session_report(*, session: Session, show: dict[str, Any], base: str,
                   cwd: str, keepalive: Optional[str],
                   gpu_layers: str, thinking: str, render: str,
                   tools_enabled: bool, auto_approve: bool, tools_policy: str,
                   tool_round: int, tools_max_rounds: int) -> dict[str, Any]:
    """Plain data for the `session_info` tool (identity, settings, tools)."""
    details = show.get("details") or {}
    if not tools_enabled:
        policy = "off"
    elif auto_approve:
        policy = "auto (all calls allowed)"
    else:
        policy = tools_policy
    return {
        "session": {
            "id": session.id,
            "title": session.display_title(),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "messages": len(session.messages),
            "turns": session.stats.get("turns", 0),
        },
        "model": session.model,
        "details": {
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "family": details.get("family"),
            "capabilities": list(show.get("capabilities") or []),
        },
        "base": base,
        "remote": session.remote,
        "cwd": cwd,
        "parameters": dict(session.parameters or {}),
        "keepalive": keepalive,
        "gpu_layers": gpu_layers,
        "thinking": thinking,
        "render": render,
        "system_prompt": ("off" if session.system == "" else
                          "custom" if session.system else "built-in default"),
        "tools": {
            "enabled": tools_enabled,
            "policy": policy,
            "round": tool_round,
            "max_rounds": tools_max_rounds,
            "disabled": sorted(session.tools_disabled or []),
        },
    }
