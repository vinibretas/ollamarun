"""Native agent endpoints: full turns, events, approvals and sessions.

These are ollamarun's own API surface (not OpenAI-shaped): a client starts a
complete agent turn and watches it instead of driving the tool loop itself.
The transport is generic (`server.py`) and the agent behavior is the
*existing* runtime: a headless `Repl` supplies the payload, registry and
policy, `AgentRuntime` runs the loop, and `TurnManager` owns the turn
record, the event envelope, replay and cancellation. This module adds only
what a server needs:

* per-turn HTTP state (the turn's client, cancel token, approval mode),
* interactive approvals parked on the API -- `approval.required` carries an
  `approval_id`, `POST /v1/agent/approvals/{id}` resolves it (the first
  decision wins; a timeout is a denial, never an approval),
* session routes over the existing `SessionStore` (a session with an
  active turn cannot be edited or deleted: 409 `session_busy`),
* the serve policy boundary: the request's `settings.tools` and
  `settings.approval` decide behavior, the configured tool policy is never
  `auto` here, and approvals stay runtime-owned decisions.

Disconnects are unsubscribes: a dropped event stream leaves the turn
running (cancellation stays explicit) unless the turn asked for
`cancel_on_disconnect`. Durability: turn records live in memory -- the
session JSON is the durable record, so a restart loses live turns; the turn
status endpoint answers 404 for one that never existed and 410 for one whose
record was swept.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .artifacts import ArtifactStore, require_vision, resolve_content_parts
from .client import OllamaError
from .events import TERMINAL_EVENTS, TURN_WARNING, RuntimeEvent
from .helpers import CancelToken, now_iso
from .openai_api import (classify_backend_error, model_listed,
                         translate_error, usage_from)
from .renderer import GenerationResult, HeadlessRenderer
from .repl import Repl
from .runtime import (AgentRuntime, ApprovalDecision, ApprovalRequest,
                      approval_notice)
from .server import ApiError, Request, Response, Router, SseWriter, StreamResponse
from .session import Session, SessionStore
from .tools import (DEFAULT_SHELL_ALLOW, ToolRule, policy_check, suggest_rules)
from .turns import (INTERNAL_ERROR_MESSAGE, TERMINAL_STATUSES, ManagerClosed,
                    SessionBusy, TooManyTurns, TurnExpired, TurnManager,
                    UnknownTurn)

logger = logging.getLogger("ollamarun.agent_api")

#: A pending approval older than this (resolved) is dropped from the registry.
_APPROVAL_RETENTION = 3600.0

_TURN_FIELDS = frozenset({"session_id", "model", "messages", "persist",
                          "settings"})
_SETTINGS_FIELDS = frozenset({"tools", "max_rounds", "approval", "system",
                              "parameters", "cancel_on_disconnect"})
_MESSAGE_ROLES = frozenset({"user", "assistant"})
_APPROVAL_ACTIONS = frozenset({"run", "remember", "rule", "auto", "deny"})

#: Fields `PATCH /v1/sessions/{id}` accepts.
_SESSION_PATCH_FIELDS = frozenset({"title", "model", "system", "tags"})


@dataclass
class AgentTurnRequest:
    """A validated `POST /v1/agent/turns` body."""

    session_id: Optional[str] = None
    model: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    persist: bool = True
    tools: bool = True
    max_rounds: int = 8
    approval: str = "policy"            # "policy" | "interactive"
    system: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    cancel_on_disconnect: bool = False


def _invalid(message: str, *, param: Optional[str] = None,
             code: Optional[str] = None) -> ApiError:
    return ApiError(400, message, param=param, code=code)


#: Cancellation notes for an approval resolved by cancelling a turn, keyed
#: by the `CancelToken` reason that caused it.
_CANCEL_NOTES = {
    "user": "cancelled by user",
    "timeout": "cancelled (turn timeout)",
    "disconnect": "cancelled (client disconnected)",
    "shutdown": "cancelled (server shutting down)",
}


def _cancel_note(reason: str) -> str:
    reason = str(reason or "user")
    return _CANCEL_NOTES.get(reason, f"cancelled ({reason})")


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise _invalid(f"{name} must be a boolean", param=name)
    return value


def _optional_string(value: Any, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise _invalid(f"{name} must be a string", param=name)
    return value


def _valid_option(value: Any) -> bool:
    if isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(isinstance(item, (bool, int, float, str)) for item in value)
    return False


def parse_agent_turn(body: Any, *, default_max_rounds: int = 8
                     ) -> AgentTurnRequest:
    """Validate one agent-turn body; raises `ApiError(400)`."""
    if not isinstance(body, dict):
        raise _invalid("request body must be a JSON object")
    unknown = set(body) - _TURN_FIELDS
    if unknown:
        raise _invalid(f"unknown field: {sorted(unknown)[0]}",
                       param=sorted(unknown)[0], code="unknown_field")

    session_id = body.get("session_id")
    if session_id is not None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise _invalid("session_id must be a non-empty string",
                           param="session_id")
        session_id = session_id.strip()
    model = _optional_string(body.get("model"), "model").strip()

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _invalid("messages must be a non-empty array", param="messages")
    parsed_messages: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise _invalid(f"messages[{index}] must be an object",
                           param=f"messages[{index}]")
        unknown_fields = set(message) - {"role", "content"}
        if unknown_fields:
            raise _invalid(f"unknown message field: {sorted(unknown_fields)[0]}",
                           param=f"messages[{index}]", code="unknown_field")
        role = message.get("role")
        if role not in _MESSAGE_ROLES:
            raise _invalid('message role must be "user" or "assistant" '
                           "(the harness owns tool messages)",
                           param=f"messages[{index}].role")
        content = message.get("content")
        if isinstance(content, str):
            if not content.strip():
                raise _invalid("message content must be a non-empty string "
                               "or an array of content parts",
                               param=f"messages[{index}].content")
        elif isinstance(content, list):
            if not content:
                raise _invalid("message content must be a non-empty string "
                               "or a non-empty array of content parts",
                               param=f"messages[{index}].content")
            for part_index, part in enumerate(content):
                if not isinstance(part, dict):
                    raise _invalid(
                        "content parts must be objects",
                        param=f"messages[{index}].content[{part_index}]")
        else:
            raise _invalid("message content must be a non-empty string or an "
                           "array of content parts",
                           param=f"messages[{index}].content")
        parsed_messages.append({"role": str(role), "content": content})

    persist = body.get("persist", True)
    _require_bool(persist, "persist")

    settings = body.get("settings")
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise _invalid("settings must be an object", param="settings")
    unknown = set(settings) - _SETTINGS_FIELDS
    if unknown:
        raise _invalid(f"unknown settings field: {sorted(unknown)[0]}",
                       param=f"settings.{sorted(unknown)[0]}",
                       code="unknown_field")

    tools = settings.get("tools", True)
    _require_bool(tools, "settings.tools")
    max_rounds = settings.get("max_rounds", default_max_rounds)
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) \
            or not 1 <= max_rounds <= 100:
        raise _invalid("settings.max_rounds must be an integer between 1 and 100",
                       param="settings.max_rounds")
    approval = settings.get("approval", "policy")
    if approval not in ("policy", "interactive"):
        raise _invalid('settings.approval must be "policy" or "interactive"',
                       param="settings.approval", code="unsupported_value")
    system = settings.get("system")
    if system is not None and not isinstance(system, str):
        raise _invalid("settings.system must be a string or null",
                       param="settings.system")
    parameters = settings.get("parameters", {})
    if not isinstance(parameters, dict):
        raise _invalid("settings.parameters must be an object",
                       param="settings.parameters")
    for key, value in parameters.items():
        if not isinstance(key, str) or not key:
            raise _invalid("settings.parameters keys must be non-empty strings",
                           param="settings.parameters")
        if not _valid_option(value):
            raise _invalid(
                f"settings.parameters.{key} must be a scalar or a list of scalars",
                param=f"settings.parameters.{key}")
    cancel_on_disconnect = settings.get("cancel_on_disconnect", False)
    _require_bool(cancel_on_disconnect, "settings.cancel_on_disconnect")

    return AgentTurnRequest(
        session_id=session_id, model=model, messages=parsed_messages,
        persist=bool(persist), tools=bool(tools), max_rounds=int(max_rounds),
        approval=str(approval), system=system,
        parameters=dict(parameters),
        cancel_on_disconnect=bool(cancel_on_disconnect))


@dataclass
class PendingApproval:
    """One parked tool decision, tied to one turn and one exact call."""

    id: str
    turn_id: str
    session_id: str
    name: str
    args: dict[str, Any]
    detail: str
    fingerprint: str
    reason: str
    suggestions: list[dict[str, Any]]
    created_at: str
    resolved_at: Optional[str] = None
    resolved_monotonic: Optional[float] = None
    decision: Optional[ApprovalDecision] = None
    event: threading.Event = field(default_factory=threading.Event)

    @property
    def resolved(self) -> bool:
        return self.decision is not None

    def body(self) -> dict[str, Any]:
        return {
            "id": self.id, "object": "approval", "turn_id": self.turn_id,
            "session_id": self.session_id, "name": self.name,
            "args": self.args, "detail": self.detail,
            "fingerprint": self.fingerprint, "reason": self.reason,
            "suggestions": self.suggestions, "created_at": self.created_at,
            "status": "resolved" if self.resolved else "pending",
            "resolved_at": self.resolved_at,
            "action": self.decision.action if self.decision else None,
            "run": self.decision.run if self.decision else None,
            "note": self.decision.note if self.decision else None,
        }


@dataclass
class AgentTurnState:
    """The HTTP side of one running turn."""

    id: str
    session_id: str
    session: Session
    repl: Any
    client: Any
    token: CancelToken
    approval_mode: str = "policy"
    cancel_on_disconnect: bool = False
    auto_approve: bool = False
    messages_pending: list[dict[str, Any]] = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)


def turn_body(snapshot: Any) -> dict[str, Any]:
    """The `GET /v1/agent/turns/{id}` shape for one snapshot."""
    result: Optional[dict[str, Any]] = None
    if snapshot.result is not None:
        final = snapshot.result.final_message or {}
        stats = dict(snapshot.result.usage or {})
        result = {
            "status": snapshot.result.status,
            "content": None if snapshot.synthetic_terminal else final.get("content"),
            "thinking": None if snapshot.synthetic_terminal else final.get("thinking"),
            "tool_calls": None if snapshot.synthetic_terminal else final.get("tool_calls"),
            "rounds": snapshot.result.rounds,
            "usage": usage_from(stats),
            "stats": stats or None,
            "artifacts": list(snapshot.result.artifacts),
        }
    error = snapshot.error
    if snapshot.synthetic_terminal and error:
        # Defensive: the manager already sanitizes its synthesized terminals
        # (the raw text goes to the --debug log), so this normally changes
        # nothing -- but no snapshot shape may leak internal detail here.
        error = INTERNAL_ERROR_MESSAGE
    return {
        "id": snapshot.id, "object": "turn", "session_id": snapshot.session_id,
        "status": snapshot.status, "created_at": snapshot.created_at,
        "updated_at": snapshot.updated_at, "seq": snapshot.seq,
        "oldest_seq": snapshot.oldest_seq, "latest_seq": snapshot.latest_seq,
        "truncated": snapshot.truncated, "context": snapshot.context,
        "error": error, "result": result,
    }


def session_body(session: Session, *, active_turn: Optional[str] = None
                 ) -> dict[str, Any]:
    """The session metadata shape shared by every session route."""
    return {
        "id": session.id, "object": "session",
        "title": session.display_title(), "model": session.model,
        "host": session.host, "remote": session.remote,
        "created_at": session.created_at, "updated_at": session.updated_at,
        "tags": list(session.tags or []),
        "messages": len(session.messages),
        "turns": (session.stats or {}).get("turns", 0),
        "system": session.system,
        "artifacts": list(session.artifacts),
        "active_turn_id": active_turn,
    }


def _after_seq(request: Request) -> Optional[int]:
    raw = (request.query.get("after") or [""])[0]
    if raw != "":
        try:
            value = int(raw)
        except ValueError:
            raise _invalid("after must be a non-negative integer", param="after")
        if value < 0:
            raise _invalid("after must be a non-negative integer", param="after")
        return value
    last = request.header("Last-Event-ID").strip()
    if last:
        tail = last.rsplit(":", 1)[-1]
        try:
            return int(tail)
        except ValueError:
            raise _invalid("invalid Last-Event-ID header", param="Last-Event-ID")
    return None


class AgentApi:
    """The native endpoints' handlers, owning the turn/approval registries."""

    def __init__(self, *, args: Any, config: dict[str, Any],
                 store: SessionStore, client_factory: Callable[[], Any],
                 manager: Optional[TurnManager] = None,
                 cwd: Optional[str] = None,
                 artifacts: Optional[ArtifactStore] = None):
        self._args = args
        self._config = dict(config or {})
        self._store = store
        self._client_factory = client_factory
        #: The serve artifact store: `file`/image parts resolve through it,
        #: turn tools publish into it, and the artifact routes read from it.
        self._artifacts = artifacts
        self._manager = manager or TurnManager(
            max_active=int(self._config.get("serve_max_active_turns") or 4),
            buffer_events=int(self._config.get("serve_turn_event_buffer")
                              or 10000),
            completed_ttl=float(self._config.get("serve_turn_event_ttl")
                                or 86400))
        #: How long a parked approval may wait for a decision, in seconds;
        #: an explicit 0 disables the timeout (the approval waits until it is
        #: resolved, cancelled or the server shuts down), so `0` must not be
        #: conflated with "unset" the way `or 300` would.
        raw_approval_timeout = self._config.get("serve_approval_timeout")
        self._approval_timeout = 300.0 if raw_approval_timeout is None \
            else float(raw_approval_timeout)
        self._turn_timeout = float(self._config.get("serve_turn_timeout") or 0)
        #: Serve never takes `auto` from the config: the client opts into
        #: broad execution per turn (`approval: "interactive"` + an `auto`
        #: decision), and `--yes` remains the only CLI auto path.
        policy = str(self._config.get("tools_policy") or "allowlist").lower()
        self._tools_policy = policy if policy in {"allowlist", "ask"} \
            else "allowlist"
        self._shell_allow = set(DEFAULT_SHELL_ALLOW) | {
            str(item) for item in (self._config.get("tools_shell_allow") or [])
            if item}
        self._cwd = str(cwd or os.getcwd())
        self._lock = threading.Lock()
        self._turns: dict[str, AgentTurnState] = {}
        self._approvals: dict[str, PendingApproval] = {}

    # ── helpers ─────────────────────────────────────────────────────────────

    @property
    def manager(self) -> TurnManager:
        return self._manager

    def _default_max_rounds(self) -> int:
        try:
            return max(1, int(self._config.get("tools_max_rounds") or 8))
        except (TypeError, ValueError):
            return 8

    def _session(self, ref: str) -> Session:
        session = self._store.resolve(ref)
        if session is None:
            raise ApiError(404, f"session {ref!r} not found",
                           code="session_not_found")
        return session

    def _ensure_model(self, client: Any, model: str) -> None:
        try:
            entries = client.tags()
        except OllamaError as exc:
            raise translate_error(exc)
        if not model_listed(model, entries):
            raise ApiError(404, f"model {model!r} not found",
                           code="model_not_found", param="model")

    def _session_rules(self, session: Session) -> list[ToolRule]:
        rules: list[ToolRule] = []
        for entry in session.tool_rules or []:
            rule = ToolRule.from_dict(entry)
            if rule is not None:
                rules.append(rule)
        return rules

    def _turn_state(self, turn_id: str) -> Optional[AgentTurnState]:
        with self._lock:
            return self._turns.get(turn_id)

    def _prune(self) -> None:
        """Drop finished turn states and long-resolved approvals."""
        now = time.monotonic()
        with self._lock:
            for turn_id, state in list(self._turns.items()):
                if state.done.is_set():
                    self._turns.pop(turn_id, None)
            for approval_id, pending in list(self._approvals.items()):
                if pending.resolved_monotonic is not None \
                        and now - pending.resolved_monotonic > _APPROVAL_RETENTION:
                    self._approvals.pop(approval_id, None)

    # ── turns ───────────────────────────────────────────────────────────────

    def start_turn(self, request: Request,
                   params: dict[str, str]) -> Response:
        spec = parse_agent_turn(
            request.json, default_max_rounds=self._default_max_rounds())
        self._prune()
        client = self._client_factory()

        if spec.session_id:
            session = self._session(spec.session_id)
            active = self._manager.active_for(session.id)
            if active:
                raise _session_busy(session.id, active)
        else:
            session = Session.create(
                spec.model or str(self._config.get("model") or ""),
                getattr(client, "base", ""))
        if spec.model:
            session.model = spec.model
        if not session.model:
            session.model = str(self._config.get("model") or "")
        if not session.model:
            raise _invalid(
                "model is required (none in the request, the session or the "
                "serve config)", param="model")
        self._ensure_model(client, session.model)
        if spec.system is not None:
            session.system = spec.system
        if spec.parameters:
            session.parameters = {**(session.parameters or {}),
                                  **spec.parameters}

        # Resolve every message's content parts before the turn is accepted:
        # a missing artifact id, an oversize image and a text/file mapping are
        # all decided here, so the client gets a structured 4xx synchronously.
        resolved_messages: list[dict[str, Any]] = []
        has_images = False
        for index, message in enumerate(spec.messages):
            resolved = resolve_content_parts(
                message.get("content"), where=f"messages[{index}].content",
                store=self._artifacts)
            payload: dict[str, Any] = {"role": message["role"],
                                       "content": resolved.text}
            if resolved.images:
                has_images = True
                payload["images"] = [
                    ({"b64": image.b64, "artifact_id": image.artifact_id}
                     if image.artifact_id else {"b64": image.b64})
                    for image in resolved.images]
            if resolved.artifacts:
                payload["artifacts"] = [ref.to_dict()
                                        for ref in resolved.artifacts]
            resolved_messages.append(payload)
        if has_images:
            require_vision(client.capabilities(session.model))

        turn_id = f"turn_{uuid.uuid4().hex[:12]}"
        token = CancelToken()
        repl = self._make_repl(session, client, spec, token, turn_id)
        state = AgentTurnState(
            id=turn_id, session_id=session.id, session=session, repl=repl,
            client=client, token=token, approval_mode=spec.approval,
            cancel_on_disconnect=spec.cancel_on_disconnect,
            messages_pending=resolved_messages)
        with self._lock:
            self._turns[turn_id] = state

        def work(sink: Callable[[RuntimeEvent], None],
                 cancel: CancelToken) -> Any:
            return self._run_turn(state, spec, sink)

        try:
            self._manager.submit(
                session_id=session.id, work=work, turn_id=turn_id,
                cancel=token, context=lambda: self._context_block(repl))
        except SessionBusy as exc:
            with self._lock:
                self._turns.pop(turn_id, None)
            raise _session_busy(exc.session_id, exc.active_turn_id)
        except TooManyTurns as exc:
            with self._lock:
                self._turns.pop(turn_id, None)
            raise ApiError(429, "too many active turns",
                           type_="rate_limit_error", code="rate_limit_error",
                           extra={"limit": exc.limit})
        except ManagerClosed:
            with self._lock:
                self._turns.pop(turn_id, None)
            raise ApiError(503, "the server is shutting down",
                           type_="api_error", code="server_shutting_down")
        self._start_watchdog(state)
        snapshot = self._manager.get(turn_id)
        body = {
            "id": turn_id, "object": "turn", "session_id": session.id,
            "status": snapshot.status if snapshot else "queued",
            "created_at": snapshot.created_at if snapshot else now_iso(),
            "status_url": f"/v1/agent/turns/{turn_id}",
            "events_url": f"/v1/agent/turns/{turn_id}/events",
            "cancel_url": f"/v1/agent/turns/{turn_id}/cancel",
        }
        return Response.json(body, status=202)

    def _make_repl(self, session: Session, client: Any,
                   spec: AgentTurnRequest, token: CancelToken,
                   turn_id: str = "") -> Repl:
        artifacts = (self._artifacts.for_turn(turn_id=turn_id,
                                              session_id=session.id)
                     if self._artifacts is not None else None)
        repl = Repl(self._args, self._config, client, self._store, session,
                    artifacts=artifacts)
        repl.headless = True
        repl.tools_enabled = bool(spec.tools) and bool(
            self._config.get("tools", True))
        repl.tools_max_rounds = spec.max_rounds
        repl.no_save = not spec.persist
        # The turn's own token must reach every tool (subprocess cancellation)
        # and the runtime's cancel check; Repl built its own for the CLI.
        repl.cancel_token = token
        repl.tool_ctx.cancel = token
        return repl

    def _context_block(self, repl: Repl) -> dict[str, Any]:
        try:
            report = repl.context_report()
        except Exception:
            logger.debug("context report failed", exc_info=True)
            return {}
        return {"used": report.get("used"), "total": report.get("window"),
                "source": report.get("source")}

    def _start_watchdog(self, state: AgentTurnState) -> None:
        if self._turn_timeout <= 0:
            return

        def watch() -> None:
            if state.done.wait(self._turn_timeout):
                return
            logger.debug("turn %s exceeded serve_turn_timeout", state.id)
            self._cancel_turn(state, "timeout")

        threading.Thread(target=watch, daemon=True,
                         name=f"turn-timeout-{state.id}").start()

    def _presenter(self, state: AgentTurnState, repl: Repl,
                   sink: Callable[[RuntimeEvent], None]
                   ) -> Callable[..., GenerationResult]:
        def present(payload: dict[str, Any], after_tools: bool,
                    cancel: Optional[CancelToken]) -> GenerationResult:
            renderer = HeadlessRenderer(on_event=sink, after_tools=after_tools)
            try:
                return renderer.stream(
                    state.client.chat(payload, timeout=repl.timeout,
                                      cancel=cancel),
                    cancel=cancel)
            except OllamaError as exc:
                logger.debug("turn %s backend error: %s", state.id, exc)
                _, message = classify_backend_error(
                    str(exc), model=state.session.model)
                return GenerationResult(error=message)
        return present

    def _run_turn(self, state: AgentTurnState, spec: AgentTurnRequest,
                  sink: Callable[[RuntimeEvent], None]) -> Any:
        session = state.session
        repl = state.repl
        try:
            for message in state.messages_pending:
                session.add_message(message["role"], message["content"],
                                    images=message.get("images") or None,
                                    artifacts=message.get("artifacts") or None)
                if message.get("artifacts"):
                    session.add_artifacts(message["artifacts"])
            state.messages_pending = []
            if spec.persist:
                self._store.save(session)

            def on_started() -> None:
                warning = repl.context_warning()
                if warning:
                    sink(RuntimeEvent(TURN_WARNING, {"message": warning}))
                if self._config.get("auto_compact") == "auto":
                    summary = repl.maybe_auto_compact()
                    if summary:
                        sink(RuntimeEvent(TURN_WARNING, {
                            "message": "context compacted into a summary"}))

            hooks = repl._turn_hooks(allow_prompt=True, quiet_tools=True)
            hooks.cancel = state.token
            hooks.on_event = sink
            hooks.on_warning = lambda message: None
            hooks.on_started = on_started
            hooks.approver = lambda request: self._approve(state, sink, request)
            hooks.present_round = self._presenter(state, repl, sink)
            # Use the runtime's own tool execution so the approver/sink above
            # apply to every round (Repl's wrapper would rebuild hooks).
            hooks.execute_tools = None
            runtime = AgentRuntime(hooks)
            result = runtime.run()
            return runtime.to_turn_result(result, session_id=session.id,
                                          turn_id=state.id)
        finally:
            state.done.set()

    def turn_status(self, request: Request,
                    params: dict[str, str]) -> Response:
        turn_id = params.get("id", "")
        snapshot = self._manager.get(turn_id)
        if snapshot is None:
            if self._manager.expired(turn_id) is not None:
                raise ApiError(410, f"turn {turn_id!r} expired",
                               code="turn_expired")
            raise ApiError(404, f"turn {turn_id!r} not found",
                           code="turn_not_found")
        return Response.json(turn_body(snapshot))

    def turn_events(self, request: Request,
                    params: dict[str, str]) -> StreamResponse:
        turn_id = params.get("id", "")
        after = _after_seq(request)
        try:
            subscription = self._manager.subscribe(turn_id, after=after)
        except UnknownTurn:
            raise ApiError(404, f"turn {turn_id!r} not found",
                           code="turn_not_found")
        except TurnExpired as exc:
            raise ApiError(410, f"turn {turn_id!r} expired",
                           code="turn_expired", extra={"status": exc.status})
        state = self._turn_state(turn_id)

        def produce(writer: SseWriter) -> None:
            try:
                while True:
                    event = subscription.get(timeout=1.0)
                    if event is None:
                        if subscription.closed or subscription.at_terminal:
                            break
                        continue
                    writer.event(json.dumps(event.to_dict()),
                                 event=event.type, id=event.id)
                    if event.type in TERMINAL_EVENTS:
                        break
            finally:
                subscription.close()

        def on_disconnect() -> None:
            # A disconnect is an unsubscribe; cancellation stays explicit
            # unless the turn asked for `cancel_on_disconnect`.
            subscription.close()
            if state is not None and state.cancel_on_disconnect \
                    and not state.done.is_set():
                self._cancel_turn(state, "disconnect")

        return StreamResponse(produce=produce, on_disconnect=on_disconnect)

    def cancel_turn(self, request: Request,
                    params: dict[str, str]) -> Response:
        turn_id = params.get("id", "")
        snapshot = self._manager.get(turn_id)
        if snapshot is None:
            if self._manager.expired(turn_id) is not None:
                raise ApiError(410, f"turn {turn_id!r} expired",
                               code="turn_expired")
            raise ApiError(404, f"turn {turn_id!r} not found",
                           code="turn_not_found")
        if snapshot.status in TERMINAL_STATUSES:
            raise ApiError(409, f"turn {turn_id!r} already finished",
                           type_="conflict_error", code="turn_finished",
                           extra={"status": snapshot.status})
        state = self._turn_state(turn_id)
        if state is not None:
            self._cancel_turn(state, "user")
        else:  # pragma: no cover - every live turn has state
            self._manager.cancel(turn_id)
        return Response.json({"id": turn_id, "object": "turn",
                              "cancel_requested": True,
                              "status": snapshot.status}, status=202)

    def _cancel_turn(self, state: AgentTurnState, reason: str = "user") -> None:
        state.token.cancel(reason)
        try:
            state.client.abort_stream()
        except Exception:  # pragma: no cover - defensive
            pass
        for pending in self._pending_approvals(state.id):
            self._first_decision(pending, ApprovalDecision(
                run=False, note=_cancel_note(reason), action="cancelled"))

    # ── approvals ───────────────────────────────────────────────────────────

    def _approve(self, state: AgentTurnState,
                 sink: Callable[[RuntimeEvent], None],
                 request: ApprovalRequest) -> ApprovalDecision:
        if state.auto_approve:
            return ApprovalDecision(run=True, note="auto-allow (API)",
                                    action="auto")
        check = policy_check(request.spec, request.args,
                             policy=self._tools_policy,
                             shell_allow=self._shell_allow,
                             fingerprints=list(state.session.tool_allow or []),
                             rules=self._session_rules(state.session),
                             fingerprint=request.fingerprint, cwd=self._cwd)
        if check.run:
            return ApprovalDecision(run=True, note=check.note,
                                    action="rule" if check.rule is not None
                                    else "run")
        if state.approval_mode != "interactive":
            return ApprovalDecision(
                run=False, note=f"not allowed by policy ({check.reason})",
                action="deny")
        suggestions = suggest_rules(request.spec.name, request.args,
                                    cwd=self._cwd)
        pending = self._new_approval(state, request, check.reason,
                                     suggestions)
        sink(approval_notice(request, reason=check.reason,
                             suggestions=suggestions, approval_id=pending.id))
        self._manager.mark_waiting_approval(state.id)
        try:
            return self._wait_for(pending, state.token)
        finally:
            self._manager.mark_waiting_approval(state.id, waiting=False)

    def _new_approval(self, state: AgentTurnState, request: ApprovalRequest,
                      reason: str, suggestions: list[ToolRule]
                      ) -> PendingApproval:
        pending = PendingApproval(
            id=f"apr_{uuid.uuid4().hex[:12]}", turn_id=state.id,
            session_id=state.session_id, name=request.spec.name,
            args=request.args, detail=request.detail,
            fingerprint=request.fingerprint, reason=reason,
            suggestions=[rule.to_dict() for rule in suggestions],
            created_at=now_iso())
        with self._lock:
            self._approvals[pending.id] = pending
        return pending

    def _pending_approvals(self, turn_id: str) -> list[PendingApproval]:
        with self._lock:
            return [pending for pending in self._approvals.values()
                    if pending.turn_id == turn_id and not pending.resolved]

    def _first_decision(self, pending: PendingApproval,
                        decision: ApprovalDecision,
                        side_effect: Optional[Callable[[], None]] = None
                        ) -> bool:
        """Record the first decision for a pending approval; False if beaten."""
        with self._lock:
            if pending.decision is not None:
                return False
            if side_effect is not None:
                side_effect()
            pending.decision = decision
            pending.resolved_at = now_iso()
            pending.resolved_monotonic = time.monotonic()
        pending.event.set()
        return True

    def _wait_for(self, pending: PendingApproval,
                  token: CancelToken) -> ApprovalDecision:
        deadline = (time.monotonic() + self._approval_timeout
                    if self._approval_timeout > 0 else None)
        while not pending.event.wait(0.2):
            if token.cancelled:
                self._first_decision(pending, ApprovalDecision(
                    run=False, note=_cancel_note(token.reason),
                    action="cancelled"))
                break
            if deadline is not None and time.monotonic() >= deadline:
                self._first_decision(pending, ApprovalDecision(
                    run=False, note="approval timed out", action="timeout"))
                break
        with self._lock:
            return pending.decision or ApprovalDecision(
                run=False, note="approval timed out", action="timeout")

    def resolve_approval(self, request: Request,
                         params: dict[str, str]) -> Response:
        body = request.json
        if not isinstance(body, dict):
            raise _invalid("request body must be a JSON object")
        unknown = set(body) - {"action", "rule"}
        if unknown:
            raise _invalid(f"unknown field: {sorted(unknown)[0]}",
                           param=sorted(unknown)[0], code="unknown_field")
        action = body.get("action")
        if action not in _APPROVAL_ACTIONS:
            raise _invalid(
                'action must be "run", "remember", "rule", "auto" or "deny"',
                param="action", code="unsupported_value")
        with self._lock:
            pending = self._approvals.get(params.get("id", ""))
        if pending is None:
            raise ApiError(404, f"approval {params.get('id')!r} not found",
                           code="approval_not_found")
        state = self._turn_state(pending.turn_id)
        decision = ApprovalDecision(run=True, note="confirmed", action="run")
        side_effect: Optional[Callable[[], None]] = None
        if action == "deny":
            decision = ApprovalDecision(run=False, note="denied by the user",
                                        action="deny")
        elif action == "auto":
            decision = ApprovalDecision(
                run=True, note="auto-allow everything (careful)", action="auto")

            def allow_everything() -> None:
                if state is not None:
                    state.auto_approve = True
            side_effect = allow_everything
        elif action == "remember":
            decision = ApprovalDecision(run=True, note="allowed (remembered)",
                                        action="remember")

            def remember_call() -> None:
                if state is None:
                    return
                allowed = list(state.session.tool_allow or [])
                if pending.fingerprint not in allowed:
                    allowed.append(pending.fingerprint)
                state.session.tool_allow = allowed
                state.session.touch()
            side_effect = remember_call
        elif action == "rule":
            rule = ToolRule.from_dict(body.get("rule"))
            if rule is None:
                raise _invalid(
                    "rule must be a {kind, pattern} object covering this call",
                    param="rule")
            decision = ApprovalDecision(run=True, note="allowed (remembered)",
                                        action="rule", rule=rule)

            def add_rule() -> None:
                if state is None:
                    return
                state.session.tool_rules = list(state.session.tool_rules
                                                or []) + [rule.to_dict()]
                state.session.touch()
            side_effect = add_rule
        if not self._first_decision(pending, decision, side_effect):
            raise ApiError(409, "this approval was already resolved",
                           type_="conflict_error", code="already_resolved")
        return Response.json({
            "id": pending.id, "object": "approval", "turn_id": pending.turn_id,
            "action": decision.action, "run": decision.run,
            "note": decision.note, "resolved_at": pending.resolved_at,
        })

    # ── sessions ────────────────────────────────────────────────────────────

    def list_sessions(self, request: Request,
                      params: dict[str, str]) -> Response:
        return Response.json({"object": "list",
                              "data": self._store.entries()})

    def create_session(self, request: Request,
                       params: dict[str, str]) -> Response:
        body = request.json if request.json is not None else {}
        if not isinstance(body, dict):
            raise _invalid("request body must be a JSON object")
        unknown = set(body) - {"title", "model", "system", "tags", "parameters"}
        if unknown:
            raise _invalid(f"unknown field: {sorted(unknown)[0]}",
                           param=sorted(unknown)[0], code="unknown_field")
        title = _optional_string(body.get("title"), "title")
        model = _optional_string(body.get("model"), "model").strip()
        system = body.get("system")
        if system is not None and not isinstance(system, str):
            raise _invalid("system must be a string or null", param="system")
        tags = body.get("tags", [])
        if not isinstance(tags, list) \
                or not all(isinstance(tag, str) for tag in tags):
            raise _invalid("tags must be an array of strings", param="tags")
        parameters = body.get("parameters", {})
        if not isinstance(parameters, dict):
            raise _invalid("parameters must be an object", param="parameters")
        client = self._client_factory()
        if model:
            self._ensure_model(client, model)
        session = Session.create(model, getattr(client, "base", ""),
                                 title=title, tags=list(tags), system=system,
                                 parameters=dict(parameters))
        self._store.save(session)
        return Response.json(session_body(session), status=201)

    def get_session(self, request: Request,
                    params: dict[str, str]) -> Response:
        session = self._session(params.get("id", ""))
        body = session_body(session,
                            active_turn=self._manager.active_for(session.id))
        body.update({
            "history": session.messages,
            "stats": dict(session.stats or {}),
            "parameters": dict(session.parameters or {}),
            "tool_allow": list(session.tool_allow or []),
            "tool_rules": list(session.tool_rules or []),
            "tools_disabled": list(session.tools_disabled or []),
        })
        return Response.json(body)

    def patch_session(self, request: Request,
                      params: dict[str, str]) -> Response:
        session = self._session(params.get("id", ""))
        active = self._manager.active_for(session.id)
        if active:
            raise _session_busy(session.id, active)
        body = request.json
        if not isinstance(body, dict) or not body:
            raise _invalid("request body must be a non-empty JSON object")
        unknown = set(body) - _SESSION_PATCH_FIELDS
        if unknown:
            raise _invalid(f"unknown field: {sorted(unknown)[0]}",
                           param=sorted(unknown)[0], code="unknown_field")
        if "title" in body:
            session.title = _optional_string(body["title"], "title")
        if "model" in body:
            model = _optional_string(body["model"], "model").strip()
            if model:
                self._ensure_model(self._client_factory(), model)
            session.model = model
        if "system" in body:
            system = body["system"]
            if system is not None and not isinstance(system, str):
                raise _invalid("system must be a string or null", param="system")
            session.system = system
        if "tags" in body:
            tags = body["tags"]
            if not isinstance(tags, list) \
                    or not all(isinstance(tag, str) for tag in tags):
                raise _invalid("tags must be an array of strings", param="tags")
            session.tags = list(tags)
        session.touch()
        self._store.save(session)
        return Response.json(session_body(
            session, active_turn=self._manager.active_for(session.id)))

    def delete_session(self, request: Request,
                       params: dict[str, str]) -> Response:
        session = self._session(params.get("id", ""))
        active = self._manager.active_for(session.id)
        if active:
            raise _session_busy(session.id, active)
        try:
            self._store.delete(session.id)
        except Exception as exc:
            logger.debug("session delete failed: %s", exc)
            raise ApiError(500, "the session could not be moved to the trash",
                           type_="api_error", code="delete_failed")
        return Response.json({"id": session.id, "object": "session.deleted",
                              "deleted": True})

    # ── lifecycle ───────────────────────────────────────────────────────────

    def shutdown(self, *, timeout: float = 10.0) -> None:
        """Resolve pending approvals as cancelled, then stop the manager."""
        for pending in self._pending_approvals_all():
            self._first_decision(pending, ApprovalDecision(
                run=False, note=_cancel_note("shutdown"), action="cancelled"))
        self._manager.shutdown(cancel=True, timeout=timeout)

    def _pending_approvals_all(self) -> list[PendingApproval]:
        with self._lock:
            return [pending for pending in self._approvals.values()
                    if not pending.resolved]


def _session_busy(session_id: str, active_turn_id: str) -> ApiError:
    return ApiError(409, f"session {session_id!r} already has an active turn",
                    type_="conflict_error", code="session_busy",
                    extra={"active_turn_id": active_turn_id})


def add_agent_routes(router: Router, *, args: Any, config: dict[str, Any],
                     store: SessionStore, client_factory: Callable[[], Any],
                     manager: Optional[TurnManager] = None,
                     cwd: Optional[str] = None,
                     artifacts: Optional[ArtifactStore] = None) -> AgentApi:
    """Register the native endpoints on `router`; returns the API object."""
    api = AgentApi(args=args, config=config, store=store,
                   client_factory=client_factory, manager=manager, cwd=cwd,
                   artifacts=artifacts)
    router.add("POST", "/v1/agent/turns", api.start_turn)
    router.add("GET", "/v1/agent/turns/{id}", api.turn_status)
    router.add("GET", "/v1/agent/turns/{id}/events", api.turn_events)
    router.add("POST", "/v1/agent/turns/{id}/cancel", api.cancel_turn)
    router.add("POST", "/v1/agent/approvals/{id}", api.resolve_approval)
    router.add("GET", "/v1/sessions", api.list_sessions)
    router.add("POST", "/v1/sessions", api.create_session)
    router.add("GET", "/v1/sessions/{id}", api.get_session)
    router.add("PATCH", "/v1/sessions/{id}", api.patch_session)
    router.add("DELETE", "/v1/sessions/{id}", api.delete_session)
    return api
