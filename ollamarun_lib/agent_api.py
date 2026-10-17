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

import base64
import json
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .artifacts import (ArtifactStore, InputBudget, input_limits,
                        require_vision, resolve_content_parts)
from .client import OllamaError
from .ctxfit import (CtxCalibrationStore, FitRequest, fit_context,
                     machine_state_for)
from .events import (SESSION_FIT, TERMINAL_EVENTS, TURN_WARNING, RuntimeEvent)
from .fits import (FitBusy, FitService, FitsClosed, TERMINAL_FIT_STATUSES,
                   TooManyFits, fit_body)
from .helpers import (CancelToken, canonical_model_name, is_local_host,
                      model_allowed, now_iso, remote_node)
from .interactive import (model_block_count, model_context_length,
                          parse_gpu_layers_spec, resolve_gpu_layers)
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
#: Fields `POST /v1/files` accepts (the JSON upload form; multipart is later).
_UPLOAD_FIELDS = frozenset({"filename", "media_type", "data_base64"})
#: Fit request fields: the sync estimate adds `apply`, the async job `probe`.
_FIT_BASE_FIELDS = frozenset({"model", "session_id", "mode", "cpu_percent",
                              "gpu_layers"})
_FIT_APPLY_FIELDS = _FIT_BASE_FIELDS | {"apply"}
_FIT_JOB_FIELDS = _FIT_APPLY_FIELDS | {"probe"}

_TURN_FIELDS = frozenset({"session_id", "model", "messages", "persist",
                          "settings"})
_SETTINGS_FIELDS = frozenset({"tools", "max_rounds", "approval", "system",
                              "parameters", "num_ctx", "cpu_percent",
                              "gpu_layers", "cancel_on_disconnect"})
_MESSAGE_ROLES = frozenset({"user", "assistant"})
_APPROVAL_ACTIONS = frozenset({"run", "remember", "rule", "auto", "deny"})

#: Fields `PATCH /v1/sessions/{id}` accepts.
_SESSION_PATCH_FIELDS = frozenset({"title", "model", "system", "tags",
                                   "num_ctx", "gpu_layers", "parameters"})


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
    #: Session mutations: an explicit context (`num_ctx` int) or a turn-start
    #: fit (`"max-gpu"`, or `"max-cpu"` with `cpu_percent`), plus a placement
    #: spec. The dedicated fields win over `parameters`.
    num_ctx: Any = None
    cpu_percent: Optional[int] = None
    gpu_layers: Optional[str] = None
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


#: Known Ollama options and their ranges: the OpenAI mapper enforces the
#: same ones where they overlap. An unknown option name is still forwarded
#: (forward compatibility with new Ollama options) but bounded to
#: scalars/lists of scalars — never a way around the limits.
_OPTION_FLOAT_RANGES: dict[str, Any] = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "min_p": (0.0, 1.0),
    "typical_p": (0.0, 1.0),
    "tfs_z": (None, None),
    "repeat_penalty": (None, None),
    "presence_penalty": (-2.0, 2.0),
    "frequency_penalty": (-2.0, 2.0),
    "mirostat_eta": (None, None),
    "mirostat_tau": (None, None),
}
_OPTION_POSITIVE = frozenset({"tfs_z", "repeat_penalty", "mirostat_eta",
                              "mirostat_tau"})
_OPTION_INT_MIN: dict[str, Optional[int]] = {
    "top_k": 0,
    "repeat_last_n": -1,
    "seed": None,
    "num_predict": -1,
    "num_ctx": 1,
    "num_gpu": 0,
    "main_gpu": 0,
    "num_thread": 1,
    "num_batch": 1,
    "num_keep": 0,
}
_OPTION_ENUMS: dict[str, frozenset] = {"mirostat": frozenset({0, 1, 2})}
_OPTION_BOOLS = frozenset({"use_mmap", "use_mlock"})
_OPTION_STRING_OR_LIST = frozenset({"stop"})
#: Bounds for one option value (known and unknown alike).
_OPTION_STRING_MAX = 4096
_OPTION_LIST_MAX = 64
_OPTION_DICT_MAX_BYTES = 64 * 1024


def _validate_option_float(key: str, value: Any, at: str) -> None:
    low, high = _OPTION_FLOAT_RANGES[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"{at} must be a number", param=at,
                       code="unsupported_value")
    number = float(value)
    if not math.isfinite(number):
        raise _invalid(f"{at} must be a finite number", param=at,
                       code="unsupported_value")
    if key in _OPTION_POSITIVE and number <= 0:
        raise _invalid(f"{at} must be > 0", param=at, code="unsupported_value")
    if low is not None and number < low:
        raise _invalid(f"{at} must be >= {low:g}", param=at,
                       code="unsupported_value")
    if high is not None and number > high:
        raise _invalid(f"{at} must be <= {high:g}", param=at,
                       code="unsupported_value")


def _validate_option_int(key: str, value: Any, at: str) -> None:
    minimum = _OPTION_INT_MIN[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(f"{at} must be an integer", param=at,
                       code="unsupported_value")
    if minimum is not None and value < minimum:
        raise _invalid(f"{at} must be >= {minimum}", param=at,
                       code="unsupported_value")


def _validate_option_unknown(value: Any, at: str) -> None:
    if isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _invalid(f"{at} must be a finite number", param=at,
                           code="unsupported_value")
        return
    if isinstance(value, str):
        if len(value) > _OPTION_STRING_MAX:
            raise _invalid(f"{at} is too long ({_OPTION_STRING_MAX} "
                           "characters max)", param=at,
                           code="unsupported_value")
        return
    if isinstance(value, list):
        if len(value) > _OPTION_LIST_MAX:
            raise _invalid(f"{at} has too many items ({_OPTION_LIST_MAX} max)",
                           param=at, code="unsupported_value")
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                raise _invalid(f"{at} items must be scalars", param=at,
                               code="unsupported_value")
            _validate_option_unknown(item, at)
        return
    raise _invalid(f"{at} must be a scalar or a list of scalars", param=at,
                   code="unsupported_value")


def validate_settings_parameters(parameters: dict[str, Any], *,
                                 where: str = "settings.parameters"
                                 ) -> dict[str, Any]:
    """Validate native Ollama options with bounded pass-through.

    Known options get their type and range checked (the same ranges the
    OpenAI mapper enforces where they overlap); an unknown option name is
    forwarded for forward compatibility, but bounded to scalars or lists of
    scalars — strings max 4 KiB, lists max 64 items, the whole object max
    64 KiB. Non-finite numbers are never valid.
    """
    serialized = len(json.dumps(parameters, default=str).encode("utf-8"))
    if serialized > _OPTION_DICT_MAX_BYTES:
        raise _invalid(f"{where} is too large "
                       f"({_OPTION_DICT_MAX_BYTES // 1024} KiB max)",
                       param=where, code="unsupported_value")
    for key, value in parameters.items():
        if not isinstance(key, str) or not key.strip():
            raise _invalid(f"{where} keys must be non-empty strings",
                           param=where, code="unsupported_value")
        at = f"{where}.{key}"
        if key in _OPTION_BOOLS:
            if not isinstance(value, bool):
                raise _invalid(f"{at} must be a boolean", param=at,
                               code="unsupported_value")
        elif key in _OPTION_ENUMS:
            options = _OPTION_ENUMS[key]
            if isinstance(value, bool) or not isinstance(value, int) \
                    or value not in options:
                allowed = "/".join(str(item) for item in sorted(options))
                raise _invalid(f"{at} must be {allowed}", param=at,
                               code="unsupported_value")
        elif key in _OPTION_STRING_OR_LIST:
            items = [value] if isinstance(value, str) else value
            if not isinstance(items, list) \
                    or not all(isinstance(item, str) for item in items):
                raise _invalid(f"{at} must be a string or a list of strings",
                               param=at, code="unsupported_value")
        elif key in _OPTION_FLOAT_RANGES:
            _validate_option_float(key, value, at)
        elif key in _OPTION_INT_MIN:
            _validate_option_int(key, value, at)
        else:
            _validate_option_unknown(value, at)
    return dict(parameters)


def _show_data(client: Any, model: str) -> Optional[dict[str, Any]]:
    """`/api/show` for validation; None when it cannot be read."""
    if not model:
        return None
    try:
        data = client.show(model)
    except (OllamaError, TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _validate_num_ctx(value: Any, *, show_data: Optional[dict[str, Any]],
                      param: str) -> Optional[int]:
    """A positive num_ctx, refused past the model's trained maximum.

    The API is strict where the CLI caps with a warning: silently altering a
    requested value is worse than naming the ceiling.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _invalid("num_ctx must be a positive integer or null", param=param,
                       code="unsupported_value")
    maximum = model_context_length(show_data) if show_data else None
    if maximum and value > maximum:
        raise _invalid(
            f"num_ctx {value} exceeds the model's trained maximum ({maximum})",
            param=param, code="unsupported_value")
    return value


def _validate_gpu_layers(value: Any, *, show_data: Optional[dict[str, Any]],
                         param: str) -> Optional[str]:
    """A gpu-layers spec the payload path can resolve for this model."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _invalid("gpu_layers must be a spec string or null", param=param,
                       code="unsupported_value")
    spec = value.strip()
    try:
        parse_gpu_layers_spec(spec)
    except ValueError as exc:
        raise _invalid(str(exc), param=param, code="unsupported_value")
    try:
        resolve_gpu_layers(spec,
                           model_block_count(show_data) if show_data else None)
    except (ValueError, RuntimeError) as exc:
        raise _invalid(str(exc), param=param, code="unsupported_value")
    return spec


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
    num_ctx = settings.get("num_ctx")
    if isinstance(num_ctx, str):
        if num_ctx not in ("max-gpu", "max-cpu"):
            raise _invalid('settings.num_ctx must be an integer, "max-gpu" '
                           'or "max-cpu"',
                           param="settings.num_ctx", code="unsupported_value")
    elif num_ctx is not None and (isinstance(num_ctx, bool)
                                  or not isinstance(num_ctx, int)
                                  or num_ctx < 1):
        raise _invalid('settings.num_ctx must be a positive integer, '
                       '"max-gpu" or "max-cpu"',
                       param="settings.num_ctx", code="unsupported_value")
    cpu_percent = settings.get("cpu_percent")
    if num_ctx == "max-cpu" and cpu_percent is None:
        raise _invalid('settings.num_ctx "max-cpu" needs settings.cpu_percent',
                       param="settings.cpu_percent")
    if cpu_percent is not None:
        if isinstance(cpu_percent, bool) or not isinstance(cpu_percent, int) \
                or not 1 <= cpu_percent <= 100:
            raise _invalid("settings.cpu_percent must be an integer between "
                           "1 and 100", param="settings.cpu_percent",
                           code="unsupported_value")
        if num_ctx != "max-cpu":
            raise _invalid('settings.cpu_percent needs settings.num_ctx '
                           '"max-cpu"', param="settings.cpu_percent")
    gpu_layers = settings.get("gpu_layers")
    if gpu_layers is not None and (not isinstance(gpu_layers, str)
                                   or not gpu_layers.strip()):
        raise _invalid("settings.gpu_layers must be a spec string or null",
                       param="settings.gpu_layers", code="unsupported_value")
    parameters = settings.get("parameters", {})
    if not isinstance(parameters, dict):
        raise _invalid("settings.parameters must be an object",
                       param="settings.parameters")
    parameters = validate_settings_parameters(parameters,
                                              where="settings.parameters")
    cancel_on_disconnect = settings.get("cancel_on_disconnect", False)
    _require_bool(cancel_on_disconnect, "settings.cancel_on_disconnect")

    return AgentTurnRequest(
        session_id=session_id, model=model, messages=parsed_messages,
        persist=bool(persist), tools=bool(tools), max_rounds=int(max_rounds),
        approval=str(approval), system=system,
        parameters=dict(parameters), num_ctx=num_ctx,
        cpu_percent=cpu_percent, gpu_layers=gpu_layers,
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
    #: A pending turn-start fit (`("max-gpu"|"max-cpu", percent|cpu=None)`).
    fit_plan: Optional[tuple[str, Optional[int]]] = None
    #: The turn-start fit's outcome dict (events + `result.fit`).
    fit: Optional[dict[str, Any]] = None
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
            "fit": snapshot.result.fit,
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


def _download_name(filename: str) -> str:
    """An ASCII-safe Content-Disposition name (quotes and bars never leak)."""
    cleaned = "".join(ch for ch in str(filename or "")
                      if ch.isprintable() and ch not in '"\\')
    return cleaned.encode("ascii", "replace").decode("ascii") or "artifact"


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
                 artifacts: Optional[ArtifactStore] = None,
                 fits: Optional[FitService] = None,
                 tracker: Optional[Any] = None):
        self._args = args
        self._config = dict(config or {})
        self._store = store
        self._client_factory = client_factory
        #: The serve artifact store: `file`/image parts resolve through it,
        #: turn tools publish into it, and the artifact routes read from it.
        self._artifacts = artifacts
        #: The serve model tracker (`helpers.ModelTracker`): feeds the
        #: unused-on-exit unload. None outside `ollamarun serve`.
        self._tracker = tracker
        # Every turn-manager key is explicit-None defaulted: a configured 0
        # must not be conflated with "unset" (`or 4` would).
        raw_max_active = self._config.get("serve_max_active_turns")
        raw_buffer = self._config.get("serve_turn_event_buffer")
        raw_turn_ttl = self._config.get("serve_turn_event_ttl")
        self._manager = manager or TurnManager(
            max_active=4 if raw_max_active is None else int(raw_max_active),
            buffer_events=10000 if raw_buffer is None else int(raw_buffer),
            completed_ttl=86400.0 if raw_turn_ttl is None
            else float(raw_turn_ttl))
        #: Asynchronous context fits (`fits.py`); the TTL is explicit-None
        #: defaulted so 0 can mean "keep forever".
        raw_fit_ttl = self._config.get("serve_fit_ttl")
        raw_fit_active = self._config.get("serve_max_active_fits")
        self._fits = fits or FitService(
            max_active=2 if raw_fit_active is None else int(raw_fit_active),
            ttl=86400.0 if raw_fit_ttl is None else float(raw_fit_ttl))
        #: How long a parked approval may wait for a decision, in seconds;
        #: an explicit 0 disables the timeout (the approval waits until it is
        #: resolved, cancelled or the server shuts down), so `0` must not be
        #: conflated with "unset" the way `or 300` would.
        raw_approval_timeout = self._config.get("serve_approval_timeout")
        self._approval_timeout = 300.0 if raw_approval_timeout is None \
            else float(raw_approval_timeout)
        raw_turn_timeout = self._config.get("serve_turn_timeout")
        self._turn_timeout = 0.0 if raw_turn_timeout is None \
            else float(raw_turn_timeout)
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
        #: The API tool workspace (always set: the configured serve_workspace
        #: or the startup cwd). Path-argument tools refuse escapes; it is an
        #: argument/default-cwd policy, never subprocess confinement.
        self._workspace = self._cwd
        #: The optional `serve_models` allowlist ([] = every model).
        self._models_allow = list(self._config.get("serve_models") or [])
        #: Aggregate resolved-input caps (`serve_max_*`): shared with the
        #: OpenAI surface through the same budget accounting.
        self._limits = input_limits(self._config)
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
        # The serve_models allowlist answers exactly like an unknown model:
        # never reveal installed-but-forbidden.
        if not model_allowed(model, self._models_allow) \
                or not model_listed(model, entries):
            raise ApiError(404, f"model {model!r} not found",
                           code="model_not_found", param="model")

    def _note_model(self, model: str) -> None:
        """Remember a model this server used (unload-on-exit; never raises)."""
        if self._tracker is not None:
            self._tracker.note(model)

    def _unload_resident(self, client: Any, model: str) -> None:
        """Unload a resident model after a placement change (never raises).

        `/gpu-layers` parity: the next request reloads with the new
        placement instead of the old one silently persisting.
        """
        if not model:
            return
        try:
            if client.is_resident(model):
                client.unload(model)
        except (OllamaError, TypeError, ValueError):
            pass

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
            fitting = self._fits.active_for(session.id)
            if fitting:
                raise _session_fitting(session.id, fitting)
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
        self._note_model(session.model)
        if spec.system is not None:
            session.system = spec.system
        if spec.parameters:
            session.parameters = {**(session.parameters or {}),
                                  **spec.parameters}
        fit_plan: Optional[tuple[str, Optional[int]]] = None
        if spec.gpu_layers is not None:
            show = _show_data(client, session.model)
            value = _validate_gpu_layers(spec.gpu_layers, show_data=show,
                                         param="settings.gpu_layers")
            if value != session.gpu_layers:
                session.gpu_layers = value
                self._unload_resident(client, session.model)
        if isinstance(spec.num_ctx, str):
            if self._fit_policy() == "off":
                raise ApiError(403, "context fitting is disabled",
                               code="fit_disabled")
            fit_plan = (spec.num_ctx, spec.cpu_percent)
        elif spec.num_ctx is not None:
            show = _show_data(client, session.model)
            value = _validate_num_ctx(spec.num_ctx, show_data=show,
                                      param="settings.num_ctx")
            session.parameters = {**(session.parameters or {})}
            session.parameters["num_ctx"] = value

        # Resolve every message's content parts before the turn is accepted:
        # a missing artifact id, an oversize image and a text/file mapping are
        # all decided here, so the client gets a structured 4xx synchronously.
        resolved_messages: list[dict[str, Any]] = []
        has_images = False
        budget = InputBudget(self._limits)
        budget.note_messages(len(spec.messages))
        for index, message in enumerate(spec.messages):
            resolved = resolve_content_parts(
                message.get("content"), where=f"messages[{index}].content",
                store=self._artifacts, budget=budget)
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
            messages_pending=resolved_messages, fit_plan=fit_plan)
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

    # ── files and artifacts ─────────────────────────────────────────────────

    def _require_artifacts(self) -> ArtifactStore:
        if self._artifacts is None:
            raise ApiError(503, "the artifact store is not available",
                           type_="api_error", code="artifacts_unavailable")
        return self._artifacts

    def upload_file(self, request: Request,
                    params: dict[str, str]) -> Response:
        """`POST /v1/files`: one JSON upload as an artifact (source user)."""
        store = self._require_artifacts()
        body = request.json
        if not isinstance(body, dict):
            raise _invalid("request body must be a JSON object")
        unknown = set(body) - _UPLOAD_FIELDS
        if unknown:
            raise _invalid(f"unknown field: {sorted(unknown)[0]}",
                           param=sorted(unknown)[0], code="unknown_field")
        filename = body.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            raise _invalid("filename must be a non-empty string",
                           param="filename")
        media_type = body.get("media_type")
        if not isinstance(media_type, str) or not media_type.strip():
            raise _invalid("media_type must be a non-empty string",
                           param="media_type")
        data_b64 = body.get("data_base64")
        if not isinstance(data_b64, str) or not data_b64:
            raise _invalid("data_base64 must be a base64 string",
                           param="data_base64")
        # Bound the decoded size before allocating (base64 is ~4/3 of it).
        if 0 < store.max_bytes < (len(data_b64) * 3) // 4:
            raise ApiError(
                413, f"artifact exceeds the {store.max_bytes} byte limit",
                param="data_base64", code="artifact_too_large")
        try:
            payload = base64.b64decode(data_b64, validate=True)
        except (ValueError, TypeError):
            raise _invalid("data_base64 is not valid base64",
                           param="data_base64")
        ref = store.register(payload, filename=filename.strip(),
                             media_type=media_type.strip(), source="user")
        return Response.json(ref.to_dict(), status=201)

    def artifact_download(self, request: Request,
                          params: dict[str, str]) -> Response:
        """`GET /v1/artifacts/{id}`: the stored bytes, with its media type."""
        store = self._require_artifacts()
        artifact_id = params.get("id", "")
        loaded = store.read(artifact_id)
        if loaded is None:
            raise ApiError(404, f"artifact {artifact_id!r} not found",
                           code="artifact_not_found")
        ref, payload = loaded
        return Response(
            status=200, body=payload, media_type=ref.media_type,
            headers=[("Content-Disposition",
                      f'attachment; filename="{_download_name(ref.filename)}"'),
                     ("X-Content-Type-Options", "nosniff")])

    def artifact_meta(self, request: Request,
                      params: dict[str, str]) -> Response:
        """`GET /v1/artifacts/{id}/meta`: the full ref."""
        store = self._require_artifacts()
        artifact_id = params.get("id", "")
        ref = store.resolve(artifact_id)
        if ref is None:
            raise ApiError(404, f"artifact {artifact_id!r} not found",
                           code="artifact_not_found")
        return Response.json(ref.to_dict())

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
        # The API tool context is jailed to the workspace: path arguments
        # resolve under it and escapes are refused (the CLI leaves
        # `workspace` unset and is unchanged).
        repl.tool_ctx.workspace = self._workspace
        repl.tool_ctx.cwd = self._workspace
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
                if state.fit_plan is not None:
                    self._run_turn_fit(state, sink)
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
            turn_result = runtime.to_turn_result(result, session_id=session.id,
                                                 turn_id=state.id)
            if state.fit is not None:
                turn_result.fit = state.fit
            return turn_result
        finally:
            state.done.set()

    def _run_turn_fit(self, state: AgentTurnState,
                      sink: Callable[[RuntimeEvent], None]) -> None:
        """The turn-start estimate for `settings.num_ctx: "max-gpu"|"max-cpu"`.

        Estimate only -- turns never block on a probe. On success the fitted
        num_ctx is persisted and the outcome rides `session.fit` and
        `result.fit`; on failure the context is left unchanged and the same
        event carries the reason.
        """
        mode = "cpu" if state.fit_plan[0] == "max-cpu" else "gpu"
        percent = state.fit_plan[1]
        try:
            request = self._fit_request(state.client, state.session.model,
                                        state.session)
            outcome = fit_context(request, mode=mode, percent=percent,
                                  probe=False)
            payload = outcome.to_dict()
            payload["applied"] = outcome.num_ctx is not None
            if outcome.num_ctx is not None:
                state.session.parameters = {**(state.session.parameters or {})}
                state.session.parameters["num_ctx"] = outcome.num_ctx
                state.session.touch()
        except Exception:
            logger.debug("turn-start fit failed", exc_info=True)
            payload = {"model": state.session.model, "mode": mode,
                       "num_ctx": None, "applied": False,
                       "error": "the fit failed"}
        state.fit = payload
        sink(RuntimeEvent(SESSION_FIT, dict(payload)))

    # ── context fits ────────────────────────────────────────────────────────

    def _fit_policy(self) -> str:
        value = str(self._config.get("serve_fit") or "probe").lower()
        return value if value in {"off", "estimate", "probe"} else "probe"

    def _fit_guard(self, session: Session) -> None:
        """Refuse to fit (or apply) while the session is busy either way."""
        active = self._manager.active_for(session.id)
        if active:
            raise _session_busy(session.id, active)
        fitting = self._fits.active_for(session.id)
        if fitting:
            raise _session_fitting(session.id, fitting)

    def _parse_fit_body(self, body: Any, *, fields: frozenset,
                        allow_probe: bool) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise _invalid("request body must be a JSON object")
        unknown = set(body) - fields
        if unknown:
            raise _invalid(f"unknown field: {sorted(unknown)[0]}",
                           param=sorted(unknown)[0], code="unknown_field")
        mode = body.get("mode", "gpu")
        if mode not in ("gpu", "cpu"):
            raise _invalid('mode must be "gpu" or "cpu"', param="mode",
                           code="unsupported_value")
        percent = body.get("cpu_percent")
        if mode == "cpu" and percent is None:
            raise _invalid('mode "cpu" needs cpu_percent', param="cpu_percent")
        if percent is not None:
            if isinstance(percent, bool) or not isinstance(percent, int) \
                    or not 1 <= percent <= 100:
                raise _invalid("cpu_percent must be an integer between 1 and 100",
                               param="cpu_percent", code="unsupported_value")
            if mode != "cpu":
                raise _invalid('cpu_percent needs mode "cpu"',
                               param="cpu_percent")
        gpu_layers = body.get("gpu_layers")
        if gpu_layers is not None:
            if not isinstance(gpu_layers, str) or not gpu_layers.strip():
                raise _invalid("gpu_layers must be a spec string or null",
                               param="gpu_layers", code="unsupported_value")
            gpu_layers = gpu_layers.strip()
            try:
                parse_gpu_layers_spec(gpu_layers)
            except ValueError as exc:
                raise _invalid(str(exc), param="gpu_layers",
                               code="unsupported_value")
        apply = False
        if "apply" in fields:
            apply = body.get("apply", False)
            _require_bool(apply, "apply")
        probe = True
        if allow_probe:
            probe = body.get("probe", True)
            _require_bool(probe, "probe")
        session_id = body.get("session_id")
        if session_id is not None:
            session_id = _optional_string(session_id, "session_id").strip() or None
        model = _optional_string(body.get("model"), "model").strip()
        return {"mode": mode, "cpu_percent": percent,
                "gpu_layers": gpu_layers, "apply": bool(apply),
                "probe": bool(probe), "session_id": session_id,
                "model": model}

    def _fit_subject(self, spec: dict[str, Any]
                     ) -> tuple[Optional[Session], Any, str]:
        session = self._session(spec["session_id"]) if spec["session_id"] \
            else None
        client = self._client_factory()
        model = spec["model"] or (session.model if session is not None else "")
        if not model:
            model = str(self._config.get("model") or "")
        if not model:
            raise _invalid("model is required (none in the request, the "
                           "session or the serve config)", param="model")
        self._ensure_model(client, model)
        return session, client, model

    def _fit_request(self, client: Any, model: str,
                     session: Optional[Session],
                     *, pinned: Optional[str] = None) -> FitRequest:
        """The ctxfit request for this server's upstream (CLI parity)."""
        remote = str(self._config.get("remote") or "") or None
        ssh_target = remote_node(remote, self._config) if remote else None
        ssh_user = self._config.get("remote_ssh_user")
        base = getattr(client, "base", "") or ""
        host_label = remote or (None if is_local_host(base) else base)
        parameters = (session.parameters if session is not None else {}) or {}
        pinned_spec = pinned if pinned is not None else \
            (session.gpu_layers if session is not None else None)
        return FitRequest(
            client=client, model=model, config=self._config,
            host_label=host_label, ssh_target=ssh_target, ssh_user=ssh_user,
            pinned_spec=pinned_spec, num_gpu=parameters.get("num_gpu"),
            base_options={key: value for key, value in parameters.items()
                          if key != "num_ctx"},
            keepalive=(session.keepalive if session is not None else None)
            or self._config.get("keepalive"),
            refresh_gpu=lambda: machine_state_for(base, ssh_target=ssh_target,
                                                  ssh_user=ssh_user))

    def _fit_warnings(self, client: Any, model: str) -> list[str]:
        """Honest notes on cost: contention now, calibration from before."""
        warnings: list[str] = []
        wanted = canonical_model_name(model)
        try:
            others = [str(entry.get("name") or entry.get("model") or "?")
                      for entry in client.ps() or []
                      if int(entry.get("size_vram") or 0) > 0
                      and canonical_model_name(str(entry.get("name")
                                                   or entry.get("model") or ""))
                      != wanted]
        except (OllamaError, TypeError, ValueError):
            others = []
        if others:
            warnings.append(f"{len(others)} other model(s) are resident on "
                            f"the backend — the probe will be slower")
        try:
            if self._manager.active_count() > 0:
                warnings.append("turns are active on this server — the probe "
                                "may be slower")
        except Exception:  # pragma: no cover - manager always answers
            pass
        digest = ""
        try:
            for entry in client.tags():
                if canonical_model_name(entry.get("name") or "") == wanted:
                    digest = str(entry.get("digest") or "")
                    break
        except (OllamaError, TypeError, ValueError):
            digest = ""
        if digest:
            cached = CtxCalibrationStore().digest_points(digest)
            if cached:
                warnings.append(f"{cached} prior measurement(s) on this host "
                                f"seed the probe (fewer loads)")
        return warnings

    def _apply_fit(self, job: Any) -> None:
        """The fit service's on_success: persist the fitted num_ctx."""
        if not job.session_id or job.outcome is None \
                or job.outcome.num_ctx is None:
            return
        session = self._session(job.session_id)
        session.parameters = {**(session.parameters or {})}
        session.parameters["num_ctx"] = job.outcome.num_ctx
        session.touch()
        self._store.save(session)

    def estimate_fit(self, request: Request,
                     params: dict[str, str]) -> Response:
        """`POST /v1/agent/fit`: synchronous estimate; optional apply."""
        if self._fit_policy() == "off":
            raise ApiError(403, "context fitting is disabled",
                           code="fit_disabled")
        spec = self._parse_fit_body(request.json, fields=_FIT_APPLY_FIELDS,
                                    allow_probe=False)
        session, client, model = self._fit_subject(spec)
        if spec["apply"] and session is None:
            raise _invalid("apply needs session_id", param="session_id")
        if spec["apply"] and session is not None:
            self._fit_guard(session)
        outcome = fit_context(
            self._fit_request(client, model, session,
                              pinned=spec["gpu_layers"]),
            mode=spec["mode"], percent=spec["cpu_percent"], probe=False)
        if outcome.error is not None:
            raise ApiError(400, outcome.error, code="fit_failed")
        body = outcome.to_dict()
        body["object"] = "fit"
        body["applied"] = False
        if spec["apply"] and session is not None \
                and outcome.num_ctx is not None:
            session.parameters = {**(session.parameters or {})}
            session.parameters["num_ctx"] = outcome.num_ctx
            session.touch()
            self._store.save(session)
            body["applied"] = True
        return Response.json(body)

    def start_fit(self, request: Request,
                  params: dict[str, str]) -> Response:
        """`POST /v1/agent/fits`: start an asynchronous probe job."""
        policy = self._fit_policy()
        if policy == "off":
            raise ApiError(403, "context fitting is disabled",
                           code="fit_disabled")
        spec = self._parse_fit_body(request.json, fields=_FIT_JOB_FIELDS,
                                    allow_probe=True)
        if spec["probe"] and policy != "probe":
            raise ApiError(403, 'probing is disabled (serve_fit is "estimate")',
                           code="probe_disabled")
        session, client, model = self._fit_subject(spec)
        if spec["probe"]:
            self._note_model(model)  # a probe loads the model; estimates don't
        if spec["apply"] and session is None:
            raise _invalid("apply needs session_id", param="session_id")
        if session is not None:
            self._fit_guard(session)
        try:
            job = self._fits.submit(
                model=model, mode=spec["mode"],
                req=self._fit_request(client, model, session,
                                      pinned=spec["gpu_layers"]),
                percent=spec["cpu_percent"], probe=spec["probe"],
                apply=spec["apply"],
                session_id=session.id if session is not None else None,
                on_success=self._apply_fit)
        except FitBusy as exc:
            raise _session_fitting(exc.session_id, exc.active_fit_id)
        except TooManyFits as exc:
            raise ApiError(429, "too many active fits",
                           type_="rate_limit_error", code="rate_limit_error",
                           extra={"limit": exc.limit})
        except FitsClosed:
            raise ApiError(503, "the server is shutting down",
                           type_="api_error", code="server_shutting_down")
        body = {
            "id": job.id, "object": "fit", "status": job.status,
            "session_id": job.session_id, "model": job.model,
            "mode": job.mode, "probe": job.probe, "apply": job.apply,
            "created_at": job.created_at,
            "status_url": f"/v1/agent/fits/{job.id}",
            "cancel_url": f"/v1/agent/fits/{job.id}/cancel",
            "warnings": self._fit_warnings(client, model),
        }
        return Response.json(body, status=202)

    def fit_status(self, request: Request,
                   params: dict[str, str]) -> Response:
        fit_id = params.get("id", "")
        job = self._fits.get(fit_id)
        if job is None:
            raise ApiError(404, f"fit {fit_id!r} not found",
                           code="fit_not_found")
        return Response.json(fit_body(job))

    def cancel_fit(self, request: Request,
                   params: dict[str, str]) -> Response:
        fit_id = params.get("id", "")
        job = self._fits.get(fit_id)
        if job is None:
            raise ApiError(404, f"fit {fit_id!r} not found",
                           code="fit_not_found")
        if job.status in TERMINAL_FIT_STATUSES:
            raise ApiError(409, f"fit {fit_id!r} already finished",
                           type_="conflict_error", code="fit_finished",
                           extra={"status": job.status})
        self._fits.cancel(job.id)
        return Response.json({"id": job.id, "object": "fit",
                              "cancel_requested": True, "status": job.status},
                             status=202)

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
        parameters = validate_settings_parameters(parameters,
                                                  where="parameters")
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
        self._fit_guard(session)
        body = request.json
        if not isinstance(body, dict) or not body:
            raise _invalid("request body must be a non-empty JSON object")
        unknown = set(body) - _SESSION_PATCH_FIELDS
        if unknown:
            raise _invalid(f"unknown field: {sorted(unknown)[0]}",
                           param=sorted(unknown)[0], code="unknown_field")
        if "title" in body:
            session.title = _optional_string(body["title"], "title")
        client: Optional[Any] = None

        def _client() -> Any:
            nonlocal client
            if client is None:
                client = self._client_factory()
            return client

        if "model" in body:
            model = _optional_string(body["model"], "model").strip()
            if model:
                self._ensure_model(_client(), model)
            session.model = model
        if "num_ctx" in body or "gpu_layers" in body:
            show = _show_data(_client(), session.model)
            if "num_ctx" in body:
                value = _validate_num_ctx(body["num_ctx"], show_data=show,
                                          param="num_ctx")
                session.parameters = {**(session.parameters or {})}
                if value is None:
                    session.parameters.pop("num_ctx", None)
                else:
                    session.parameters["num_ctx"] = value
            if "gpu_layers" in body:
                value = _validate_gpu_layers(body["gpu_layers"], show_data=show,
                                             param="gpu_layers")
                if value != session.gpu_layers:
                    session.gpu_layers = value
                    self._unload_resident(_client(), session.model)
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
        if "parameters" in body:
            incoming = body["parameters"]
            if not isinstance(incoming, dict):
                raise _invalid("parameters must be an object",
                               param="parameters")
            incoming = validate_settings_parameters(incoming,
                                                    where="parameters")
            session.parameters = {**(session.parameters or {}), **incoming}
        session.touch()
        self._store.save(session)
        return Response.json(session_body(
            session, active_turn=self._manager.active_for(session.id)))

    def delete_session(self, request: Request,
                       params: dict[str, str]) -> Response:
        session = self._session(params.get("id", ""))
        self._fit_guard(session)
        try:
            self._store.delete(session.id)
        except Exception as exc:
            logger.debug("session delete failed: %s", exc)
            raise ApiError(500, "the session could not be moved to the trash",
                           type_="api_error", code="delete_failed")
        return Response.json({"id": session.id, "object": "session.deleted",
                              "deleted": True})

    # ── lifecycle ───────────────────────────────────────────────────────────

    def begin_shutdown(self) -> None:
        """Refuse new turns and fits; cancelling and joining come after.

        Called before the pending work is cancelled so an in-flight HTTP
        handler racing the shutdown cannot start a new turn or fit — its
        submit answers 503 `server_shutting_down` through the typed errors
        the registries already raise once closed.
        """
        self._manager.begin_shutdown()
        self._fits.begin_shutdown()

    def shutdown(self, *, timeout: float = 10.0) -> None:
        """Resolve pending approvals as cancelled, then stop the registries."""
        self.begin_shutdown()
        for pending in self._pending_approvals_all():
            self._first_decision(pending, ApprovalDecision(
                run=False, note=_cancel_note("shutdown"), action="cancelled"))
        self._manager.shutdown(cancel=True, timeout=timeout)
        self._fits.shutdown(cancel=True, timeout=timeout)

    def _pending_approvals_all(self) -> list[PendingApproval]:
        with self._lock:
            return [pending for pending in self._approvals.values()
                    if not pending.resolved]


def _session_busy(session_id: str, active_turn_id: str) -> ApiError:
    return ApiError(409, f"session {session_id!r} already has an active turn",
                    type_="conflict_error", code="session_busy",
                    extra={"active_turn_id": active_turn_id})


def _session_fitting(session_id: str, active_fit_id: str) -> ApiError:
    return ApiError(409, f"session {session_id!r} is fitting",
                    type_="conflict_error", code="session_busy",
                    extra={"active_fit_id": active_fit_id})


def build_agent_api(*, args: Any, config: dict[str, Any],
                    store: SessionStore, client_factory: Callable[[], Any],
                    manager: Optional[TurnManager] = None,
                    cwd: Optional[str] = None,
                    artifacts: Optional[ArtifactStore] = None,
                    fits: Optional[FitService] = None,
                    tracker: Optional[Any] = None) -> "AgentApi":
    """The one `AgentApi` the route-registration functions share."""
    return AgentApi(args=args, config=config, store=store,
                    client_factory=client_factory, manager=manager, cwd=cwd,
                    artifacts=artifacts, fits=fits, tracker=tracker)


def add_artifact_routes(router: Router, api: "AgentApi") -> None:
    """Register the shared file/artifact endpoints.

    These serve both API surfaces: the OpenAI `file` parts and every
    artifact download stay available when the native agent routes are
    switched off (`serve_agent: false`).
    """
    router.add("POST", "/v1/files", api.upload_file)
    router.add("GET", "/v1/artifacts/{id}", api.artifact_download)
    router.add("GET", "/v1/artifacts/{id}/meta", api.artifact_meta)


def add_agent_routes(router: Router, api: "AgentApi") -> None:
    """Register the native agent endpoints (turns, approvals, sessions, fits)."""
    router.add("POST", "/v1/agent/turns", api.start_turn)
    router.add("GET", "/v1/agent/turns/{id}", api.turn_status)
    router.add("GET", "/v1/agent/turns/{id}/events", api.turn_events)
    router.add("POST", "/v1/agent/turns/{id}/cancel", api.cancel_turn)
    router.add("POST", "/v1/agent/approvals/{id}", api.resolve_approval)
    router.add("POST", "/v1/agent/fit", api.estimate_fit)
    router.add("POST", "/v1/agent/fits", api.start_fit)
    router.add("GET", "/v1/agent/fits/{id}", api.fit_status)
    router.add("POST", "/v1/agent/fits/{id}/cancel", api.cancel_fit)
    router.add("GET", "/v1/sessions", api.list_sessions)
    router.add("POST", "/v1/sessions", api.create_session)
    router.add("GET", "/v1/sessions/{id}", api.get_session)
    router.add("PATCH", "/v1/sessions/{id}", api.patch_session)
    router.add("DELETE", "/v1/sessions/{id}", api.delete_session)
