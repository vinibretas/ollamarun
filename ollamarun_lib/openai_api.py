"""The OpenAI-compatible `/v1` surface: model turns over HTTP.

These endpoints are a translation proxy. One chat request becomes one Ollama
`/api/chat` call whose response is reshaped into OpenAI schemas (`n > 1` is
emulated with that many sequential generations); one embeddings request
becomes one `/api/embed` call. Client `tools` are passed to the backend and
come back as standard `tool_calls` with `finish_reason: "tool_calls"` -- they
are never executed here. The agent loop (harness tools, approvals, sessions)
is not reachable from this module; full agent turns belong to the native
`/v1/agent/*` endpoints in `agent_api.py`.

Streaming speaks OpenAI SSE only (role chunk, content/thinking deltas,
finish reason, optional usage chunk, `[DONE]`); the typed `turn.*` event
stream never appears on this wire. Thinking models surface their reasoning
in the non-standard `delta.reasoning_content` (normal clients ignore it).
A disconnecting client cancels its own upstream generation -- a model-only
completion has no recoverable job behind it.

The request extension `system` keeps ordinary OpenAI requests untouched:
absent or `""` sends no system prompt, `"auto"` injects the current
built-in default, any other text becomes the leading system message.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from .artifacts import (MAX_IMAGE_BYTES, InputBudget, InputLimits,
                        input_limits, require_vision, resolve_content_parts)
from .client import OllamaError, build_chat_payload
from .constants import DEFAULT_TIMEOUT
from .helpers import CancelToken, canonical_model_name, model_allowed
from .prompts import default_system_prompt
from .renderer import GenerationResult, consume_chunk, stats_from_final
from .server import ApiError, Request, Response, Router, SseWriter, StreamResponse

logger = logging.getLogger("ollamarun.openai_api")

#: Fields a chat-completion request may carry. `user`, `metadata` and a
#: `parallel_tool_calls` boolean are accepted (`user`/`metadata` are
#: tracking hints with no backend meaning; `parallel_tool_calls: false` is
#: honored by truncating to the first returned call); everything else is
#: rejected explicitly.
_CHAT_FIELDS = frozenset({
    "model", "messages", "stream", "stream_options", "temperature", "top_p",
    "max_tokens", "max_completion_tokens", "stop", "seed",
    "presence_penalty", "frequency_penalty", "response_format", "tools",
    "tool_choice", "n", "logprobs", "top_logprobs", "logit_bias", "audio",
    "modalities", "reasoning_effort", "system",
    "user", "metadata", "parallel_tool_calls",
})

#: Fields an embeddings request may carry (`user` is accepted and ignored,
#: matching the chat mapper).
_EMBEDDING_FIELDS = frozenset({"model", "input", "encoding_format",
                               "dimensions", "user"})

#: `n > 1` is emulated with that many sequential upstream generations, so a
#: request holds its handler for roughly n generations; the cap keeps one
#: request's fan-out bounded.
MAX_N = 8


@dataclass
class ChatRequest:
    """A validated chat-completion request, mapped for the backend."""

    model: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    tools: Optional[list[dict[str, Any]]] = None
    fmt: Any = None
    think: Any = None
    stream: bool = False
    include_usage: bool = False
    has_images: bool = False
    n: int = 1
    parallel_tool_calls: Optional[bool] = None


@dataclass
class EmbeddingRequest:
    """A validated embeddings request, mapped for the backend."""

    model: str
    inputs: list[str] = field(default_factory=list)
    encoding_format: str = "float"
    dimensions: Optional[int] = None


# ── request validation and mapping ──────────────────────────────────────────


def parse_chat_request(body: Any, *,
                       store: Optional[Any] = None,
                       limits: Optional[InputLimits] = None) -> ChatRequest:
    """Validate one `/v1/chat/completions` body; raises `ApiError(400)`."""
    if not isinstance(body, dict):
        raise ApiError(400, "request body must be a JSON object")
    for name in body:
        if name not in _CHAT_FIELDS:
            raise ApiError(400, f"unknown field: {name}", param=name,
                           code="unknown_field")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ApiError(400, "model must be a non-empty string", param="model")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "messages must be a non-empty array",
                       param="messages")

    _reject_unsupported(body)
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise ApiError(400, "stream must be a boolean", param="stream")
    include_usage = _stream_options(body)
    system = _system_from(body)
    converted, has_images = _messages_from(messages, system, store=store,
                                           budget=InputBudget(limits))
    return ChatRequest(
        model=model.strip(),
        messages=converted,
        options=_options_from(body),
        tools=_tools_from(body),
        fmt=_format_from(body),
        think=_think_from(body),
        stream=stream,
        include_usage=include_usage,
        has_images=has_images,
        n=_n_from(body),
        parallel_tool_calls=_parallel_tool_calls_from(body),
    )


def _n_from(body: dict[str, Any]) -> int:
    """The validated `n`: a positive integer, capped at `MAX_N`."""
    value = body.get("n")
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ApiError(400, "n must be a positive integer", param="n")
    if value > MAX_N:
        raise ApiError(
            400, f"n is limited to {MAX_N} emulated generations",
            param="n", code="unsupported_value")
    return value


def _parallel_tool_calls_from(body: dict[str, Any]) -> Optional[bool]:
    if "parallel_tool_calls" not in body:
        return None
    value = body["parallel_tool_calls"]
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ApiError(400, "parallel_tool_calls must be a boolean",
                       param="parallel_tool_calls")
    return value


def _reject_unsupported(body: dict[str, Any]) -> None:
    if body.get("logprobs"):
        raise ApiError(400, "logprobs is not supported", param="logprobs",
                       code="unsupported_parameter")
    if body.get("top_logprobs") is not None:
        raise ApiError(400, "top_logprobs is not supported",
                       param="top_logprobs", code="unsupported_parameter")
    if body.get("logit_bias"):
        raise ApiError(400, "logit_bias is not supported", param="logit_bias",
                       code="unsupported_parameter")
    if body.get("audio") is not None:
        raise ApiError(400, "audio is not supported", param="audio",
                       code="unsupported_parameter")
    modalities = body.get("modalities")
    if modalities is not None:
        if not isinstance(modalities, list) \
                or any(item != "text" for item in modalities):
            raise ApiError(400, 'only the "text" modality is supported',
                           param="modalities", code="unsupported_value")


def _stream_options(body: dict[str, Any]) -> bool:
    options = body.get("stream_options")
    if options is None:
        return False
    if not isinstance(options, dict):
        raise ApiError(400, "stream_options must be an object",
                       param="stream_options")
    for key in options:
        if key != "include_usage":
            raise ApiError(400, f"unknown stream_options field: {key}",
                           param=f"stream_options.{key}", code="unknown_field")
    if "include_usage" not in options:
        return False
    value = options["include_usage"]
    if not isinstance(value, bool):
        raise ApiError(400, "stream_options.include_usage must be a boolean",
                       param="stream_options.include_usage")
    return value


def _system_from(body: dict[str, Any]) -> Optional[str]:
    if "system" not in body or body["system"] is None:
        return None
    system = body["system"]
    if not isinstance(system, str):
        raise ApiError(400, 'system must be a string or "auto"', param="system")
    if system.lower() == "auto":
        return default_system_prompt(with_tools=False)
    return system or None


def _number(value: Any, name: str, *, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(400, f"{name} must be a number", param=name)
    if not low <= value <= high:
        raise ApiError(400, f"{name} must be between {low} and {high}",
                       param=name)
    return float(value)


def _options_from(body: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if body.get("temperature") is not None:
        options["temperature"] = _number(body["temperature"], "temperature",
                                         low=0, high=2)
    if body.get("top_p") is not None:
        options["top_p"] = _number(body["top_p"], "top_p", low=0, high=1)
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) \
                or max_tokens < 1:
            raise ApiError(400, "max_tokens must be a positive integer",
                           param="max_tokens")
        options["num_predict"] = int(max_tokens)
    stop = body.get("stop")
    if stop is not None:
        if isinstance(stop, str):
            options["stop"] = stop
        elif isinstance(stop, list) and all(isinstance(item, str)
                                            for item in stop):
            options["stop"] = list(stop)
        else:
            raise ApiError(400, "stop must be a string or an array of strings",
                           param="stop")
    seed = body.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ApiError(400, "seed must be an integer", param="seed")
        options["seed"] = seed
    for name in ("presence_penalty", "frequency_penalty"):
        if body.get(name) is not None:
            options[name] = _number(body[name], name, low=-2, high=2)
    return options


def _format_from(body: dict[str, Any]) -> Any:
    rf = body.get("response_format")
    if rf is None:
        return None
    if not isinstance(rf, dict):
        raise ApiError(400, "response_format must be an object",
                       param="response_format")
    kind = rf.get("type")
    if kind is None or kind == "text":
        return None
    if kind == "json_object":
        return "json"
    if kind == "json_schema":
        wrapper = rf.get("json_schema")
        if not isinstance(wrapper, dict):
            raise ApiError(400, "json_schema must be an object",
                           param="response_format.json_schema")
        unknown = set(wrapper) - {"name", "description", "schema", "strict"}
        if unknown:
            raise ApiError(400, f"unknown json_schema field: {sorted(unknown)[0]}",
                           param="response_format.json_schema",
                           code="unknown_field")
        for key in ("name", "description"):
            if key in wrapper and not isinstance(wrapper[key], str):
                raise ApiError(400, f"json_schema.{key} must be a string",
                               param=f"response_format.json_schema.{key}")
        if "strict" in wrapper and not isinstance(wrapper["strict"], bool):
            raise ApiError(400, "json_schema.strict must be a boolean",
                           param="response_format.json_schema.strict")
        schema = wrapper.get("schema")
        if not isinstance(schema, dict):
            raise ApiError(400, "json_schema needs a schema object",
                           param="response_format.json_schema.schema")
        return schema
    raise ApiError(400, f"unsupported response_format type: {kind!r}",
                   param="response_format.type", code="unsupported_value")


def _think_from(body: dict[str, Any]) -> Any:
    effort = body.get("reasoning_effort")
    if effort is None:
        return None
    if effort in ("low", "medium", "high"):
        return effort
    raise ApiError(400, 'reasoning_effort must be "low", "medium" or "high"',
                   param="reasoning_effort", code="unsupported_value")


def _tools_from(body: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    tools = body.get("tools")
    choice = body.get("tool_choice")
    if tools is None:
        if choice is not None:
            raise ApiError(400, "tool_choice requires tools", param="tool_choice")
        return None
    if not isinstance(tools, list) or not tools:
        raise ApiError(400, "tools must be a non-empty array", param="tools")
    out: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ApiError(400, "only function tools are supported",
                           param=f"tools[{index}]", code="unsupported_value")
        unknown = set(tool) - {"type", "function"}
        if unknown:
            raise ApiError(400, f"unknown tool field: {sorted(unknown)[0]}",
                           param=f"tools[{index}]", code="unknown_field")
        function = tool.get("function")
        if not isinstance(function, dict) \
                or not isinstance(function.get("name"), str) \
                or not function["name"]:
            raise ApiError(400, "each function tool needs a name",
                           param=f"tools[{index}].function.name")
        unknown = set(function) - {"name", "description", "parameters", "strict"}
        if unknown:
            raise ApiError(400, f"unknown function field: {sorted(unknown)[0]}",
                           param=f"tools[{index}].function", code="unknown_field")
        if "strict" in function:
            if not isinstance(function["strict"], bool):
                raise ApiError(400, "strict must be a boolean",
                               param=f"tools[{index}].function.strict")
            if function["strict"]:
                # Ollama does not enforce tool-argument schemas; accepting
                # `true` would promise an enforcement that does not exist.
                raise ApiError(
                    400,
                    "strict tool schemas are not supported (the backend does "
                    "not enforce tool-argument schemas)",
                    param=f"tools[{index}].function.strict",
                    code="unsupported_parameter")
        entry: dict[str, Any] = {"type": "function",
                                 "function": {"name": function["name"]}}
        if function.get("description") is not None:
            if not isinstance(function["description"], str):
                raise ApiError(400, "tool description must be a string",
                               param=f"tools[{index}].function.description")
            entry["function"]["description"] = function["description"]
        if function.get("parameters") is not None:
            if not isinstance(function["parameters"], dict):
                raise ApiError(400, "tool parameters must be a JSON schema",
                               param=f"tools[{index}].function.parameters")
            entry["function"]["parameters"] = function["parameters"]
        out.append(entry)

    if choice is None or choice == "auto":
        return out
    if choice == "none":
        return None
    if isinstance(choice, dict):
        unknown = set(choice) - {"type", "function"}
        if unknown:
            raise ApiError(400, f"unknown tool_choice field: {sorted(unknown)[0]}",
                           param="tool_choice", code="unknown_field")
        function = choice.get("function")
        if not isinstance(function, dict):
            raise ApiError(400, "invalid tool_choice", param="tool_choice")
        function_unknown = set(function) - {"name"}
        if function_unknown:
            raise ApiError(
                400,
                f"unknown tool_choice function field: "
                f"{sorted(function_unknown)[0]}",
                param="tool_choice", code="unknown_field")
        name = function.get("name")
        if choice.get("type") != "function" or not isinstance(name, str):
            raise ApiError(400, "invalid tool_choice", param="tool_choice")
        filtered = [tool for tool in out
                    if tool["function"]["name"] == name]
        if not filtered:
            raise ApiError(400, f"tool_choice names an unknown function: {name}",
                           param="tool_choice")
        return filtered
    raise ApiError(400, 'tool_choice must be "auto", "none" or a specific '
                        "function (the backend cannot force a call)",
                   param="tool_choice", code="unsupported_value")


def _messages_from(messages: list[Any], system: Optional[str], *,
                   store: Optional[Any] = None,
                   budget: Optional[InputBudget] = None
                   ) -> tuple[list[dict[str, Any]], bool]:
    """OpenAI messages -> Ollama payloads; resolves `tool_call_id` -> name."""
    # Validate every element before any role/field inspection so a malformed
    # element is always a structured 400 (never an AttributeError -> 500).
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ApiError(400, f"messages[{index}] must be an object",
                           param=f"messages[{index}]")
    if budget is not None:
        budget.note_messages(len(messages))
    name_by_id: dict[str, str] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            function = call.get("function") or {}
            if isinstance(call_id, str) and isinstance(function, dict) \
                    and isinstance(function.get("name"), str):
                name_by_id[call_id] = function["name"]

    out: list[dict[str, Any]] = []
    if system and (not messages or messages[0].get("role") != "system"):
        out.append({"role": "system", "content": system})
    has_images = False
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            raise ApiError(400, f"unsupported role: {role!r}",
                           param=f"messages[{index}].role")
        unknown = set(message) - {"role", "content", "tool_calls",
                                  "tool_call_id", "name"}
        if unknown:
            raise ApiError(400, f"unknown message field: {sorted(unknown)[0]}",
                           param=f"messages[{index}]", code="unknown_field")
        if role == "tool":
            content = message.get("content")
            if not isinstance(content, str):
                raise ApiError(400, "tool messages need string content",
                               param=f"messages[{index}].content")
            call_id = message.get("tool_call_id")
            name = name_by_id.get(call_id) if isinstance(call_id, str) else None
            if not name:
                raise ApiError(
                    400,
                    f"tool message references an unknown tool_call_id: "
                    f"{call_id!r}",
                    param=f"messages[{index}].tool_call_id")
            if budget is not None:
                budget.note_text(len(content.encode("utf-8")),
                                 f"messages[{index}].content")
            out.append({"role": "tool", "content": content, "tool_name": name})
            continue
        content, images = _content_parts(message.get("content"), index,
                                         store=store, budget=budget)
        has_images = has_images or bool(images)
        payload: dict[str, Any] = {"role": role, "content": content}
        if images:
            payload["images"] = images
        if role == "assistant" and message.get("tool_calls") is not None:
            payload["tool_calls"] = _tool_calls_in(message["tool_calls"], index)
        out.append(payload)
    return out, has_images


def _content_parts(content: Any, index: int, *,
                   store: Optional[Any] = None,
                   budget: Optional[InputBudget] = None
                   ) -> tuple[str, list[str]]:
    """One message's content mapped through the shared resolver.

    The extension part `{"type": "file", "file_id": "art_…"}` resolves
    through the artifact store; `image_url` accepts data: URIs and http(s)
    URLs (bounded fetch). `image_max` is read here so tests can shrink the
    cap on this module.
    """
    resolved = resolve_content_parts(
        content, where=f"messages[{index}].content", store=store,
        image_max=MAX_IMAGE_BYTES, budget=budget)
    return resolved.text, [image.b64 for image in resolved.images]


def _tool_calls_in(calls: Any, index: int) -> list[dict[str, Any]]:
    if not isinstance(calls, list):
        raise ApiError(400, "tool_calls must be an array",
                       param=f"messages[{index}].tool_calls")
    out: list[dict[str, Any]] = []
    for call_index, call in enumerate(calls):
        where = f"messages[{index}].tool_calls[{call_index}]"
        if not isinstance(call, dict):
            raise ApiError(400, "tool calls must be objects", param=where)
        function = call.get("function")
        if not isinstance(function, dict) \
                or not isinstance(function.get("name"), str):
            raise ApiError(400, "tool calls need a function name",
                           param=f"{where}.function.name")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError:
                raise ApiError(400, "tool call arguments must be valid JSON",
                               param=f"{where}.function.arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ApiError(400, "tool call arguments must be a JSON object",
                           param=f"{where}.function.arguments")
        out.append({"function": {"name": function["name"],
                                 "arguments": arguments}})
    return out


# ── embeddings validation ───────────────────────────────────────────────────


def parse_embeddings_request(body: Any, *,
                             limits: Optional[InputLimits] = None
                             ) -> EmbeddingRequest:
    """Validate one `/v1/embeddings` body; raises `ApiError(400)`."""
    if not isinstance(body, dict):
        raise ApiError(400, "request body must be a JSON object")
    for name in body:
        if name not in _EMBEDDING_FIELDS:
            raise ApiError(400, f"unknown field: {name}", param=name,
                           code="unknown_field")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ApiError(400, "model must be a non-empty string", param="model")
    inputs = _embedding_inputs(body.get("input"))
    dimensions = body.get("dimensions")
    if dimensions is not None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) \
                or dimensions < 1:
            raise ApiError(400, "dimensions must be a positive integer",
                           param="dimensions")
    encoding = body.get("encoding_format")
    if encoding is None:
        encoding = "float"
    if encoding not in ("float", "base64"):
        raise ApiError(
            400, 'encoding_format must be "float" or "base64"',
            param="encoding_format", code="unsupported_value")
    budget = InputBudget(limits)
    budget.note_messages(len(inputs), param="input")
    for index, text in enumerate(inputs):
        budget.note_text(len(text.encode("utf-8")), f"input[{index}]")
    return EmbeddingRequest(model=model.strip(), inputs=inputs,
                            encoding_format=encoding, dimensions=dimensions)


def _embedding_inputs(value: Any) -> list[str]:
    """`input` as a non-empty list of non-empty strings.

    OpenAI also accepts token arrays; the backend takes text only, so those
    are rejected explicitly (never silently reinterpreted).
    """
    if isinstance(value, str):
        if not value:
            raise ApiError(400, "input must not be empty", param="input")
        return [value]
    if isinstance(value, list):
        if not value:
            raise ApiError(400, "input must not be an empty array",
                           param="input")
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item:
                raise ApiError(
                    400,
                    "input must be a string or an array of non-empty strings "
                    "(token arrays are not supported)",
                    param=f"input[{index}]", code="unsupported_value")
        return list(value)
    raise ApiError(400, "input must be a string or an array of strings",
                   param="input")


# ── response shaping ────────────────────────────────────────────────────────


def model_object(entry: dict[str, Any]) -> dict[str, Any]:
    """One `/v1/models` entry from an Ollama `/api/tags` model."""
    name = entry.get("name") or entry.get("model") or ""
    return {"id": str(name), "object": "model",
            "created": _epoch(entry.get("modified_at")),
            "owned_by": "ollamarun"}


def _epoch(value: Any) -> int:
    """Unix seconds from an Ollama ISO timestamp (nanoseconds tolerated)."""
    if not isinstance(value, str):
        return 0
    match = re.match(
        r"^(?P<head>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"
        r"(?:\.(?P<frac>\d+))?"
        r"(?P<tz>Z|[+-]\d{2}:?\d{2})?$", value.strip())
    if match is None:
        return 0
    text = match.group("head").replace(" ", "T")
    fraction = match.group("frac")
    if fraction:
        text += "." + fraction[:6]
    zone = match.group("tz") or ""
    if zone == "Z":
        zone = "+00:00"
    elif zone and ":" not in zone:
        zone = zone[:3] + ":" + zone[3:]
    try:
        return int(datetime.fromisoformat(text + zone).timestamp())
    except ValueError:
        return 0


def openai_tool_calls(calls: list[dict[str, Any]], *,
                      with_index: bool = False) -> list[dict[str, Any]]:
    """Ollama tool calls -> OpenAI `tool_calls` (arguments as JSON string)."""
    out: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        function = (call.get("function") or {}) if isinstance(call, dict) else {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {})
        entry: dict[str, Any] = {
            "id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": str(function.get("name") or ""),
                         "arguments": arguments},
        }
        if with_index:
            entry["index"] = index
        out.append(entry)
    return out


def usage_from(stats: dict[str, Any]) -> dict[str, int]:
    prompt = int(stats.get("prompt_tokens") or 0)
    completion = int(stats.get("completion_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


def embeddings_body(plan: EmbeddingRequest,
                    response: dict[str, Any]) -> dict[str, Any]:
    """The `/v1/embeddings` body from an Ollama `/api/embed` reply.

    `encoding_format: "base64"` is the little-endian float32 packing the
    official SDK decodes; `"float"` returns plain JSON numbers.
    """
    vectors = [vector for vector in (response.get("embeddings") or [])
               if isinstance(vector, list)]
    data = []
    for index, vector in enumerate(vectors):
        values = [float(item) for item in vector]
        if plan.encoding_format == "base64":
            packed = struct.pack(f"<{len(values)}f", *values) if values else b""
            embedding: Any = base64.b64encode(packed).decode("ascii")
        else:
            embedding = values
        data.append({"object": "embedding", "index": index,
                     "embedding": embedding})
    prompt = int(response.get("prompt_eval_count") or 0)
    return {"object": "list", "data": data, "model": plan.model,
            "usage": {"prompt_tokens": prompt, "total_tokens": prompt}}


def finish_reason(stats: dict[str, Any], has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    done = str(stats.get("done_reason") or "").lower()
    return "length" if done == "length" else "stop"


def assistant_message(content: str, thinking: str,
                      calls: list[dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant",
                               "content": None if (not content and calls)
                               else content}
    if thinking:
        message["reasoning_content"] = thinking
    if calls:
        message["tool_calls"] = openai_tool_calls(calls)
    return message


def completion_body(plan: ChatRequest,
                    responses: list[dict[str, Any]]) -> dict[str, Any]:
    """One non-streaming `chat.completion` for one or more emulated choices.

    `n > 1` is emulation: the same request ran `n` sequential backend
    generations, so `usage` is the sum across them rather than one backend
    call's counters.
    """
    choices = []
    for index, response in enumerate(responses):
        message = response.get("message") or {}
        calls = _forwarded_tool_calls(plan, message.get("tool_calls") or [])
        stats = stats_from_final(response)
        choices.append({
            "index": index,
            "message": assistant_message(message.get("content") or "",
                                         message.get("thinking") or "",
                                         calls),
            "finish_reason": finish_reason(stats, bool(calls)),
        })
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": plan.model,
        "choices": choices,
        "usage": aggregate_usage([stats_from_final(response)
                                  for response in responses]),
    }


def _forwarded_tool_calls(plan: ChatRequest,
                          calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`parallel_tool_calls: false` keeps only the first returned call.

    The backend cannot be told to emit a single call, so the flag is honored
    by truncation rather than model steering.
    """
    if plan.parallel_tool_calls is False and len(calls) > 1:
        return calls[:1]
    return calls


def aggregate_usage(stats_list: list[dict[str, Any]]) -> dict[str, int]:
    """The summed usage over per-generation stats (`n > 1` emulation)."""
    prompt = sum(int(stats.get("prompt_tokens") or 0)
                 for stats in stats_list)
    completion = sum(int(stats.get("completion_tokens") or 0)
                     for stats in stats_list)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


def stream_chunk(completion_id: str, created: int, model: str,
                 delta: dict[str, Any], *,
                 finish: Optional[str] = None,
                 index: int = 0) -> dict[str, Any]:
    return {"id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": index, "delta": delta, "finish_reason": finish}]}


def stream_error(message: str, *, code: Optional[str] = None) -> dict[str, Any]:
    return {"error": {"message": message[:400], "type": "api_error",
                      "param": None, "code": code}}


def translate_error(exc: OllamaError, *, model: str = "") -> ApiError:
    """Map an upstream failure onto a sanitized structured API error.

    The client sees a stable message and code; the raw upstream text (which
    can carry backend URLs and internal detail) goes to the debug log only.
    """
    raw = str(exc)
    logger.debug("upstream error (model=%s): %s", model or "-", raw)
    code, message = classify_backend_error(raw, model=model)
    status = 404 if code == "model_not_found" else 502
    return ApiError(status, message,
                    type_="invalid_request_error" if status == 404
                    else "api_error", code=code)


def stream_backend_error(raw: str, *, model: str = "") -> dict[str, Any]:
    """The in-band `{"error": {...}}` SSE event for an upstream failure."""
    logger.debug("upstream stream error (model=%s): %s", model or "-", raw)
    code, message = classify_backend_error(raw, model=model)
    return stream_error(message, code=code)


def classify_backend_error(raw: str, *, model: str = "") -> tuple[str, str]:
    """(code, stable message) for an upstream failure text.

    Shared with the native agent endpoints so both surfaces classify a
    backend failure identically (and keep the raw text out of responses).
    """
    if "HTTP 404" in raw or "not found" in raw.lower():
        message = (f"model {model!r} is not available" if model
                   else "the requested model is not available")
        return "model_not_found", message
    if "cannot reach" in raw:
        return "backend_unreachable", "cannot reach the model backend"
    return "backend_error", "the model backend failed to complete the request"


def model_listed(model: str, entries: list[dict[str, Any]]) -> bool:
    """True when `model` (tag-normalized) appears in an `/api/tags` list."""
    wanted = canonical_model_name(model)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidate = entry.get("name") or entry.get("model") or ""
        if canonical_model_name(str(candidate)) == wanted:
            return True
    return False


def ollama_payload(plan: ChatRequest, *, keepalive: Optional[str] = None
                   ) -> dict[str, Any]:
    """The `/api/chat` body for a validated request."""
    return build_chat_payload(plan.model, plan.messages, stream=plan.stream,
                              fmt=plan.fmt, think=plan.think,
                              keepalive=keepalive,
                              options=plan.options or None,
                              tools=plan.tools or None)


# ── the API object and its routes ───────────────────────────────────────────


class OpenAIApi:
    """Handlers for the OpenAI-compatible endpoints.

    `client_factory()` builds one `OllamaClient` per request: `abort_stream`
    has a single active-stream slot, so a disconnecting SSE client must be
    able to cancel its own generation without touching another request's.
    Model-list and capability lookups are cached briefly to keep the
    endpoints cheap without hiding a freshly pulled model for long.
    """

    MODEL_CACHE_TTL = 5.0
    CAPABILITY_CACHE_TTL = 60.0

    def __init__(self, client_factory: Callable[[], Any], *,
                 config: Optional[dict[str, Any]] = None,
                 timeout: Optional[float] = None,
                 keepalive: Optional[str] = None,
                 artifacts: Optional[Any] = None,
                 tracker: Optional[Any] = None):
        self._client_factory = client_factory
        self.config = dict(config or {})
        #: The artifact store (`artifacts.ArtifactStore`) when serve runs
        #: one: `file` content parts resolve through it. None otherwise.
        self._artifacts = artifacts
        #: The serve model tracker (`helpers.ModelTracker`): feeds the
        #: unload-on-exit of the models this server used. None outside
        #: `ollamarun serve`.
        self._tracker = tracker
        #: Aggregate resolved-input caps (`serve_max_*`); the JSON body cap
        #: bounds the request as sent, these bound what parts may assemble.
        self._limits = input_limits(self.config)
        #: The optional `serve_models` allowlist ([] = every model).
        self._models_allow = list(self.config.get("serve_models") or [])
        self.timeout: float = (timeout if timeout is not None else
                               self.config.get("timeout") or DEFAULT_TIMEOUT)
        self.keepalive = (keepalive if keepalive is not None
                          else self.config.get("keepalive"))
        self._lock = threading.Lock()
        #: Held across an upstream fetch so concurrent cold requests share it
        #: (one `/api/tags` or `/api/show` instead of a stampede).
        self._fetch_lock = threading.Lock()
        self._capability_lock = threading.Lock()
        self._models: Optional[tuple[float, list[dict[str, Any]]]] = None
        self._capability_cache: dict[str, tuple[float, list[str]]] = {}

    # ── models ──────────────────────────────────────────────────────────────

    def list_models(self, request: Request, params: dict[str, str]) -> Response:
        entries = [entry for entry in self._safe_models()
                   if model_allowed(str(entry.get("name")
                                        or entry.get("model") or ""),
                                    self._models_allow)]
        return Response.json({"object": "list",
                              "data": [model_object(entry)
                                       for entry in entries]})

    def get_model(self, request: Request, params: dict[str, str]) -> Response:
        name = params.get("model", "")
        if not model_allowed(name, self._models_allow):
            raise ApiError(404, f"model {name!r} not found",
                           code="model_not_found", param="model")
        wanted = canonical_model_name(name)
        for entry in self._safe_models():
            candidate = entry.get("name") or entry.get("model") or ""
            if canonical_model_name(str(candidate)) == wanted:
                return Response.json(model_object(entry))
        raise ApiError(404, f"model {name!r} not found",
                       code="model_not_found", param="model")

    def _model_entries(self) -> list[dict[str, Any]]:
        cached = self._cached_models()
        if cached is not None:
            return cached
        with self._fetch_lock:
            cached = self._cached_models()
            if cached is not None:
                return cached
            entries = [entry for entry in self._client_factory().tags()
                       if isinstance(entry, dict)]
            with self._lock:
                self._models = (time.monotonic(), entries)
            return entries

    def _cached_models(self) -> Optional[list[dict[str, Any]]]:
        with self._lock:
            cached = self._models
        if cached is not None \
                and time.monotonic() - cached[0] < self.MODEL_CACHE_TTL:
            return cached[1]
        return None

    def _safe_models(self) -> list[dict[str, Any]]:
        try:
            return self._model_entries()
        except OllamaError as exc:
            raise translate_error(exc)

    def _ensure_model(self, model: str) -> None:
        allowed = model_allowed(model, self._models_allow)
        if not allowed or not model_listed(model, self._safe_models()):
            raise ApiError(404, f"model {model!r} not found",
                           code="model_not_found", param="model")

    def _capabilities(self, client: Any, model: str) -> list[str]:
        cached = self._cached_capabilities(model)
        if cached is not None:
            return cached
        with self._capability_lock:
            cached = self._cached_capabilities(model)
            if cached is not None:
                return cached
            # `capabilities()` probes `/api/show` and degrades to an empty
            # list; an unverifiable model passes the vision gate (an
            # unreachable backend fails the chat request with its own 502).
            capabilities = [str(capability)
                            for capability in client.capabilities(model)]
            self._capability_cache[model] = (time.monotonic(), capabilities)
            return capabilities

    def _cached_capabilities(self, model: str) -> Optional[list[str]]:
        cached = self._capability_cache.get(model)
        if cached is not None \
                and time.monotonic() - cached[0] < self.CAPABILITY_CACHE_TTL:
            return cached[1]
        return None

    # ── embeddings ──────────────────────────────────────────────────────────

    def embeddings(self, request: Request, params: dict[str, str]) -> Response:
        plan = parse_embeddings_request(request.json, limits=self._limits)
        client = self._client_factory()
        self._ensure_model(plan.model)
        capabilities = self._capabilities(client, plan.model)
        if capabilities and "embedding" not in capabilities:
            # An empty capability list passes, matching the vision gate:
            # older backends cannot be verified.
            raise ApiError(
                400, f"model {plan.model!r} does not support embeddings",
                param="model", code="model_does_not_support_embeddings")
        if self._tracker is not None:
            self._tracker.note(plan.model)
        try:
            response = client.embed(plan.model, plan.inputs,
                                    dimensions=plan.dimensions)
        except OllamaError as exc:
            raise translate_error(exc, model=plan.model)
        if isinstance(response, dict) and response.get("error"):
            logger.debug("upstream error (model=%s): %s", plan.model,
                         response["error"])
            raise ApiError(502, "the model backend failed to complete "
                                "the request", type_="api_error",
                           code="backend_error")
        return Response.json(embeddings_body(plan, response))

    # ── chat completions ────────────────────────────────────────────────────

    def chat_completions(self, request: Request,
                         params: dict[str, str]) -> Response | StreamResponse:
        plan = parse_chat_request(request.json, store=self._artifacts,
                                  limits=self._limits)
        client = self._client_factory()
        self._ensure_model(plan.model)
        if self._tracker is not None:
            self._tracker.note(plan.model)
        if plan.has_images:
            require_vision(self._capabilities(client, plan.model))
        payload = ollama_payload(plan, keepalive=self.keepalive)

        if not plan.stream:
            responses: list[dict[str, Any]] = []
            for _ in range(plan.n):
                try:
                    response = client.chat_once(payload, timeout=self.timeout)
                except OllamaError as exc:
                    raise translate_error(exc, model=plan.model)
                if isinstance(response, dict) and response.get("error"):
                    logger.debug("upstream error (model=%s): %s", plan.model,
                                 response["error"])
                    raise ApiError(502, "the model backend failed to complete "
                                        "the request", type_="api_error",
                                   code="backend_error")
                responses.append(response)
            return Response.json(completion_body(plan, responses))

        cancel = CancelToken()

        def on_disconnect() -> None:
            cancel.cancel()
            try:
                client.abort_stream()
            except Exception:  # pragma: no cover - defensive
                pass

        def produce(writer: SseWriter) -> None:
            self._stream_completion(plan, client, payload, writer, cancel)

        return StreamResponse(produce=produce, on_disconnect=on_disconnect)

    def _stream_completion(self, plan: ChatRequest, client: Any,
                           payload: dict[str, Any], writer: SseWriter,
                           cancel: CancelToken) -> None:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        results: list[GenerationResult] = []
        # n > 1 is emulated: the choices stream one after another, each with
        # its own index, and the usage chunk aggregates them at the end.
        for choice_index in range(plan.n):
            if writer.broken or cancel.cancelled:
                return
            writer.event(json.dumps(stream_chunk(
                completion_id, created, plan.model,
                {"role": "assistant", "content": ""}, index=choice_index)))
            result = GenerationResult()
            error: Optional[str] = None
            try:
                for chunk in client.chat(payload, timeout=self.timeout,
                                         cancel=cancel):
                    if writer.broken:
                        break
                    if chunk.get("error"):
                        error = str(chunk["error"])
                        break
                    thinking, content = consume_chunk(result, chunk)
                    if thinking:
                        writer.event(json.dumps(stream_chunk(
                            completion_id, created, plan.model,
                            {"reasoning_content": thinking},
                            index=choice_index)))
                    if content:
                        writer.event(json.dumps(stream_chunk(
                            completion_id, created, plan.model,
                            {"content": content}, index=choice_index)))
            except OllamaError as exc:
                error = str(exc)
            if error is not None:
                writer.event(json.dumps(stream_backend_error(error,
                                                             model=plan.model)))
                writer.event("[DONE]")
                return
            calls = _forwarded_tool_calls(plan, result.tool_calls)
            if calls:
                writer.event(json.dumps(stream_chunk(
                    completion_id, created, plan.model,
                    {"tool_calls": openai_tool_calls(calls, with_index=True)},
                    index=choice_index)))
            writer.event(json.dumps(stream_chunk(
                completion_id, created, plan.model, {},
                finish=finish_reason(result.stats, bool(calls)),
                index=choice_index)))
            results.append(result)
        if plan.include_usage:
            writer.event(json.dumps({
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": plan.model,
                "choices": [],
                "usage": aggregate_usage([result.stats for result in results])}))
        writer.event("[DONE]")


def build_router(client_factory: Callable[[], Any], *,
                 config: Optional[dict[str, Any]] = None,
                 timeout: Optional[float] = None,
                 keepalive: Optional[str] = None,
                 artifacts: Optional[Any] = None,
                 tracker: Optional[Any] = None) -> Router:
    """The `/v1` router: models, one model, embeddings, chat completions."""
    api = OpenAIApi(client_factory, config=config, timeout=timeout,
                    keepalive=keepalive, artifacts=artifacts, tracker=tracker)
    router = Router()
    router.add("GET", "/v1/models", api.list_models)
    router.add("GET", "/v1/models/{model}", api.get_model)
    router.add("POST", "/v1/embeddings", api.embeddings)
    router.add("POST", "/v1/chat/completions", api.chat_completions)
    return router
