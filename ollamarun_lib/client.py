"""Ollama HTTP client: wraps `/api/chat`, `/api/generate`, `/api/embed`,
`/api/ps`, `/api/create` etc. over raw `requests`, streaming response bodies
line-by-line and supporting cancellation of an in-flight stream.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable, Iterator, Optional

import requests

from .constants import DEFAULT_HOST, DEFAULT_TIMEOUT
from .helpers import CancelToken, canonical_model_name, normalize_host


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Minimal raw-HTTP client: full control and llm-jukebox friendly."""

    def __init__(self, host: str = DEFAULT_HOST, *, insecure: bool = False,
                 timeout: int = DEFAULT_TIMEOUT):
        self.base = normalize_host(host)
        self.insecure = insecure
        self.timeout = timeout
        self.http = requests.Session()
        # The stream being read right now, so another thread can close it and
        # unblock the reader (see `abort_stream`).
        self._stream_lock = threading.Lock()
        self._active_stream: Optional[requests.Response] = None

    def _url(self, path: str) -> str:
        return self.base + path

    def _request(self, method: str, path: str, *, payload: Optional[dict] = None,
                 stream: bool = False, timeout: Optional[float] = None):
        try:
            response = self.http.request(
                method, self._url(path), json=payload, stream=stream,
                timeout=timeout or self.timeout, verify=not self.insecure)
        except requests.RequestException as exc:
            raise OllamaError(f"cannot reach Ollama at {self.base}: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text.strip()
            try:
                detail = response.json().get("error") or detail
            except ValueError:
                pass
            response.close()
            raise OllamaError(f"HTTP {response.status_code} on {path}: {detail}")
        return response

    def _json(self, method: str, path: str, payload: Optional[dict] = None,
              timeout: Optional[float] = None) -> Any:
        response = self._request(method, path, payload=payload, timeout=timeout)
        try:
            return response.json()
        except ValueError as exc:
            raise OllamaError(f"{path}: invalid JSON response") from exc
        finally:
            response.close()

    def version(self, timeout: Optional[float] = None) -> str:
        return self._json("GET", "/api/version", timeout=timeout).get("version", "?")

    def tags(self) -> list[dict[str, Any]]:
        return self._json("GET", "/api/tags").get("models", [])

    def show(self, model: str) -> dict[str, Any]:
        return self._json("POST", "/api/show", {"model": model})

    def ps(self) -> list[dict[str, Any]]:
        return self._json("GET", "/api/ps").get("models", [])

    def resident_models(self) -> list[str]:
        names: list[str] = []
        for entry in self.ps():
            name = entry.get("name") or entry.get("model")
            if name:
                names.append(name)
        return names

    def is_resident(self, model: str) -> bool:
        """True when model is currently loaded (tag-normalized name compare)."""
        if not model:
            return False
        wanted = canonical_model_name(model)
        try:
            return any(canonical_model_name(name) == wanted
                       for name in self.resident_models())
        except OllamaError:
            return False

    def capabilities(self, model: str) -> list[str]:
        try:
            return list(self.show(model).get("capabilities") or [])
        except OllamaError:
            return []

    def model_exists(self, model: str) -> bool:
        try:
            self.show(model)
            return True
        except OllamaError:
            return False

    def abort_stream(self) -> bool:
        """Close the stream being read right now (safe no-op when idle).

        Called from the UI thread to make a blocked `iter_lines()` return, so a
        cancelled turn stops immediately instead of waiting for the next chunk.
        """
        with self._stream_lock:
            response = self._active_stream
        if response is None:
            return False
        try:
            response.close()
        except Exception:  # closing a half-read response must never raise
            pass
        return True

    def stream_ndjson(self, path: str, payload: dict,
                      timeout: Optional[float] = None,
                      cancel: Optional[CancelToken] = None
                      ) -> Iterator[dict[str, Any]]:
        response = self._request("POST", path, payload=payload, stream=True,
                                 timeout=timeout or self.timeout)
        with self._stream_lock:
            self._active_stream = response
        try:
            for raw in response.iter_lines(decode_unicode=False):
                if cancel is not None and cancel.cancelled:
                    return
                if not raw:
                    continue
                try:
                    yield json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise OllamaError(f"bad stream chunk from {path}: {exc}") from exc
        except requests.RequestException as exc:
            if cancel is not None and cancel.cancelled:
                return  # the abort we asked for, not an error
            raise OllamaError(f"stream interrupted: {exc}") from exc
        except Exception as exc:
            # Closing the response under a blocked read surfaces as assorted
            # low-level errors (urllib3/socket/ValueError); with a cancel
            # requested they all just mean "cancelled".
            if cancel is not None and cancel.cancelled:
                return
            raise OllamaError(f"stream interrupted: {exc}") from exc
        finally:
            with self._stream_lock:
                self._active_stream = None
            response.close()

    def chat(self, payload: dict, timeout: Optional[float] = None,
             cancel: Optional[CancelToken] = None) -> Iterator[dict[str, Any]]:
        payload = {**payload, "stream": True}
        return self.stream_ndjson("/api/chat", payload, timeout=timeout, cancel=cancel)

    def chat_once(self, payload: dict, timeout: Optional[float] = None) -> dict[str, Any]:
        payload = {**payload, "stream": False}
        return self._json("POST", "/api/chat", payload)

    def embed(self, model: str, inputs: list[str], *, truncate: bool = True,
              dimensions: Optional[int] = None,
              options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "input": inputs, "truncate": truncate}
        if dimensions:
            payload["dimensions"] = dimensions
        if options:
            payload["options"] = options
        return self._json("POST", "/api/embed", payload)

    def create_model(self, name: str, from_model: str, *, system: Optional[str] = None,
                     parameters: Optional[dict[str, Any]] = None,
                     progress: Optional[Callable[[str], None]] = None) -> None:
        payload: dict[str, Any] = {"model": name, "from": from_model, "stream": True}
        if system:
            payload["system"] = system
        if parameters:
            payload["parameters"] = parameters
        for event in self.stream_ndjson("/api/create", payload):
            if event.get("error"):
                raise OllamaError(str(event["error"]))
            status = event.get("status")
            if status and progress:
                progress(status)

    def unload(self, model: str) -> None:
        self._json("POST", "/api/generate", {"model": model, "keep_alive": 0})

    def load(self, model: str, *, keepalive: Optional[str] = None,
             options: Optional[dict[str, Any]] = None) -> None:
        """Warms a model into memory without generating anything."""
        payload: dict[str, Any] = {"model": model}
        if keepalive:
            payload["keep_alive"] = keepalive
        if options:
            payload["options"] = options
        self._json("POST", "/api/generate", payload)

    def pull_model(self, model: str,
                   progress: Optional[Callable[[dict[str, Any]], None]] = None) -> None:
        """Downloads a model from the registry, streaming progress events."""
        for event in self.stream_ndjson("/api/pull", {"model": model, "stream": True}):
            if event.get("error"):
                raise OllamaError(str(event["error"]))
            if progress:
                progress(event)


def build_chat_payload(model: str, messages: list[dict[str, Any]], *, stream: bool = True,
                       fmt: Optional[Any] = None, think: Optional[Any] = None,
                       keepalive: Optional[str] = None,
                       options: Optional[dict[str, Any]] = None,
                       tools: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    if fmt:
        payload["format"] = fmt
    if think is not None:
        payload["think"] = think
    if keepalive:
        payload["keep_alive"] = keepalive
    if options:
        payload["options"] = options
    if tools:
        payload["tools"] = tools
    return payload
