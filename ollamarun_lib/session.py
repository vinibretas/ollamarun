"""Session storage: the in-memory conversation model plus atomic on-disk
read/write of the per-conversation JSON files and `index.json`, and the
three export formats (markdown, json, html).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from .client import OllamaError
from .constants import INDEX_PATH, SCHEMA_VERSION, SESSIONS_DIR, THEME
from .helpers import naive_title, new_session_id, now_iso
from .images import image_payload
from .renderer import render_transcript


@dataclass
class Session:
    id: str
    model: str = ""
    title: str = ""
    host: str = ""
    remote: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    tags: list[str] = field(default_factory=list)
    system: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    think: Optional[Any] = None
    keepalive: Optional[str] = None
    format: Optional[Any] = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    tool_allow: list[str] = field(default_factory=list)
    tools_disabled: list[str] = field(default_factory=list)
    tts_auto: Optional[bool] = None
    tts_voice: Optional[str] = None
    tts_lang: Optional[str] = None
    schema: int = SCHEMA_VERSION

    @classmethod
    def create(cls, model: str, host: str, **kwargs: Any) -> "Session":
        stamp = now_iso()
        session = cls(id=new_session_id(), model=model, host=host,
                      created_at=stamp, updated_at=stamp, **kwargs)
        session.stats = {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0,
                         "duration_s": 0.0}
        return session

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def display_title(self) -> str:
        return self.title or naive_title(self.messages[0].get("content", "")) if self.messages \
            else (self.title or "new session")

    def touch(self) -> None:
        self.updated_at = now_iso()

    def add_message(self, role: str, content: str, **extra: Any) -> dict[str, Any]:
        message = {"role": role, "content": content, "ts": now_iso()}
        for key, value in extra.items():
            if value is not None:
                message[key] = value
        self.messages.append(message)
        self.touch()
        return message

    def message_payloads(self, *, with_tools: bool = False) -> list[dict[str, Any]]:
        """Messages as the API expects them.

        `with_tools` passes assistant `tool_calls` and `role: tool` results
        through; without it those messages are omitted (and an empty assistant
        tool-call turn is dropped entirely) so a model that was never given
        tools never sees a tool role.
        """
        payloads: list[dict[str, Any]] = []
        for message in self.messages:
            role = message.get("role")
            if role == "tool":
                if not with_tools:
                    continue
                payload: dict[str, Any] = {"role": "tool",
                                           "content": message.get("content", "")}
                if message.get("tool_name"):
                    payload["tool_name"] = message["tool_name"]
                payloads.append(payload)
                continue
            if message.get("cancelled") and not message.get("content"):
                continue
            content = message.get("content", "")
            tool_calls = message.get("tool_calls")
            if not content and tool_calls and not with_tools:
                continue
            payload = {"role": role, "content": content}
            images = image_payload(message.get("images"))
            if images:
                payload["images"] = images
            if with_tools and tool_calls:
                payload["tool_calls"] = tool_calls
            payloads.append(payload)
        return payloads

    def last_user_index(self) -> Optional[int]:
        for index in range(len(self.messages) - 1, -1, -1):
            if self.messages[index].get("role") == "user":
                return index
        return None

    def record_turn(self, stats: dict[str, Any], count_turn: bool = True) -> None:
        if count_turn:
            self.stats["turns"] = self.stats.get("turns", 0) + 1
        self.stats["prompt_tokens"] = self.stats.get("prompt_tokens", 0) + (stats.get("prompt_tokens") or 0)
        self.stats["completion_tokens"] = self.stats.get("completion_tokens", 0) + (stats.get("completion_tokens") or 0)
        self.stats["duration_s"] = self.stats.get("duration_s", 0.0) + (stats.get("total_duration") or 0.0)
        if stats.get("eval_tps"):
            self.stats["last_eval_tps"] = stats["eval_tps"]


def _index_entry(session: Session) -> dict[str, Any]:
    return {
        "id": session.id,
        "title": session.display_title(),
        "model": session.model,
        "host": session.host,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "messages": len(session.messages),
        "turns": session.stats.get("turns", 0),
        "prompt_tokens": session.stats.get("prompt_tokens", 0),
        "completion_tokens": session.stats.get("completion_tokens", 0),
        "tags": list(session.tags),
    }


class SessionStore:
    def __init__(self, sessions_dir: Path = SESSIONS_DIR,
                 index_path: Path = INDEX_PATH):
        self.dir = Path(sessions_dir)
        self.index_path = Path(index_path)

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def save(self, session: Session) -> None:
        session.touch()
        self.dir.mkdir(parents=True, exist_ok=True)
        target = self._path(session.id)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(session.to_dict(), indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        os.replace(tmp, target)
        self._update_index(session)

    def _update_index(self, session: Session) -> None:
        entries = self._read_index()
        entry = _index_entry(session)
        by_id = {item["id"]: item for item in entries}
        by_id[session.id] = entry
        rows = sorted(by_id.values(),
                      key=lambda item: (item.get("updated_at") or "", item.get("id") or ""),
                      reverse=True)
        self._write_index(rows)

    def _read_index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return []
        if data.get("schema") != SCHEMA_VERSION:
            return []
        return list(data.get("sessions") or [])

    def _write_index(self, rows: list[dict[str, Any]]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"schema": SCHEMA_VERSION, "sessions": rows},
                                  indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.index_path)

    def rebuild_index(self) -> list[dict[str, Any]]:
        rows = []
        if self.dir.exists():
            for path in sorted(self.dir.glob("*.json")):
                try:
                    session = Session.from_dict(json.loads(path.read_text(encoding="utf-8")))
                except (ValueError, OSError, TypeError):
                    continue
                rows.append(_index_entry(session))
        rows.sort(key=lambda item: (item.get("updated_at") or "", item.get("id") or ""),
                  reverse=True)
        self._write_index(rows)
        return rows

    def entries(self) -> list[dict[str, Any]]:
        rows = self._read_index()
        if not rows and self.dir.exists() and any(self.dir.glob("*.json")):
            rows = self.rebuild_index()
        return rows

    def load(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.exists():
            raise FileNotFoundError(session_id)
        return Session.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def resolve(self, ref: str) -> Optional[Session]:
        """Resolve an exact id, id prefix, or title (case-insensitive)."""
        if not ref:
            return None
        ref = ref.strip()
        if self._path(ref).exists():
            return self.load(ref)
        try:
            matches = ([p.stem for p in self.dir.glob(f"{ref}*.json")]
                       if re.fullmatch(r"[\w.-]+", ref) else [])
        except OSError:
            matches = []
        if len(matches) == 1:
            return self.load(matches[0])
        lowered = ref.lower()
        for entry in self.entries():
            if entry.get("title", "").lower() == lowered:
                return self.load(entry["id"])
        for entry in self.entries():
            if lowered in entry.get("title", "").lower():
                return self.load(entry["id"])
        return None

    def latest(self, model: Optional[str] = None) -> Optional[Session]:
        for entry in self.entries():
            if model is None or entry.get("model") == model:
                try:
                    return self.load(entry["id"])
                except (FileNotFoundError, ValueError):
                    continue
        return None

    def delete(self, ref: str, use_trash: bool = True) -> Session:
        session = self.resolve(ref)
        if session is None:
            raise FileNotFoundError(ref)
        path = self._path(session.id)
        if use_trash:
            if not shutil.which("trash-put"):
                raise OllamaError("trash-put not found; refusing to delete without a trash")
            result = subprocess.run(["trash-put", str(path)], capture_output=True, text=True)
            if result.returncode != 0:
                raise OllamaError(f"trash-put failed: {result.stderr.strip()}")
        else:
            path.unlink(missing_ok=True)
        rows = [entry for entry in self.entries() if entry["id"] != session.id]
        self._write_index(rows)
        return session

    def search(self, query: str, limit: int = 50) -> list[tuple[Session, str]]:
        needle = query.lower()
        results: list[tuple[Session, str]] = []
        if not self.dir.exists():
            return results
        for entry in self.entries():
            try:
                session = self.load(entry["id"])
            except (FileNotFoundError, ValueError):
                continue
            haystacks = [session.display_title(), session.model, " ".join(session.tags)]
            haystacks += [message.get("content", "") for message in session.messages]
            haystacks += [message.get("thinking", "") for message in session.messages]
            for haystack in haystacks:
                position = (haystack or "").lower().find(needle)
                if position >= 0:
                    start = max(0, position - 40)
                    snippet = haystack[start:position + len(query) + 60].replace("\n", " ")
                    prefix = "…" if start else ""
                    results.append((session, f"{prefix}{snippet}…"))
                    break
            if len(results) >= limit:
                break
        return results


def export_markdown(session: Session) -> str:
    lines = [
        f"# {session.display_title()}",
        "",
        f"- **id:** `{session.id}`",
        f"- **model:** `{session.model}`",
        f"- **created:** {session.created_at}",
        f"- **updated:** {session.updated_at}",
    ]
    if session.tags:
        lines.append(f"- **tags:** {', '.join(session.tags)}")
    if session.system:
        lines += ["", "## System", "", session.system]
    for message in session.messages:
        role = message.get("role", "?")
        heading = {"user": "User", "assistant": "Assistant"}.get(role, role.title())
        lines += ["", f"## {heading}", ""]
        for image in message.get("images") or []:
            lines.append(f"![{image.get('name', 'image')}]({image.get('path', '')})")
        content = message.get("content") or ""
        if message.get("thinking"):
            lines += ["", "<details><summary>thinking</summary>", "",
                      message["thinking"], "", "</details>"]
        lines.append(content)
        if message.get("tool_calls"):
            lines += ["", "```json",
                      json.dumps(message["tool_calls"], indent=2, ensure_ascii=False),
                      "```"]
    stats = session.stats or {}
    if stats:
        lines += ["", "---", "",
                  f"_{stats.get('turns', 0)} turns · "
                  f"{stats.get('prompt_tokens', 0)} prompt tokens · "
                  f"{stats.get('completion_tokens', 0)} completion tokens · "
                  f"{stats.get('duration_s', 0):.0f}s total_"]
    return "\n".join(lines).rstrip() + "\n"


def export_json(session: Session) -> str:
    return json.dumps(session.to_dict(), indent=2, ensure_ascii=False) + "\n"


def export_html(session: Session, width: int = 110) -> str:
    recorder = Console(record=True, width=width, theme=THEME)
    render_transcript(session, recorder, show_stats=True, pager=False)
    return recorder.export_html(inline_styles=True)
