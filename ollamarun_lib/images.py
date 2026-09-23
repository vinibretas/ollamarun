"""Image encoding: base64 embedding for vision-capable models."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any, Optional


def encode_image(path: Path, storage: str = "embed") -> dict[str, Any]:
    """Read an image into an API-ready block; raises OSError/ValueError."""
    if not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    if not data:
        raise ValueError(f"{path}: empty file")
    record: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "sha1": hashlib.sha1(data).hexdigest(),
        "size": len(data),
    }
    if storage == "embed" or len(data) <= 8 * 1024 * 1024:
        record["b64"] = base64.b64encode(data).decode("ascii")
    return record


def image_payload(records: Optional[list[dict[str, Any]]]) -> list[str]:
    out: list[str] = []
    for record in records or []:
        if record.get("b64"):
            out.append(record["b64"])
        elif record.get("path") and Path(record["path"]).is_file():
            out.append(base64.b64encode(Path(record["path"]).read_bytes()).decode("ascii"))
    return out
