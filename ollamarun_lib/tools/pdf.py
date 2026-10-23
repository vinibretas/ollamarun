"""PDF plumbing: text extraction, page selection, rasterizing and vision reads.

The text layer comes from pypdf, then poppler's `pdftotext`; a scan is handled
by rasterizing pages with `pdftoppm` and transcribing them through a vision
model (`pick_vision_model()` chooses it: pinned, the session model, or the best
installed one).
"""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from ..client import OllamaError
from ..helpers import canonical_model_name
from .html import _normalize_text
from .spec import MAX_PDF_VISION_PAGES, ToolError


MAX_PDF_CHARS = 40_000
MAX_PDF_PAGES = 50
#: Rasterization resolution: small print stays legible, and a page still fits
#: one image part comfortably.
PDF_VISION_DPI = 150
#: A PDF this large is refused before it is parsed.
MAX_PDF_BYTES = 32 * 1024 * 1024
#: Per-page vision request timeout: a cold model load can be slow, and the
#: tool should not give up on the first page because of it.
PDF_VISION_TIMEOUT = 300.0
#: Vision-model families tried in this order when the session model cannot
#: see: a family earlier in the list wins, its smallest tag first.
PDF_VISION_PREFERENCE = (
    "qwen3-vl", "qwen2.5vl", "qwen2.5-vl", "qwen3.5-vl", "minicpm-v",
    "llama3.2-vision", "gemma4", "gemma3", "moondream", "llava",
    "internvl", "pixtral", "granite-vision",
)
#: Last-resort transcription instruction; the editable copy lives in
#: `defs/prompts/pdf_vision.md` and the REPL injects it (`ToolContext`).
PDF_VISION_PROMPT = ("Transcribe this page into plain text, preserving "
                     "headings, lists and reading order.")


def extract_pdf_text(data: bytes, max_chars: int = MAX_PDF_CHARS,
                     pages: Optional[list[int]] = None) -> Optional[str]:
    """Text of a PDF from its bytes, or None when nothing can extract it.

    An empty string means the PDF was read but carries no extractable text
    (typically a scan). `pages` is a list of 0-based page indices to read;
    None reads from the first page up to MAX_PDF_PAGES.
    """
    if not data:
        return None
    text = _pdf_text_reader(data, max_chars, pages)
    if text is None:
        text = _pdf_text_tool(data, max_chars, pages)
    return text


def _pdf_text_reader(data: bytes, max_chars: int,
                     pages: Optional[list[int]] = None) -> Optional[str]:
    """pypdf-based extraction (in-process); None when pypdf is missing/broken."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        reader = PdfReader(io.BytesIO(data))
        wanted = set(pages) if pages is not None else None
        chunks: list[str] = []
        total = 0
        for index, page in enumerate(reader.pages):
            if wanted is not None:
                if index not in wanted:
                    continue
            elif index >= MAX_PDF_PAGES:
                break
            page_text = page.extract_text() or ""
            chunks.append(page_text)
            total += len(page_text)
            if total >= max_chars:
                break
    except Exception:
        return None
    return _normalize_text("\n\n".join(chunks))[:max_chars]


def _pdf_text_tool(data: bytes, max_chars: int,
                   pages: Optional[list[int]] = None) -> Optional[str]:
    """pdftotext fallback; None when the binary is missing or fails."""
    binary = shutil.which("pdftotext")
    if not binary:
        return None
    ranges = _page_ranges(pages) if pages is not None else [(None, None)]
    chunks: list[str] = []
    total = 0
    for first, last in ranges:
        command = [binary, "-layout"]
        if first is not None and last is not None:
            command += ["-f", str(first + 1), "-l", str(last + 1)]
        command += ["-", "-"]
        try:
            proc = subprocess.run(command, input=data, capture_output=True,
                                  timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        chunk = proc.stdout.decode("utf-8", errors="replace")
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    return _normalize_text("\n\n".join(chunks))[:max_chars]


def _page_ranges(pages: list[int]) -> list[tuple[Optional[int], Optional[int]]]:
    """Contiguous runs of 0-based pages: [0, 1, 2, 6] -> [(0, 2), (6, 6)]."""
    ranges: list[tuple[Optional[int], Optional[int]]] = []
    for page in sorted(set(pages)):
        if ranges and ranges[-1][1] is not None and page == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], page)
        else:
            ranges.append((page, page))
    return ranges


def parse_page_spec(spec: Any, total: Optional[int] = None,
                    cap: int = MAX_PDF_VISION_PAGES) -> Optional[list[int]]:
    """Parse a page selection ("1-3,7", "all") into 0-based page indices.

    Empty/None/"all" returns None (the caller picks its default window). A
    selection is sorted, de-duplicated, dropped of pages past `total` and
    capped at `cap` entries; anything unparseable, or a selection with no
    page left, raises ToolError so the model gets a usable message.
    """
    raw = str(spec if spec is not None else "").strip().lower()
    if not raw or raw == "all":
        return None
    selected: set[int] = set()
    for token in raw.replace(" ", "").split(","):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if match is None:
            raise ToolError(f"bad pages value {spec!r} (use e.g. 1-3,7 or all)")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if first < 1 or last < first:
            raise ToolError(f"bad pages range {token!r} (pages start at 1)")
        selected.update(range(first - 1, last))
    if total is not None:
        selected = {page for page in selected if page < total}
    if not selected:
        raise ToolError(f"pages {spec!r} selects no page of this PDF")
    return sorted(selected)[:max(1, cap)]


def select_pdf_pages(spec: Any, total: Optional[int], cap: int) -> list[int]:
    """The 0-based pages to read: the spec, or the first `cap` by default."""
    pages = parse_page_spec(spec, total, cap)
    if pages is not None:
        return pages
    count = min(total, cap) if total else cap
    return list(range(max(1, count)))


def pdf_page_count(data: bytes) -> Optional[int]:
    """How many pages a PDF has (pypdf, then `pdfinfo`), or None."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(io.BytesIO(data)).pages)
    except Exception:
        pass
    binary = shutil.which("pdfinfo")
    if not binary:
        return None
    try:
        proc = subprocess.run([binary, "-"], input=data, capture_output=True,
                              timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout.decode("utf-8", errors="replace")
    match = re.search(r"^Pages:\s+(\d+)\s*$", text, re.M)
    return int(match.group(1)) if match else None


def render_pdf_page(data: bytes, page: int, directory: str,
                    dpi: int = PDF_VISION_DPI) -> Path:
    """Rasterize one 0-based page to a JPEG file under `directory`."""
    binary = shutil.which("pdftoppm")
    if not binary:
        raise ToolError("reading a scan needs pdftoppm (install poppler-utils)")
    prefix = os.path.join(directory, f"page-{page + 1}")
    command = [binary, "-jpeg", "-r", str(max(72, min(int(dpi), 300))),
               "-jpegopt", "quality=90", "-singlefile",
               "-f", str(page + 1), "-l", str(page + 1), "-", prefix]
    try:
        proc = subprocess.run(command, input=data, capture_output=True,
                              timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolError(f"could not render page {page + 1}: {exc}") from exc
    path = Path(prefix + ".jpg")
    if proc.returncode != 0 or not path.is_file():
        detail = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        first = detail.splitlines()[0] if detail else ""
        raise ToolError(f"could not render page {page + 1}"
                        + (f": {first}" if first else ""))
    return path


def _model_capabilities(client: Any, model: str,
                        show: Optional[Callable[[str], dict[str, Any]]]
                        ) -> list[str]:
    """One model's capability list; [] when the backend cannot answer."""
    try:
        data = show(model) if show is not None else client.show(model)
    except Exception:
        return []
    return [str(capability) for capability in (data.get("capabilities") or [])]


def _preference_rank(model: str) -> int:
    name = canonical_model_name(model).lower()
    for rank, family in enumerate(PDF_VISION_PREFERENCE):
        if family in name:
            return rank
    return len(PDF_VISION_PREFERENCE)


def _model_size(entry: dict[str, Any]) -> int:
    try:
        return int(entry.get("size") or 0)
    except (TypeError, ValueError):
        return 0


def vision_models_installed(client: Any,
                            show: Optional[Callable[[str], dict[str, Any]]]
                            ) -> list[dict[str, Any]]:
    """Installed models that report the `vision` capability, best first."""
    found: list[dict[str, Any]] = []
    for entry in client.tags():
        name = str(entry.get("name") or entry.get("model") or "")
        if not name:
            continue
        capabilities = _model_capabilities(client, name, show)
        if "vision" in capabilities:
            found.append({"name": name, "size": _model_size(entry),
                          "capabilities": capabilities})
    return sorted(found, key=lambda item: (_preference_rank(item["name"]),
                                           item["size"], item["name"]))


def pick_vision_model(client: Any, *, current: Optional[str],
                      pinned: Optional[str] = None,
                      show: Optional[Callable[[str], dict[str, Any]]] = None,
                      allow_switch: bool = True) -> tuple[str, str]:
    """The model that reads a scanned PDF, with a short reason.

    Order: the pinned `pdf_vision_model`, then the session model when it can
    see, then the best installed vision model (preference list, smallest
    tag). With `allow_switch` off a model other than the session's is
    refused instead of loaded. Raises ToolError when nothing can be used.
    """
    chosen = ""
    reason = ""
    if pinned:
        if not client.model_exists(pinned):
            raise ToolError(
                f"the configured pdf_vision_model {pinned!r} is not "
                f"installed; pull it or clear that config key")
        chosen, reason = pinned, "pinned by pdf_vision_model"
    elif current and "vision" in _model_capabilities(client, current, show):
        chosen, reason = current, "the session model can read images"
    else:
        try:
            installed = vision_models_installed(client, show)
        except OllamaError as exc:
            raise ToolError(f"could not list the installed models: {exc}") from exc
        if not installed:
            raise ToolError(
                "this PDF has no text layer (a scan) and no vision model is "
                "installed; pull one (for example `ollamarun pull qwen3-vl:2b`) "
                "or set pdf_vision_model in the config")
        chosen, reason = installed[0]["name"], "picked automatically"
    if (not allow_switch and current
            and canonical_model_name(chosen) != canonical_model_name(current)):
        raise ToolError(
            f"this PDF has no text layer (a scan) and {current} cannot read "
            f"images; using {chosen} would unload it, which pdf_vision_switch "
            f"forbids — set pdf_vision_switch: true or pin a vision session "
            f"model")
    return chosen, reason


def vision_transcribe(client: Any, model: str, images: list[str], prompt: str,
                      *, timeout: float, cancel: Optional[Any] = None,
                      capabilities: Optional[list[str]] = None) -> str:
    """One vision-model chat call over page images; returns its text."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images}],
        "options": {"temperature": 0, "num_predict": 2048},
    }
    if capabilities and "thinking" in capabilities:
        payload["think"] = False
    chunks: list[str] = []
    try:
        for event in client.chat(payload, timeout=timeout, cancel=cancel):
            if event.get("error"):
                raise ToolError(f"vision model {model}: {event['error']}")
            message = event.get("message") or {}
            piece = message.get("content") or ""
            if piece:
                chunks.append(piece)
    except OllamaError as exc:
        raise ToolError(f"vision model {model} failed: {exc}") from exc
    return "".join(chunks).strip()
