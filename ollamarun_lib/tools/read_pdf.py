"""The `read_pdf` tool: the text layer first, page images when it is a scan."""
from __future__ import annotations

import shutil
import tempfile
import urllib.parse
from typing import Any, Optional

import requests

from ..client import OllamaError
from ..helpers import canonical_model_name
from ..images import encode_image
from .net import BROWSER_HEADERS, FETCH_TIMEOUT, _request_error
from .pdf import (MAX_PDF_BYTES, MAX_PDF_CHARS, MAX_PDF_PAGES,
                  PDF_VISION_PROMPT, PDF_VISION_TIMEOUT,
                  _model_capabilities, extract_pdf_text,
                  pdf_page_count, pick_vision_model, render_pdf_page,
                  select_pdf_pages, vision_transcribe)
from .spec import (MAX_PDF_VISION_PAGES, ToolCancelled, ToolContext, ToolError,
                   _clamp_timeout)
from .workspace import _resolve_in_workspace


def _pdf_pages_label(total: Optional[int]) -> str:
    if total:
        return f"{total} page{'s' if total != 1 else ''}"
    return "page count unknown"


def _check_read_pdf_cancel(ctx: ToolContext) -> None:
    cancel = getattr(ctx, "cancel", None)
    if cancel is not None and cancel.cancelled:
        raise ToolCancelled("read_pdf cancelled: the turn was stopped")


def _read_pdf_source(args: dict[str, Any],
                     ctx: ToolContext) -> tuple[str, bytes]:
    """The PDF bytes and a label, from `path` or `url` (exactly one)."""
    raw_path = str(args.get("path") or "").strip()
    raw_url = str(args.get("url") or "").strip()
    if bool(raw_path) == bool(raw_url):
        raise ToolError("give exactly one of 'path' or 'url'")
    if not raw_url:
        path = _resolve_in_workspace(raw_path, ctx)
        if not path.is_file():
            raise ToolError(f"not a file: {path}")
        try:
            if path.stat().st_size > MAX_PDF_BYTES:
                raise ToolError(
                    f"{path} is larger than the "
                    f"{MAX_PDF_BYTES // (1024 * 1024)} MB PDF limit")
            return str(path), path.read_bytes()
        except OSError as exc:
            raise ToolError(f"could not read {path}: {exc}") from exc
    parsed = urllib.parse.urlsplit(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolError("url must be an absolute http:// or https:// URL")
    timeout = _clamp_timeout(args.get("timeout"), min(ctx.timeout, FETCH_TIMEOUT))
    try:
        response = requests.get(raw_url, timeout=timeout,
                                headers=dict(BROWSER_HEADERS),
                                allow_redirects=True, stream=True)
    except requests.RequestException as exc:
        raise ToolError(f"could not fetch {raw_url}: {_request_error(exc)}") from exc
    collected = bytearray()
    try:
        if response.status_code >= 400:
            raise ToolError(f"could not fetch {raw_url}: "
                            f"HTTP {response.status_code}")
        for chunk in response.iter_content(64 * 1024):
            collected.extend(chunk)
            if len(collected) > MAX_PDF_BYTES:
                raise ToolError(
                    f"{raw_url} is larger than the "
                    f"{MAX_PDF_BYTES // (1024 * 1024)} MB PDF limit")
    finally:
        response.close()
    return raw_url, bytes(collected)


def _pdf_text_result(label: str, text: Optional[str],
                     total: Optional[int]) -> str:
    """The text-layer answer for one read_pdf call."""
    head = f"{label} ({_pdf_pages_label(total)})"
    if text is None:
        return (f"{head}; its text could not be extracted — the file may be "
                f"encrypted or malformed, or this machine has no PDF text "
                f"reader (install poppler-utils or pypdf)")
    if not text.strip():
        return (f"{head}; no text layer — the pages are images (a scan), not "
                f"selectable text")
    return f"{head}, text layer:\n{text}"


def _pdf_vision_result(ctx: ToolContext, data: bytes, label: str,
                       args: dict[str, Any], total: Optional[int],
                       max_chars: int) -> str:
    """Rasterize the selected pages and have a vision model read them."""
    if not ctx.pdf_vision:
        return (f"{label} ({_pdf_pages_label(total)}); no text layer — a scan. "
                f"Reading pages with a vision model is disabled "
                f"(pdf_vision: false)")
    if ctx.client is None:
        raise ToolError(
            "reading a scan needs a vision model, and no Ollama client is "
            "available here; set pdf_vision: false to skip that path")
    current = ctx.current_model() if ctx.current_model is not None else None
    model, reason = pick_vision_model(
        ctx.client, current=current, pinned=ctx.pdf_vision_model,
        show=ctx.model_show, allow_switch=ctx.pdf_vision_switch)
    capabilities = _model_capabilities(ctx.client, model, ctx.model_show)
    try:
        page_cap = int(ctx.pdf_vision_pages)
    except (TypeError, ValueError):
        page_cap = MAX_PDF_VISION_PAGES
    page_cap = max(1, min(page_cap, MAX_PDF_VISION_PAGES))
    pages = select_pdf_pages(args.get("pages"), total, page_cap)
    prompt = (ctx.vision_prompt or "").strip() or PDF_VISION_PROMPT
    swapped = bool(current and canonical_model_name(current)
                   != canonical_model_name(model))
    unloaded_current = False
    ctx.title_note = (f"{current} → {model}" if swapped else f"vision: {model}")
    directory = tempfile.mkdtemp(prefix="ollamarun-pdf-")
    timeout = max(ctx.timeout, PDF_VISION_TIMEOUT)
    sections: list[str] = []
    produced = 0
    try:
        if swapped:
            try:
                if ctx.client.is_resident(current):
                    ctx.client.unload(current)
                    unloaded_current = True
            except OllamaError:
                pass
        for page in pages:
            _check_read_pdf_cancel(ctx)
            image_path = render_pdf_page(data, page, directory)
            try:
                record = encode_image(image_path)
            finally:
                try:
                    image_path.unlink()
                except OSError:
                    pass
            page_text = vision_transcribe(
                ctx.client, model, [record["b64"]], prompt, timeout=timeout,
                cancel=ctx.cancel, capabilities=capabilities)
            _check_read_pdf_cancel(ctx)
            if page_text:
                sections.append(f"--- page {page + 1} ---\n{page_text}")
                produced += len(page_text)
            else:
                sections.append(f"--- page {page + 1} ---\n(no text recognized)")
            if produced >= max_chars:
                break
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        if swapped:
            try:
                ctx.client.unload(model)
            except OllamaError:
                pass
    notes: list[str] = []
    if swapped:
        notes.append(
            f"note: {current} cannot read images, so the pages were read with "
            f"{model} ({reason})"
            + ("; the session model was unloaded and reloads on the next round"
               if unloaded_current else
               "; the session model reloads on the next round"))
    body = "\n\n".join(sections)[:max_chars]
    if not body.strip():
        body = "(the vision model returned no text for these pages)"
    head = f"{label} ({_pdf_pages_label(total)}; read with {model})"
    return "\n".join([head] + notes + [body])


def run(args: dict[str, Any], ctx: ToolContext) -> str:
    label, data = _read_pdf_source(args, ctx)
    if not data:
        raise ToolError(f"{label} is empty")
    if len(data) > MAX_PDF_BYTES:
        raise ToolError(f"{label} is larger than the "
                        f"{MAX_PDF_BYTES // (1024 * 1024)} MB PDF limit")
    if b"%PDF-" not in data[:1024]:
        raise ToolError(f"{label} is not a PDF (no %PDF header)")
    mode = str(args.get("mode") or "auto").strip().lower()
    if mode not in {"auto", "text", "vision"}:
        raise ToolError("mode must be 'auto', 'text' or 'vision'")
    try:
        max_chars = int(args.get("max_chars") or MAX_PDF_CHARS)
    except (TypeError, ValueError):
        max_chars = MAX_PDF_CHARS
    max_chars = max(1000, min(max_chars, MAX_PDF_CHARS))
    cache_key = f"{label}#{args.get('pages') or 'all'}#{mode}#{max_chars}"
    cached = ctx.pdf_reads.get(cache_key)
    if cached is not None:
        return ("(the same PDF was read earlier in this session — not reading "
                "it again)\n" + cached)
    total = pdf_page_count(data)
    text: Optional[str] = None
    if mode != "vision":
        selected = select_pdf_pages(args.get("pages"), total, MAX_PDF_PAGES)
        text = extract_pdf_text(data, max_chars, pages=selected)
    # Any real text layer wins in auto: a readable PDF must never cost a
    # model swap. A scan (no text at all) goes to vision; `mode=vision`
    # forces it even when there is text, for figures and tables.
    has_text = bool(text and text.strip())
    if mode == "text" or (mode == "auto" and has_text):
        result = _pdf_text_result(label, text, total)
    else:
        result = _pdf_vision_result(ctx, data, label, args, total, max_chars)
    ctx.pdf_reads[cache_key] = result
    return result
