"""The `fetch_url` tool: readable page text, PDFs and search-engine reroutes."""
from __future__ import annotations

import urllib.parse
from typing import Any

import requests

from .net import (BROWSER_HEADERS, DEFAULT_SEARCH_RESULTS, FETCH_TIMEOUT,
                  SEARCH_TIMEOUT, _decode_response, _is_textual,
                  _request_error, _status_url_line, format_fetch_response,
                  search_engine_query)
from .pdf import extract_pdf_text
from .spec import ToolContext, ToolError, _clamp_timeout
from .web_search import run_web_search


def _handler_fetch_url(args: dict[str, Any], ctx: ToolContext) -> str:
    url = str(args.get("url") or "").strip()
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolError("url must be an absolute http:// or https:// URL")

    query = search_engine_query(url)
    if query:
        timeout = _clamp_timeout(args.get("timeout"),
                                 min(ctx.timeout, SEARCH_TIMEOUT))
        results = run_web_search(query, DEFAULT_SEARCH_RESULTS, timeout)
        if results is None:
            raise ToolError(
                "this is a search-engine results URL, which only returns a "
                "JavaScript shell when fetched; the web_search tool searches "
                "from here instead, but its ddgs backend is not installed")
        return (f"(search-engine URL: ran web_search for {query!r} instead of "
                f"fetching the page)\n{results}")

    previous = ctx.fetched.get(url)
    if previous:
        preview = previous[:400] + ("…" if len(previous) > 400 else "")
        return (f"(already fetched {url} earlier in this session — not fetching "
                f"it again)\nstart of the earlier result:\n{preview}")

    timeout = _clamp_timeout(args.get("timeout"), min(ctx.timeout, FETCH_TIMEOUT))
    try:
        response = requests.get(url, timeout=timeout,
                                headers=dict(BROWSER_HEADERS),
                                allow_redirects=True)
    except requests.RequestException as exc:
        raise ToolError(f"could not fetch {url}: {_request_error(exc)}") from exc
    try:
        status = response.status_code
        content_type = (response.headers.get("Content-Type") or "")
        content_type = content_type.split(";")[0].strip().lower()
        size = len(response.content or b"")
        if status < 400 and ("pdf" in content_type
                             or parsed.path.lower().endswith(".pdf")):
            result = _pdf_result(url, response, size)
        else:
            body = _decode_response(response) if _is_textual(content_type) else ""
            result = format_fetch_response(url=url, final_url=response.url,
                                           status=status,
                                           content_type=content_type,
                                           body=body, size=size)
    finally:
        response.close()
    ctx.fetched[url] = result
    return result


def _pdf_result(url: str, response: Any, size: int) -> str:
    head = _status_url_line(url, response.url or url, response.status_code)
    text = extract_pdf_text(response.content or b"")
    if text is None:
        return (f"{head}\n(a PDF, {size} bytes; its text could not be "
                f"extracted — the file may be encrypted or malformed. If it is "
                f"a scan, call read_pdf with this URL to read the pages with a "
                f"vision model)")
    if not text.strip():
        return (f"{head}\n(a PDF, {size} bytes; no text layer — a scan. Call "
                f"read_pdf with this URL to read the pages with a vision model)")
    return f"{head}\n(PDF turned into text, {size} bytes)\n{text}"
