"""The `web_search` tool: the ddgs backend, result formatting and handler."""
from __future__ import annotations

from typing import Any, Optional

from .net import (DEFAULT_SEARCH_RESULTS, MAX_SEARCH_RESULTS, SEARCH_TIMEOUT,
                  _short_error)
from .spec import ToolContext, ToolError, _clamp_timeout

try:  # the search backend for the web_search tool
    from ddgs import DDGS  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    DDGS = None


def search_available() -> bool:
    """True when the web_search backend (ddgs) can be imported."""
    return DDGS is not None


def format_search_results(query: str, results: list[tuple[str, str, str]]) -> str:
    if not results:
        return (f"web_search: no results for {query!r}. That is not proof the "
                f"information does not exist — search again with different or "
                f"narrower words (synonyms, exact phrases, site:, a year) instead "
                f"of deciding it is not available; only report it as not found "
                f"after a few rephrased searches come up empty (do not guess URLs).")
    lines = [f"web_search: {query!r} — {len(results)} result(s)"]
    for index, (title, url, snippet) in enumerate(results, 1):
        lines.append(f"{index}. {title or '(untitled)'}")
        lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {' '.join(snippet.split())[:300]}")
    lines.append("")
    lines.append("Next: call fetch_url on the most promising result(s) to read them. "
                 "If none of them answers the question, search again with different "
                 "or narrower words instead of settling.")
    return "\n".join(lines)


def run_web_search(query: str, max_results: int = DEFAULT_SEARCH_RESULTS,
                   timeout: float = SEARCH_TIMEOUT) -> Optional[str]:
    """Search through ddgs; None when the backend is not installed."""
    client_class = DDGS
    if client_class is None:
        return None
    try:
        with client_class(timeout=timeout) as client:
            raw = client.text(query, max_results=max_results)
    except TypeError:  # a backend whose constructor signature differs
        try:
            with client_class() as client:
                raw = client.text(query, max_results=max_results)
        except Exception as exc:
            raise ToolError(f"web search failed: {_short_error(exc)}") from exc
    except Exception as exc:
        raise ToolError(f"web search failed: {_short_error(exc)}") from exc
    results: list[tuple[str, str, str]] = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "")
        link = str(entry.get("href") or entry.get("url")
                   or entry.get("link") or "")
        snippet = str(entry.get("body") or entry.get("snippet")
                      or entry.get("description") or "")
        if title or link:
            results.append((title, link, snippet))
    return format_search_results(query, results)


def _handler_web_search(args: dict[str, Any], ctx: ToolContext) -> str:
    query = str(args.get("query") or args.get("q") or "").strip()
    if not query:
        raise ToolError("missing 'query'")
    cached = ctx.searches.get(query.lower())
    if cached:
        return f"(cached result from earlier in this session)\n{cached}"
    try:
        max_results = int(args.get("max_results") or DEFAULT_SEARCH_RESULTS)
    except (TypeError, ValueError):
        max_results = DEFAULT_SEARCH_RESULTS
    max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
    timeout = _clamp_timeout(args.get("timeout"), min(ctx.timeout, SEARCH_TIMEOUT))
    text = run_web_search(query, max_results, timeout)
    if text is None:
        raise ToolError("web search is unavailable (the ddgs package is not installed)")
    ctx.searches[query.lower()] = text
    return text
