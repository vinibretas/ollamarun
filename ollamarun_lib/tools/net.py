"""HTTP responses and URLs: readable text, failure notes, search reroutes."""
from __future__ import annotations

import json
import re
import urllib.parse
from http import HTTPStatus
from typing import Any

import requests

from .html import extract_html_text

FETCH_TIMEOUT = 30.0
#: Browser-ish headers: the plain requests user-agent is a bot flag for many
#: sites and gets served challenges or error pages instead of content.
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/json,"
               "text/plain;q=0.9,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
}

SEARCH_TIMEOUT = 30.0
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 10
MAX_ERROR_DETAIL = 300
#: Markers that identify a page which is only a JavaScript shell (or a bot
#: wall served with a 200) rather than content.
JS_SHELL_MARKERS = (
    "enablejs", "enable javascript", "javascript is disabled",
    "please turn javascript on", "requires javascript",
)
BOT_WALL_MARKERS = (
    "just a moment", "checking your browser", "cf-browser-verification",
    "challenge-platform", "cf-chl", "access denied", "attention required",
    "are you a robot", "unusual traffic", "captcha",
)
#: The subset that also shows up on challenge pages served with a 200.
SOFT_BLOCK_MARKERS = (
    "just a moment", "checking your browser", "cf-browser-verification",
    "challenge-platform", "cf-chl",
)
#: Hosts whose /search-style pages never return results to a plain HTTP
#: client; `fetch_url` reroutes them to `web_search` instead.
SEARCH_ENGINE_HOSTS = (
    "google.com", "bing.com", "duckduckgo.com", "search.yahoo.com",
    "search.brave.com", "ecosia.org", "mojeek.com", "startpage.com",
    "search.marginalia.nu", "yandex.com",
)


def _http_phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return ""


def _status_url_line(url: str, final_url: str, status: int) -> str:
    phrase = _http_phrase(status)
    line = f"HTTP {status}" + (f" {phrase}" if phrase else "")
    line += f" — {final_url or url}"
    if final_url and final_url != url:
        line += f" (asked for {url})"
    return line


def _looks_like_js_shell(body: str, text: str) -> bool:
    lower = (body or "").lower()
    if any(marker in lower
           for marker in JS_SHELL_MARKERS + SOFT_BLOCK_MARKERS):
        return True
    # A large document that yields almost no text is a scripted page (or a
    # wall) no matter what it says.
    return len(text) < 120 and len(body) > 4000


def _failure_note(status: int, content_type: str, body: str) -> str:
    """What went wrong and what to do next -- never the site's error page."""
    lower = (body or "").lower()
    blocked = any(marker in lower for marker in BOT_WALL_MARKERS)
    if status in (404, 410):
        return ("the page does not exist at this URL. Do not guess more URLs on "
                "this site; call web_search to find the right page — and if the "
                "first search misses it, search again with different words.")
    if status in (401, 407):
        return ("the page needs a login. If it should be public, call web_search "
                "for another source; otherwise ask the user for access.")
    if status == 429:
        return ("the site is rate-limiting requests. Wait a moment, or call "
                "web_search for another source.")
    if blocked:
        return ("blocked by a bot-protection challenge (Cloudflare/Akamai-style), "
                "not the real page. Do not retry this URL or guess others here; "
                "call web_search for another source.")
    if status == 403:
        return ("the site refused the request, so nothing readable came back. "
                "Do not retry this URL; call web_search for another source.")
    if status >= 500:
        return ("the server had a temporary error. The URL may work later; "
                "meanwhile call web_search for a cached or alternative source.")
    if status >= 400:
        return "the server rejected the request. Check the URL or call web_search."
    return "the request did not return usable content."


def _error_detail(content_type: str, body: str) -> str:
    """Short excerpt of a failed response's own message (JSON/plain text)."""
    body = (body or "").strip()
    if not body or "html" in content_type:
        return ""
    text = " ".join(body.split())
    return text[:MAX_ERROR_DETAIL] + ("…" if len(text) > MAX_ERROR_DETAIL else "")


def _is_json(content_type: str) -> bool:
    return content_type == "application/json" or content_type.endswith("+json")


def _is_textual(content_type: str) -> bool:
    if content_type.startswith("text/"):
        return True
    return content_type.endswith("+xml") or content_type in {
        "application/json", "application/xml", "application/javascript",
        "application/x-javascript", "application/xhtml+xml",
        "application/rss+xml", "application/atom+xml",
    }


def _decode_response(response: Any) -> str:
    """Decoded body text, preferring UTF-8 when the site omits its charset.

    `requests` follows the RFC and assumes latin-1 for `text/html` without a
    charset, which mangles UTF-8 pages (docs sites, news, ...).
    """
    data = response.content or b""
    encoding = (getattr(response, "encoding", None) or "").lower()
    if encoding in {"", "iso-8859-1", "latin-1", "ascii"}:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            pass
    try:
        return response.text
    except (AttributeError, UnicodeDecodeError):
        return data.decode("utf-8", errors="replace")


def format_fetch_response(*, url: str, final_url: str, status: int,
                          content_type: str, body: str, size: int = 0) -> str:
    """The model-readable text for one HTTP response.

    HTML becomes extracted text plus the page's links, JSON is pretty-printed,
    error pages are replaced by an explanation of what happened and what to do
    next, and JavaScript-only shells say so instead of looking empty.
    """
    head = _status_url_line(url, final_url or url, status)
    if status >= 400:
        lines = [head, f"fetch failed: {_failure_note(status, content_type, body)}"]
        detail = _error_detail(content_type, body)
        if detail:
            lines.append(f"server said: {detail}")
        return "\n".join(lines)
    if not body.strip() and not size:
        return head + "\n(empty response body)"
    if "html" in content_type or (not content_type and "<html" in body[:2000].lower()):
        title, text, links = extract_html_text(body, base_url=final_url or url)
        if len(text) < 120 and _looks_like_js_shell(body, text):
            lines = [head]
            if title:
                lines.append(f"Title: {title}")
            lines.append(
                "no readable text: this page is rendered by JavaScript (only "
                "the app shell came back). Fetching it again will not help — "
                "call web_search and pick a result that serves static text, an "
                "API endpoint or a docs page; if nothing fits, search again "
                "with different words.")
            return "\n".join(lines)
        parts = [head]
        if title:
            parts.append(f"Title: {title}")
        parts.append("")
        parts.append(text or "(no readable text)")
        if links:
            parts.append("")
            parts.append("Links on this page:")
            parts.extend(f"- {label or link} — {link}" for label, link in links)
        return "\n".join(parts)
    if _is_json(content_type):
        try:
            pretty = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
        except ValueError:
            pretty = body
        return f"{head}\n{pretty}"
    if _is_textual(content_type):
        return f"{head}\n{body}"
    shown = size or len(body)
    return (f"{head}\n(binary content: {content_type or 'unknown type'}, "
            f"{shown} bytes — not shown as text)")


def search_engine_query(url: str) -> str:
    """The query inside a search-engine URL, or '' when it is not one."""
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return ""
    known = any(host == candidate or host.endswith("." + candidate)
                for candidate in SEARCH_ENGINE_HOSTS)
    googleish = host.startswith("google.") or ".google." in host
    if not (known or googleish):
        return ""
    params = urllib.parse.parse_qs(parsed.query)
    for key in ("q", "query", "p", "text", "search_query"):
        values = params.get(key)
        if values and values[0].strip():
            return values[0].strip()
    return ""


def _short_error(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text[:200] or exc.__class__.__name__


def _request_error(exc: requests.RequestException) -> str:
    """A short, actionable reason for a failed request."""
    if isinstance(exc, requests.Timeout):
        return "the request timed out (the site may be slow or blocking this machine)"
    text = str(exc)
    if ("Name or service not known" in text or "Failed to resolve" in text
            or "No address associated with hostname" in text):
        return ("the host name could not be resolved — the domain is probably "
                "misspelled or wrong; call web_search to find the real URL")
    match = re.search(r"Caused by ([^\n]+)", text)
    if match:
        text = match.group(1)
    return _short_error(text)
