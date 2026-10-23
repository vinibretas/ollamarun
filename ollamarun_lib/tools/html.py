"""HTML -> readable text and links: the page reader behind `fetch_url`."""
from __future__ import annotations

import re
import urllib.parse
from html.parser import HTMLParser
from typing import Optional


#: How many links one page reports back to the model.
MAX_PAGE_LINKS = 25


class _HTMLTextExtractor(HTMLParser):
    """Collect readable text and links from one HTML document.

    Boilerplate (scripts, styles, navigation, forms) is dropped, block tags
    become line breaks and headings/list items keep a markdown-ish marker.
    Content inside `<article>`/`<main>` is tracked separately so a page's
    chrome can be left out of the result.
    """

    SKIP_TAGS = {
        "script", "style", "noscript", "template", "svg", "iframe", "canvas",
        "form", "button", "select", "option", "nav", "header", "footer",
        "aside", "dialog", "map", "object", "embed",
    }
    BLOCK_TAGS = {
        "address", "article", "blockquote", "dd", "div", "dl", "dt",
        "fieldset", "figcaption", "figure", "hr", "li", "main", "ol", "p",
        "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead",
        "tr", "ul",
    }
    HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
    MAIN_TAGS = {"article", "main"}

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_chunks: list[str] = []
        self.chunks: list[tuple[str, bool]] = []
        self.links: list[tuple[str, str, bool]] = []
        self._in_title = False
        self._in_head = False
        self._skip_depth = 0
        self._main_depth = 0
        self._open_link: Optional[tuple[list[str], str, bool]] = None

    def _emit(self, text: str) -> None:
        if not text:
            return
        self.chunks.append((text, self._main_depth > 0))
        if self._open_link is not None:
            self._open_link[0].append(text)

    def _break(self, count: int = 1) -> None:
        self.chunks.append(("\n" * count, self._main_depth > 0))

    @staticmethod
    def _usable_href(href: str) -> bool:
        href = href.strip()
        return bool(href) and not href.startswith(
            ("#", "javascript:", "mailto:", "tel:", "data:"))

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self.MAIN_TAGS:
            self._main_depth += 1
        if tag == "title":
            self._in_title = True
        elif tag == "head":
            self._in_head = True
        elif tag in self.HEADING_TAGS:
            self._break(2)
            self._emit("#" * self.HEADING_TAGS[tag] + " ")
        elif tag == "li":
            self._break()
            self._emit("- ")
        elif tag in self.BLOCK_TAGS or tag == "br":
            self._break()
        elif tag == "a":
            href = dict(attrs).get("href") or ""
            if self._usable_href(href):
                self._open_link = ([], href, self._main_depth > 0)

    def handle_startendtag(self, tag: str,
                           attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS or self._skip_depth:
            return
        if tag in self.BLOCK_TAGS or tag == "br":
            self._break()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag == "head":
            self._in_head = False
        elif tag == "a" and self._open_link is not None:
            text, href, in_main = self._open_link
            label = " ".join("".join(text).split())
            self.links.append((label, href, in_main))
            self._open_link = None
        elif tag in self.HEADING_TAGS or tag in self.BLOCK_TAGS or tag == "li":
            self._break()
        if tag in self.MAIN_TAGS:
            self._main_depth = max(0, self._main_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data:
            return
        if self._in_title:
            self.title_chunks.append(data)
        if self._in_head:
            return
        self._emit(data)


def _normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_html_text(html: str,
                      base_url: str = "") -> tuple[str, str, list[tuple[str, str]]]:
    """Return (title, readable text, links) for one HTML document.

    The `<article>`/`<main>` region wins when it holds most of the text;
    otherwise the whole body is used. Links come back as absolute, decoded,
    de-duplicated (label, url) pairs, in page order.
    """
    parser = _HTMLTextExtractor(base_url)
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup must never fail a fetch
        pass
    title = _normalize_text("".join(parser.title_chunks))
    full = _normalize_text("".join(chunk for chunk, _ in parser.chunks))
    main = _normalize_text("".join(chunk for chunk, in_main in parser.chunks
                                   if in_main))
    if main and len(main) >= max(200, int(0.2 * len(full))):
        text = main
        entries = [entry for entry in parser.links if entry[2]]
    else:
        text = full
        entries = parser.links
    seen: set[str] = set()
    links: list[tuple[str, str]] = []
    for label, href, _ in entries:
        url = urllib.parse.urljoin(base_url, href)
        if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
            continue
        if url in seen:
            continue
        seen.add(url)
        links.append((label, url))
        if len(links) >= MAX_PAGE_LINKS:
            break
    return title, text, links

