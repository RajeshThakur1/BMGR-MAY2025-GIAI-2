"""HTML / website loaders and a small BFS URL discovery helper."""

from __future__ import annotations

from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import HTML_CHAR_LIMIT
from llm import build_page_payload, extract_content_chunks, extract_metadata


# Many sites (Medium, Cloudflare-fronted hosts) reject the default requests UA.
CRAWL_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}


def split_html_char_chunks(html: str, char_limit: int = HTML_CHAR_LIMIT) -> list[str]:
    """Extract semantic HTML blocks and pack them into ~char_limit chunks.

    Args:
        html: Raw HTML string.
        char_limit: Soft max characters per chunk.

    Returns:
        List of plain-text chunks suitable for LLM extraction / indexing.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    blocks = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li", "pre", "td", "th"]):
        text = el.get_text(" ", strip=True)
        if text:
            blocks.append(text)

    chunks, current = [], ""
    min_chunk = char_limit // 4
    for block in blocks:
        candidate = f"{current}\n{block}".strip() if current else block
        if len(candidate) > char_limit and current and len(current) >= min_chunk:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current and len(current) >= 200:
        chunks.append(current)
    elif current and chunks:
        chunks[-1] = f"{chunks[-1]}\n{current}"
    elif current:
        chunks.append(current)
    return chunks


def process_html(html: str, source: str, use_llm: bool = True) -> list[dict]:
    """Convert HTML (local file or fetched page) into indexable payloads.

    Args:
        html: Raw HTML markup.
        source: Source URL or file path stored in metadata.
        use_llm: LLM extraction vs heuristic metadata.

    Returns:
        Indexable document dicts for `index_documents`.
    """
    parts = split_html_char_chunks(html)
    docs = []
    for i, part in enumerate(parts, start=1):
        if use_llm:
            meta = extract_metadata(part, page_number=i)
            content = extract_content_chunks(part)
        else:
            meta = {
                "page_number": i,
                "page_title": part[:60],
                "summary": part[:120],
                "keywords": [],
            }
            content = [part]
        docs.append(
            {
                "page_content": build_page_payload(meta, content),
                "metadata": {"source": source, "pg": i},
            }
        )
    return docs



def discover_urls(seed: str, max_pages: int = 10, max_depth: int = 2) -> list[str]:
    """Tiny BFS crawl sketch of production URL discovery (no Playwright).

    Args:
        seed: Starting URL (also used as same-origin prefix filter).
        max_pages: Maximum HTML pages to collect.
        max_depth: Max link hops from the seed.

    Returns:
        Ordered list of discovered HTML URLs.
    """
    seen, queue, out = set(), [(seed, 0)], []
    while queue and len(out) < max_pages:
        url, depth = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            r = requests.get(url, timeout=15, headers=CRAWL_HEADERS)
            if "text/html" not in r.headers.get("content-type", ""):
                continue
            out.append(url)
            if depth >= max_depth:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"]).split("#")[0]
                if href.startswith(seed.rstrip("/")):
                    queue.append((href, depth + 1))
        except Exception as e:
            print("skip", url, e)
    return out
