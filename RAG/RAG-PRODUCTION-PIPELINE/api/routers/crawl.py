"""Optional website crawl + index routes."""

from __future__ import annotations

import requests
from fastapi import APIRouter, HTTPException

from api.schemas import CrawlRequest, CrawlResponse

from config import require_openai_key

from loaders.html_loader import CRAWL_HEADERS, discover_urls, process_html
from store import index_documents


router = APIRouter(prefix="/crawl", tags=["crawl"])

@router.post("", response_model=CrawlResponse)
def crawl_and_maybe_index(body: CrawlRequest) -> CrawlResponse:
    """Discover same-origin URLs from a seed; optionally index HTML pages.

    Args (body):
        seed: Starting URL.
        max_pages / max_depth: Crawl limits.
        index: When True, fetch each page and upsert into Qdrant.
        use_llm: LiteLLM extraction when indexing.
    """
    try:
        urls = discover_urls(body.seed, max_pages=body.max_pages, max_depth=body.max_depth)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crawl failed: {e}") from e

    indexed: dict[str, int] = {}
    errors: dict[str, str] = {}
    note: str | None = None

    if not body.index:
        note = (
            "Discovery only — nothing was written to Qdrant because "
            '"index" was false. Send "index": true to store these pages.'
        )
    else:
        if body.use_llm:
            try:
                require_openai_key()
            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        for url in urls:
            try:
                resp = requests.get(url, timeout=20, headers=CRAWL_HEADERS)
                resp.raise_for_status()
                if "text/html" not in resp.headers.get("content-type", ""):
                    errors[url] = f"non-HTML content-type: {resp.headers.get('content-type')}"
                    continue
                docs = process_html(resp.text, source=url, use_llm=body.use_llm)
                if not docs:
                    errors[url] = "no extractable text blocks"
                    continue
                indexed[url] = index_documents(docs, file_name=url)
            except Exception as e:
                errors[url] = f"{type(e).__name__}: {e}"

    if not urls:
        note = (
            "No URLs discovered. The seed may block automated requests or "
            "serve links outside the seed prefix."
        )

    return CrawlResponse(
        seed=body.seed,
        urls=urls,
        indexed=indexed,
        total_points=sum(indexed.values()),
        errors=errors,
        note=note,
    )

