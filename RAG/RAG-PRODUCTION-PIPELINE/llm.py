"""LiteLLM helpers for JSON extraction, metadata, and page payloads."""

from __future__ import annotations

import json
from typing import Any

import litellm

from config import LLM_MODEL

litellm.drop_params = True


def llm_json(prompt: str, system: str = "Return valid JSON only.") -> Any:
    """Call ChatGPT via LiteLLM and parse the response as JSON.

    Shared helper for indexing: content extraction and metadata extraction
    both go through this function so every call uses the same model,
    temperature, and JSON response format.

    Args:
        prompt: User message describing what JSON to produce.
        system: System instruction. Defaults to forcing valid JSON only.

    Returns:
        Parsed Python object from the model JSON (usually a dict).

    Raises:
        json.JSONDecodeError: If the model returns non-JSON text.
        Exception: Propagates LiteLLM / API errors.
    """
    resp = litellm.completion(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)



def extract_content_chunks(text: str) -> list[str]:
    """Split a page/section into retrieval-friendly content chunks via LLM.

    Mirrors production `generate_for_page`: returns coherent strings that
    become the `content` array inside the indexed JSON payload.

    Args:
        text: Raw page or section text (truncated to 12k chars for the prompt).

    Returns:
        List of chunk strings.
    """
    result = llm_json(
        f"""Extract the content from this page.
Return JSON: {{"chunks": ["..."]}} where each string is a coherent chunk.
- Preserve useful structure; avoid redundant whitespace.
- Optimize for retrieval while keeping output minimal.

PAGE:
{text[:12000]}""",
        system="You extract retrieval-friendly content chunks. Return JSON only.",
    )
    chunks = result.get("chunks", result if isinstance(result, list) else [])
    return [str(c) for c in chunks]




def extract_metadata(text: str, page_number: int = 1) -> dict:
    """Generate page-level metadata (title, summary, keywords) via LLM.

    Args:
        text: Source text for this page/chunk (truncated to 8k chars).
        page_number: Logical page or chunk index to stamp into the result.

    Returns:
        Dict with at least `page_number`, `page_title`, `summary`, `keywords`.
    """
    result = llm_json(
        f"""Given this document chunk, return JSON with keys:
page_number (int), page_title (str), summary (str), keywords (list of str).

Use page_number={page_number}.

TEXT:
{text[:8000]}"""
    )
    result["page_number"] = page_number
    return result



def build_page_payload(
    entry: dict | None,
    content_chunks: list[str],
    images_line: str | None = None,
) -> str:
    """Build the production-shaped JSON string stored as Qdrant `content`.

    Args:
        entry: Metadata dict from `extract_metadata`, or None for content-only.
        content_chunks: List of content strings for the `content` field.
        images_line: Optional single-line image descriptions (`imgs` field).

    Returns:
        JSON string with keys like `pg`, `title`, `sum`, `kw`, `content`,
        and optionally `imgs`.
    """
    if entry:
        kw = entry.get("keywords", [])
        if isinstance(kw, list):
            kw = ", ".join(str(k) for k in kw)
        payload = {
            "pg": entry.get("page_number", 1),
            "title": entry.get("page_title", ""),
            "sum": entry.get("summary", ""),
            "kw": kw,
            "content": content_chunks,
        }
        if images_line:
            payload["imgs"] = images_line
    else:
        payload = {"content": content_chunks}
    return json.dumps(payload, ensure_ascii=False)



# if __name__ == "__main__":
#     # Quick test of the JSON extraction helper
#     test_text = "This is a test page. It has some content. It also has a title."
#     metadata = extract_metadata(test_text, page_number=1)
#     chunks = extract_content_chunks(test_text)
#     payload = build_page_payload(metadata, chunks)
#     print(payload)