"""Markdown loader — merge heading sections until min word count."""

from __future__ import annotations

import re
from pathlib import Path

from config import MD_MIN_WORDS
from llm import build_page_payload, extract_content_chunks, extract_metadata


def split_markdown_smart(path: Path, min_word_count: int = MD_MIN_WORDS) -> list[str]:
    """Split Markdown on top-level `#` headings, merging small sections.

    Args:
        path: Path to a Markdown file.
        min_word_count: Minimum words before flushing a chunk (default 500).

    Returns:
        List of Markdown section strings.
    """
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?=^# )", text, flags=re.M)
    sections = [s.strip() for s in sections if s.strip()]
    chunks, buf = [], ""
    for sec in sections:
        candidate = (buf + "\n\n" + sec).strip() if buf else sec
        if len(candidate.split()) >= min_word_count and buf:
            chunks.append(buf)
            buf = sec
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


def process_markdown(path: Path, use_llm: bool = True) -> list[dict]:
    """Convert a Markdown guide into indexable JSON payloads.

    Args:
        path: Path to the `.md` file.
        use_llm: Use LiteLLM extraction when True; heuristic metadata otherwise.

    Returns:
        Indexable document dicts for `index_documents`.
    """
    parts = split_markdown_smart(path)
    docs = []
    for i, part in enumerate(parts, start=1):
        if use_llm:
            meta = extract_metadata(part, page_number=i)
            content = extract_content_chunks(part)
        else:
            meta = {
                "page_number": i,
                "page_title": part.splitlines()[0][:80],
                "summary": " ".join(part.split()[:40]),
                "keywords": [],
            }
            content = [part]
        docs.append(
            {
                "page_content": build_page_payload(meta, content),
                "metadata": {"source": str(path), "pg": i},
            }
        )
    return docs
