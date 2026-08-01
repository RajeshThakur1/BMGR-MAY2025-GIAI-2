"""TXT loader — 50 lines per chunk, then metadata / content extraction."""

from __future__ import annotations

from pathlib import Path

from config import TXT_LINES_PER_CHUNK
from llm import build_page_payload, extract_content_chunks, extract_metadata


def split_txt(path: Path, lines_per_file: int = TXT_LINES_PER_CHUNK) -> list[str]:
    """Split a plain-text file into fixed-size line groups.

    Args:
        path: Path to a `.txt` file.
        lines_per_file: Number of lines per raw chunk (default 50).

    Returns:
        List of raw text strings (joined lines), one per chunk.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    return [
        "".join(lines[i : i + lines_per_file])
        for i in range(0, len(lines), lines_per_file)
    ]


def process_txt(path: Path, use_llm: bool = True) -> list[dict]:
    """Turn a TXT file into indexable documents (payload + metadata).

    Args:
        path: Path to the text file.
        use_llm: If True, call LiteLLM for metadata and chunks.

    Returns:
        List of `{"page_content": json_payload, "metadata": {...}}` dicts.
    """
    raw_chunks = split_txt(path)
    docs = []
    for i, chunk_text in enumerate(raw_chunks, start=1):
        if use_llm:
            meta = extract_metadata(chunk_text, page_number=i)
            content = extract_content_chunks(chunk_text)
        else:
            meta = {
                "page_number": i,
                "page_title": f"TXT chunk {i}",
                "summary": chunk_text[:120],
                "keywords": [],
            }
            content = [chunk_text.strip()]
        payload = build_page_payload(meta, content)
        docs.append({"page_content": payload, "metadata": {"source": str(path), "pg": i}})
    return docs
