"""CSV loader — batch rows into CSV-text chunks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import CSV_ROWS_PER_CHUNK
from llm import build_page_payload, extract_content_chunks, extract_metadata

def split_csv(path: Path, chunk_size: int = CSV_ROWS_PER_CHUNK) -> list[str]:
    """Split a CSV into row-batches serialized back to CSV text.

    Args:
        path: Path to a `.csv` file.
        chunk_size: Rows per batch (default 50).

    Returns:
        List of CSV strings (each includes the header row via `to_csv`).
    """
    df = pd.read_csv(path)
    chunks = []
    for start in range(0, len(df), chunk_size):
        part = df.iloc[start : start + chunk_size]
        chunks.append(part.to_csv(index=False))
    return chunks


def process_csv(path: Path, use_llm: bool = True) -> list[dict]:
    """Convert CSV row-batches into indexable document payloads.

    Args:
        path: Path to the CSV file.
        use_llm: LLM extraction vs heuristic metadata.

    Returns:
        Indexable document dicts for `index_documents`.
    """
    parts = split_csv(path)
    docs = []
    for i, part in enumerate(parts, start=1):
        if use_llm:
            meta = extract_metadata(part, page_number=i)
            content = extract_content_chunks(part)
        else:
            meta = {
                "page_number": i,
                "page_title": (
                    f"CSV rows {(i - 1) * CSV_ROWS_PER_CHUNK + 1}"
                    f"-{i * CSV_ROWS_PER_CHUNK}"
                ),
                "summary": part[:120],
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

