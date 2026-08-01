"""Fallback RecursiveCharacterTextSplitter path (non-LLM)."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import FALLBACK_CHUNK_OVERLAP, FALLBACK_CHUNK_SIZE


def split_text_from_docs(
    docs: list[Document],
    chunk_size: int = FALLBACK_CHUNK_SIZE,
    chunk_overlap: int = FALLBACK_CHUNK_OVERLAP,
) -> list[dict]:
    """Fallback chunker when LLM extraction is disabled.

    Uses LangChain `RecursiveCharacterTextSplitter` (500 / 100 by default),
    matching production `file_utils.split_text_from_docs`.

    Args:
        docs: LangChain `Document` objects with `page_content` + `metadata`.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of dicts: `{"page_content": str, "metadata": dict}`.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    split = splitter.split_documents(docs)
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in split]
