"""Index sample (or custom) files into Qdrant."""

from __future__ import annotations

from pathlib import Path


from loaders import (
    process_csv,
    process_html,
    process_markdown,
    process_pdf,
    process_txt,
)

from store import ensure_collection, index_documents





def index_file(path: Path, use_llm: bool = True) -> int:
    """Index a single file by extension into Qdrant.

    Args:
        path: Path to a `.txt`, `.md`, `.csv`, `.html`, or `.pdf` file.
        use_llm: Whether to use LiteLLM extraction.

    Returns:
        Number of points upserted.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        docs = process_txt(path, use_llm=use_llm)
    elif suffix in {".md", ".markdown"}:
        docs = process_markdown(path, use_llm=use_llm)
    elif suffix == ".csv":
        docs = process_csv(path, use_llm=use_llm)
    elif suffix in {".html", ".htm"}:
        docs = process_html(path.read_text(encoding="utf-8"), source=str(path), use_llm=use_llm)
    elif suffix == ".pdf":
        docs = process_pdf(path, use_llm=use_llm, process_images=False)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    n = index_documents(docs, file_name=path.name)
    print(f"Indexed {path.name} → {n} points")
    return n