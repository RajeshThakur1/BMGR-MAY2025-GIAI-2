"""PDF/DOCX indexing via model server (tutorial: ~220-word chunks + full-page PNG)."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import fitz
import httpx

from backend.store import DataStore
from shared.config import settings


def pdf_to_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[dict[str, Any]] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text")
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        png = pix.tobytes("png")
        pages.append(
            {
                "page_number": i + 1,
                "text": text,
                "image_base64": base64.b64encode(png).decode("ascii"),
            }
        )
    doc.close()
    return pages


def index_pdf_bytes(store: DataStore, document_id: str, pdf_bytes: bytes) -> dict[str, Any]:
    pages = pdf_to_pages(pdf_bytes)
    job_id = document_id
    store.upsert_job(job_id, document_id, len(pages), 0, "running")

    base = settings.model_server_url.rstrip("/")
    headers = {"X-API-Key": settings.model_server_api_key}
    all_chunks: list[dict[str, Any]] = []

    with httpx.Client(timeout=300.0) as client:
        for idx, page in enumerate(pages, start=1):
            r = client.post(
                f"{base}/index/document",
                headers=headers,
                json={"document_id": document_id, "pages": [page]},
            )
            r.raise_for_status()
            all_chunks.extend(r.json()["chunks"])
            store.upsert_job(job_id, document_id, len(pages), idx, "running")

    store.index_chunks(all_chunks)
    store.update_document_status(document_id, "ready")
    store.upsert_job(job_id, document_id, len(pages), len(pages), "completed")
    return {"pages": len(pages), "chunks": len(all_chunks)}
