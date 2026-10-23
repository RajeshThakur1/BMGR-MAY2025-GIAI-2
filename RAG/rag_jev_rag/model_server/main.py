"""FastAPI model server — mirrors Lightning AI Studio endpoints from the tutorial."""

from __future__ import annotations

import base64
import io
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from shared.config import settings
from shared.decisions import score_candidate
from shared.embeddings import embed_query, embed_texts, pseudo_embed_image

app = FastAPI(title="Jev RAG Model Server", version="0.1.0")


def verify_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.model_server_api_key and x_api_key != settings.model_server_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


class EmbedTextRequest(BaseModel):
    texts: list[str]


class EmbedTextResponse(BaseModel):
    vectors: list[list[float]]
    dim: int


class EmbedImageRequest(BaseModel):
    image_base64: str


class RerankItem(BaseModel):
    id: str
    text: str
    modality: Literal["text", "image"] = "text"
    page_number: int | None = None


class RerankRequest(BaseModel):
    query: str
    candidates: list[RerankItem]


class RerankResult(BaseModel):
    id: str
    relevance: float
    answerable: float
    passed: bool


class RerankResponse(BaseModel):
    results: list[RerankResult]
    provider: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "decision_provider": settings.decision_provider}


@app.post("/embed/text", response_model=EmbedTextResponse, dependencies=[Depends(verify_key)])
def embed_text(body: EmbedTextRequest) -> EmbedTextResponse:
    vectors = embed_texts(body.texts)
    dim = len(vectors[0]) if vectors else settings.embedding_dim
    return EmbedTextResponse(vectors=vectors, dim=dim)


@app.post("/embed/query", response_model=EmbedTextResponse, dependencies=[Depends(verify_key)])
def embed_single_query(body: EmbedTextRequest) -> EmbedTextResponse:
    if len(body.texts) != 1:
        raise HTTPException(400, "Provide exactly one query string in texts")
    return EmbedTextResponse(vectors=[embed_query(body.texts[0])], dim=settings.embedding_dim)


@app.post("/embed/image", response_model=EmbedTextResponse, dependencies=[Depends(verify_key)])
def embed_image(body: EmbedImageRequest) -> EmbedTextResponse:
    raw = base64.b64decode(body.image_base64)
    return EmbedTextResponse(vectors=[pseudo_embed_image(raw)], dim=settings.embedding_dim)


@app.post("/rerank", response_model=RerankResponse, dependencies=[Depends(verify_key)])
def rerank(body: RerankRequest) -> RerankResponse:
    out: list[RerankResult] = []
    provider = settings.decision_provider
    for c in body.candidates:
        scores = score_candidate(
            body.query,
            c.text,
            modality=c.modality,
            page_number=c.page_number,
        )
        provider = scores.provider
        rel_t = (
            settings.text_relevance_threshold
            if c.modality == "text"
            else settings.image_relevance_threshold
        )
        ans_t = (
            settings.text_answerable_threshold
            if c.modality == "text"
            else settings.image_answerable_threshold
        )
        passed = scores.relevance >= rel_t and scores.answerable >= ans_t
        out.append(
            RerankResult(
                id=c.id,
                relevance=scores.relevance,
                answerable=scores.answerable,
                passed=passed,
            )
        )
    out.sort(key=lambda x: (x.relevance + x.answerable), reverse=True)
    return RerankResponse(results=out, provider=provider)


class IndexPageChunk(BaseModel):
    page_number: int
    text: str
    image_base64: str | None = None


class IndexDocumentRequest(BaseModel):
    document_id: str
    pages: list[IndexPageChunk]


class IndexDocumentResponse(BaseModel):
    chunks: list[dict[str, Any]]


def _chunk_text(text: str, words_per_chunk: int = 220) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    for i in range(0, len(words), words_per_chunk):
        chunks.append(" ".join(words[i : i + words_per_chunk]))
    return chunks


@app.post("/index/document", response_model=IndexDocumentResponse, dependencies=[Depends(verify_key)])
def index_document(body: IndexDocumentRequest) -> IndexDocumentResponse:
    """Split page text ~220 words + attach full-page image vectors (tutorial pipeline)."""
    records: list[dict[str, Any]] = []
    for page in body.pages:
        for ci, chunk in enumerate(_chunk_text(page.text)):
            vec = embed_texts([chunk])[0]
            records.append(
                {
                    "id": f"{body.document_id}-p{page.page_number}-t{ci}",
                    "document_id": body.document_id,
                    "page_number": page.page_number,
                    "modality": "text",
                    "content": chunk,
                    "vector": vec,
                }
            )
        if page.image_base64:
            raw = base64.b64decode(page.image_base64)
            records.append(
                {
                    "id": f"{body.document_id}-p{page.page_number}-img",
                    "document_id": body.document_id,
                    "page_number": page.page_number,
                    "modality": "image",
                    "content": f"[Page {page.page_number} image]",
                    "vector": pseudo_embed_image(raw),
                }
            )
    return IndexDocumentResponse(chunks=records)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)