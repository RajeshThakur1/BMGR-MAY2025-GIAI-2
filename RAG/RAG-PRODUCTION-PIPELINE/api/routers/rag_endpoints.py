"""Search and RAG ask routes (optional document filter)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import AskRequest, AskResponse, SearchHit, SearchRequest, SearchResponse

from config import TENANT_ID, require_openai_key

from rag import rag_answer

from store import search

router = APIRouter(tags=["rag"])


@router.post("/search", response_model=SearchResponse)
def search_chunks(body: SearchRequest) -> SearchResponse:
    """Dense vector search in Qdrant (retrieve only, no LLM answer).

    Pass `file_names` to restrict retrieval to specific indexed documents.
    """
    tenant = body.tenant_id or TENANT_ID
    names = body.file_names or []
    try:
        hits = search(
            body.query,
            tenant_id=tenant,
            limit=body.limit,
            file_names=names or None,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Search failed: {e}") from e

    results: list[SearchHit] = []
    for i, h in enumerate(hits, start=1):
        payload = h.payload or {}
        results.append(
            SearchHit(
                rank=i,
                score=float(h.score) if h.score is not None else None,
                file_name=payload.get("fileName"),
                source=payload.get("source"),
                content_preview=(payload.get("content") or "")[:400],
            )
        )
    return SearchResponse(query=body.query, hits=results, file_names=names)


@router.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest) -> AskResponse:
    """Full RAG pipeline: retrieve top-k chunks and generate a grounded answer.

    Optionally restrict retrieval to specific documents via `file_names`
    (exact `fileName` values from indexing). Empty / omitted = search all.

    Requires `OPENAI_API_KEY` for the generation step.
    """
    try:
        require_openai_key()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    tenant = body.tenant_id or TENANT_ID
    try:
        result = rag_answer(
            body.question,
            limit=body.limit,
            tenant_id=tenant,
            show_sources=body.show_sources,
            file_names=body.file_names,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failed: {e}") from e

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        file_names=result.get("file_names", []),
    )




