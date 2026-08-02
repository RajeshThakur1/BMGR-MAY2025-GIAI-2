"""Health and collection info routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from api.schemas import HealthResponse, CollectionInfoResponse
from config import COLLECTION, OPENAI_API_KEY, QDRANT_URL, TENANT_ID
from store import get_qdrant

router = APIRouter(tags=["health"])


## Why Return is not Expected
@router.get("/health", response_model=HealthResponse)
def health():
    """
    Return service liveness and key config flags.

    Does not fail if Qdrant is down — use `/collection` to verify the vector DB.
    """

    return HealthResponse(
        status="ok",
        qudrant_url=QDRANT_URL,
        collection=COLLECTION,
        tenant_id=TENANT_ID,
        openai_configured=bool(OPENAI_API_KEY),
    )


@router.get("/collection", response_model=CollectionInfoResponse)
def collection_info() -> CollectionInfoResponse:
    """Return Qdrant collection existence and point counts.

    Raises:
        HTTPException: 503 if Qdrant cannot be reached.
    """
    try:
        client = get_qdrant()
        names = {c.name for c in client.get_collections().collections}
        if COLLECTION not in names:
            return CollectionInfoResponse(name=COLLECTION, exists=False)
        info = client.get_collection(COLLECTION)
        return CollectionInfoResponse(
            name=COLLECTION,
            exists=True,
            points_count=getattr(info, "points_count", None),
            vectors_count=getattr(info, "vectors_count", None),
            status=str(getattr(info, "status", None)),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {e}") from e