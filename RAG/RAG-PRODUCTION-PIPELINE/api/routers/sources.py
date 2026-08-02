"""Source management — list documents and delete by fileName."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.schemas import DeleteSourceResponse, DocumentListResponse
from config import TENANT_ID
from store import delete_by_filename, list_filenames

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("", response_model=DocumentListResponse)
def list_sources(
    tenant_id: str | None = Query(
        default=None, description="Tenant scope; defaults to config TENANT_ID"
    ),
) -> DocumentListResponse:
    """List distinct indexed `fileName` values for the tenant.

    Use this to populate document pickers before calling `/ask` or `/search`
    with a `file_names` filter.
    """
    tenant = tenant_id or TENANT_ID
    try:
        names = list_filenames(tenant_id=tenant)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"List sources failed: {e}") from e
    return DocumentListResponse(
        tenant_id=tenant,
        file_names=names,
        count=len(names),
    )


@router.delete("/{file_name}", response_model=DeleteSourceResponse)
def delete_source(
    file_name: str,
    tenant_id: str | None = Query(
        default=None, description="Tenant scope; defaults to config TENANT_ID"
    ),
) -> DeleteSourceResponse:
    """Delete all Qdrant points for a given `fileName` within a tenant.

    Args:
        file_name: Exact name used at index time (e.g. `catalog.csv`).
        tenant_id: Optional tenant override.
    """
    tenant = tenant_id or TENANT_ID
    try:
        delete_by_filename(file_name, tenant_id=tenant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}") from e
    return DeleteSourceResponse(
        message="Deleted points for file",
        file_name=file_name,
        tenant_id=tenant,
    )
