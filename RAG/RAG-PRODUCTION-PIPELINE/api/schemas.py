"""Pydantic request/response models for the RAG FastAPI services."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness / readiness payload."""

    status: str
    qdrant_url: str
    collection: str
    tenant_id: str
    openai_configured: bool


class IndexSamplesRequest(BaseModel):
    """Options for indexing the whole sample_data directory."""

    use_llm: bool = Field(
        default=True,
        description="Use LiteLLM for metadata/content extraction. Set false for dry-run.",
    )
    process_pdf_images: bool = Field(
        default=False,
        description="Caption embedded PDF images with the vision model.",
    )


class IndexFileResponse(BaseModel):
    """Result of indexing one uploaded or path-based file."""

    message: str
    file_name: str
    points: int

class IndexPathRequest(BaseModel):
    """Index a file already on disk (server-local path)."""

    path: str = Field(..., description="Absolute or project-relative file path")
    use_llm: bool = True


class SearchRequest(BaseModel):
    """Dense retrieval over Qdrant (no LLM answer)."""

    query: str
    limit: int = Field(default=4, ge=1, le=20)
    tenant_id: str | None = Field(
        default=None, description="Defaults to TENANT_ID from config"
    )
    file_names: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of document fileName values to search within. "
            "Omit or pass [] to search all documents."
        ),
    )


class SearchHit(BaseModel):
    """One retrieval hit returned to the client."""

    rank: int
    score: float | None
    file_name: str | None
    source: str | None
    content_preview: str

class SearchResponse(BaseModel):
    """Search results without generation."""

    query: str
    hits: list[SearchHit]
    file_names: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    """Full RAG: retrieve + generate answer."""

    question: str
    limit: int = Field(default=4, ge=1, le=20)
    tenant_id: str | None = None
    show_sources: bool = True
    file_names: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of document fileName values to answer from. "
            "Example: [\"product_faq.txt\", \"policy.pdf\"]. "
            "Omit or [] to use the full knowledge base."
        ),
    )

class SourceInfo(BaseModel):
    """Citation metadata for a retrieved chunk."""

    rank: int
    score: float | None
    file_name: str | None = Field(None, alias="fileName")
    source: str | None = None
    preview: str | None = None

    model_config = {"populate_by_name": True}


class AskResponse(BaseModel):
    """Grounded RAG answer plus sources."""

    question: str
    answer: str
    sources: list[dict[str, Any]]
    file_names: list[str] = Field(
        default_factory=list,
        description="Document filter that was applied (empty = all docs)",
    )

class DocumentListResponse(BaseModel):
    """Distinct indexed documents available for filtering Ask/Search."""

    tenant_id: str
    file_names: list[str]
    count: int



class DeleteSourceRequest(BaseModel):
    """Optional body for delete; path param file_name is primary."""

    tenant_id: str | None = None


class DeleteSourceResponse(BaseModel):
    """Confirmation after deleting points by fileName."""

    message: str
    file_name: str
    tenant_id: str



class CollectionInfoResponse(BaseModel):
    """Basic Qdrant collection stats."""

    name: str
    exists: bool
    points_count: int | None = None
    vectors_count: int | None = None
    status: str | None = None




class CrawlRequest(BaseModel):
    """Discover URLs then optionally index HTML pages."""

    seed: str
    max_pages: int = Field(default=5, ge=1, le=50)
    max_depth: int = Field(default=1, ge=0, le=5)
    index: bool = Field(
        default=False,
        description="If true, fetch each URL and index HTML into Qdrant",
    )
    use_llm: bool = True



class CrawlResponse(BaseModel):
    """Crawl discovery (and optional index) result."""

    seed: str
    urls: list[str]
    indexed: dict[str, int] = Field(default_factory=dict)
    total_points: int = 0
    errors: dict[str, str] = Field(
        default_factory=dict,
        description="Per-URL failures encountered while fetching or indexing",
    )
    note: str | None = Field(
        default=None,
        description="Explains why nothing was indexed, when applicable",
    )