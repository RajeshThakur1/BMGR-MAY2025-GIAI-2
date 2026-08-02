"""FastAPI application factory for all RAG pipeline services."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import crawl, health

def create_app() -> FastAPI:
    """Build and configure the FastAPI app with all service routers.

    Returns:
        Configured `FastAPI` instance (docs at `/docs`).
    """
    app = FastAPI(
        title="Multi-Modal RAG API",
        description=(
            "FastAPI services for indexing (TXT/MD/CSV/HTML/PDF), "
            "Qdrant search, and LiteLLM-grounded RAG answers."
        ),
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(crawl.router)
    app.include_router(health.router)

    @app.get("/", tags=["health"])
    def root():
        """API root with a short service map."""
        return {
            "service": "multi-modal-rag",
            "docs": "/docs",
            "endpoints": {
                "health": "GET /health",
                "collection": "GET /collection",
                "prepare_samples": "POST /samples/prepare",
                "index_samples": "POST /index/samples",
                "index_path": "POST /index/path",
                "index_upload": "POST /index/upload",
                "list_sources": "GET /sources",
                "search": "POST /search",
                "ask": "POST /ask",
                "delete_source": "DELETE /sources/{file_name}",
                "crawl": "POST /crawl",
            },
        }

    return app


app = create_app()