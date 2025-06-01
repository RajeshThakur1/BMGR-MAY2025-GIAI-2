"""FastAPI web interface for the RAG application."""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os
from pathlib import Path

from src.rag.rag_pipeline import RAGPipeline
from config.settings import settings
from loguru import logger

# Initialize FastAPI app
app = FastAPI(
    title="Generic RAG Application",
    description="A generic RAG application supporting multiple LLM providers and document types",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    custom_prompt: Optional[str] = None
    include_sources: bool = True

class TextIngestionRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    source_documents: List[Dict[str, Any]] = []

class IngestionResponse(BaseModel):
    status: str
    message: str
    stats: Dict[str, Any]

class StatsResponse(BaseModel):
    vector_store: Dict[str, Any]
    current_llm_provider: str
    available_llm_providers: List[str]
    settings: Dict[str, Any]

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Generic RAG Application API",
        "version": "1.0.0",
        "docs": "/docs",
        "available_endpoints": [
            "/query",
            "/ingest/files",
            "/ingest/directory",
            "/ingest/text",
            "/search",
            "/stats",
            "/reset"
        ]
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system with a question."""
    try:
        result = rag_pipeline.query(
            question=request.question,
            custom_prompt=request.custom_prompt,
            include_sources=request.include_sources
        )
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/files", response_model=IngestionResponse)
async def ingest_files(files: List[UploadFile] = File(...)):
    """Upload and ingest files into the RAG system."""
    try:
        temp_files = []
        
        for file in files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                temp_files.append(tmp_file.name)
        
        # Ingest documents
        stats = rag_pipeline.ingest_documents(temp_files)
        
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return IngestionResponse(
            status="success",
            message=f"Ingested {stats['successful_files']} files successfully",
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"File ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/directory", response_model=IngestionResponse)
async def ingest_directory(directory_path: str = Form(...)):
    """Ingest all supported files from a directory."""
    try:
        if not os.path.exists(directory_path):
            raise HTTPException(status_code=404, detail="Directory not found")
        
        stats = rag_pipeline.ingest_directory(directory_path)
        
        return IngestionResponse(
            status="success",
            message=f"Ingested {stats['total_documents']} documents from directory",
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"Directory ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/text", response_model=IngestionResponse)
async def ingest_text(request: TextIngestionRequest):
    """Ingest raw text data into the RAG system."""
    try:
        stats = rag_pipeline.ingest_text(request.texts, request.metadatas)
        
        return IngestionResponse(
            status="success",
            message=f"Ingested {stats['total_texts']} text chunks",
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"Text ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def similarity_search(query: str, k: Optional[int] = None):
    """Perform similarity search without LLM generation."""
    try:
        results = rag_pipeline.similarity_search(query, k=k)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get pipeline statistics and configuration."""
    try:
        stats = rag_pipeline.get_stats()
        return StatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_pipeline():
    """Reset the RAG pipeline (clear all data)."""
    try:
        rag_pipeline.reset()
        return {
            "status": "success",
            "message": "Pipeline reset successfully"
        }
        
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_provider": settings.llm_provider,
        "available_providers": rag_pipeline.llm_factory.list_available_providers() if hasattr(rag_pipeline, 'llm_factory') else []
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting RAG API server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    ) 