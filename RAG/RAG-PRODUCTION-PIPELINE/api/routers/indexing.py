"""Indexing routes — sample directory, local path, and file upload."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.deps import ALLOWED_SUFFIXES, UPLOAD_DIR
from api.schemas import (
IndexFileResponse,
IndexPathRequest
)

from config import require_openai_key
from indexing import index_file

router = APIRouter(prefix="/index", tags=["indexing"])


@router.post("/path", response_model=IndexFileResponse)
def index_local_path(body: IndexPathRequest) -> IndexFileResponse:
    """Index a file that already exists on the server filesystem.

    Args (body):
        path: Absolute or relative path to the document.
        use_llm: Whether to call LiteLLM during extraction.
    """
    path = Path(body.path).expanduser()
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type {path.suffix}. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )
    if body.use_llm:
        try:
            require_openai_key()
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        n = index_file(path, use_llm=body.use_llm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}") from e
    return IndexFileResponse(
        message="File indexed",
        file_name=path.name,
        points=n,
    )


@router.post("/upload", response_model=IndexFileResponse)
async def index_upload(
    file: UploadFile = File(..., description="Document to index"),
    use_llm: bool = Form(default=True),
) -> IndexFileResponse:
    """Upload a document and index it into Qdrant.

    Supported extensions: `.txt`, `.md`, `.csv`, `.html`, `.pdf`.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type {suffix}. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )
    if use_llm:
        try:
            require_openai_key()
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        n = index_file(dest, use_llm=use_llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload/index failed: {e}") from e
    finally:
        await file.close()

    return IndexFileResponse(
        message="Uploaded file indexed",
        file_name=Path(file.filename).name,
        points=n,
    )
