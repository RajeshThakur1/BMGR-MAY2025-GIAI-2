"""Shared FastAPI helpers and upload paths."""

from __future__ import annotations
from config import PROJECT_ROOT

UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SUFFIXES =  {".txt", ".md", ".markdown", ".csv", ".html", ".htm", ".pdf"}