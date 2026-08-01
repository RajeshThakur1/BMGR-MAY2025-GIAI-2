"""Configuration for the Qdrant + LiteLLM indexing / RAG pipeline.

Values are loaded from environment variables (and an optional `.env` file).
Never commit real API keys — set `OPENAI_API_KEY` in your environment.
"""


from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o-mini")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-m3")

# --- Qdrant ----
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "BIAMGR")
TENANT_ID = os.getenv("TENANT_ID", "demo_tenant")



# --- Chunking (production-aligned) ---
FALLBACK_CHUNK_SIZE = 500
FALLBACK_CHUNK_OVERLAP = 100
TXT_LINES_PER_CHUNK = 50
CSV_ROWS_PER_CHUNK = 50
MD_MIN_WORDS = 500
HTML_CHAR_LIMIT = 8000  # ~2000 tokens * 4
EMBED_TARGET_CHARS = 16000  # ~4000 tokens @ 4 chars/token (BGE-M3)

# --- Paths ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent








