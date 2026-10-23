"""Local text embeddings (swap for EmbeddingGemma 2 on Lightning AI in production)."""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _model()
    vectors = list(model.embed(texts))
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def pseudo_embed_image(image_bytes: bytes) -> list[float]:
    """Placeholder when page PNGs are indexed without a vision encoder locally."""
    digest = hashlib.sha256(image_bytes).digest()
    rng = np.frombuffer(digest * 48, dtype=np.uint8)[:384].astype(np.float32)
    rng = (rng - 128.0) / 128.0
    norm = np.linalg.norm(rng) or 1.0
    return (rng / norm).tolist()
