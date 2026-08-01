"""BGE-M3 embedding helpers (shared by indexing and retrieval)."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


from config import EMBED_MODEL_NAME

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load and cache the SentenceTransformer embedder.

    Returns:
        Shared `SentenceTransformer` instance for BGE-M3 (or configured model).
    """
    return SentenceTransformer(EMBED_MODEL_NAME)


def vector_size() -> int:
    """Return embedding dimensionality for Qdrant collection creation.

    Returns:
        Integer vector size from the loaded embedder.
    """
    return get_embedder().get_sentence_embedding_dimension()


def normalize_for_embed(content: str) -> str:
    """Collapse whitespace/newlines before embedding (production behavior).

    Args:
        content: Raw chunk or query string.

    Returns:
        Single-line string with repeated whitespace removed.
    """
    return " ".join(content.replace("\n", " ").split())


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed one or more strings with BGE-M3 (L2-normalized).

    Args:
        texts: Strings to embed. Prefer pre-normalized text via
            `normalize_for_embed` for consistency with production.

    Returns:
        List of float vectors (one per input), L2-normalized for cosine search.
    """
    vectors = get_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=True
    )
    return [v.tolist() for v in vectors]


# if __name__ == "__main__":
#     # Quick test of the embedder and vector size
#     print(f"Embedding model: {EMBED_MODEL_NAME}")
#     print(f"Vector size: {vector_size()}")
#     test_texts = ["Hello world!", "This is a test.", "BGE-M3 embeddings are cool."]
#     embeddings = embed_texts(test_texts)
#     for text, vec in zip(test_texts, embeddings):
#         print(f"Text: {text}\nVector length: {len(vec)}\n")