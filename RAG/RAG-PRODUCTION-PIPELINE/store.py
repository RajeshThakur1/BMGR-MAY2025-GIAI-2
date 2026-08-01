"""Qdrant collection management, upsert, search, and delete helpers."""

from __future__ import annotations

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


from config import COLLECTION, QDRANT_URL, TENANT_ID
from embeddings import embed_texts, normalize_for_embed, vector_size

@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    """Create a cached Qdrant client for `QDRANT_URL`.

    Returns:
        Connected `QdrantClient` instance.
    """
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(name: str = COLLECTION, dim: int | None = None) -> None:
    """Create the Qdrant collection and payload indexes if missing.

    Vectors use cosine distance. Keyword indexes on `fileName`, `source`, and
    `tenant_id` enable filtered search and delete-by-filename.

    Args:
        name: Collection name (default: `COLLECTION`).
        dim: Vector size. Defaults to the embedding model dimension.
    """
    if dim is None:
        dim = vector_size()
    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        print(f"Collection '{name}' already exists")
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )
    for field in ("fileName", "source", "tenant_id"):
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
    print(f"Created collection '{name}' (dim={dim}, cosine)")


def index_documents(
    docs: list[dict],
    file_name: str,
    tenant_id: str = TENANT_ID,
    collection: str = COLLECTION,
) -> int:
    """Embed documents and upsert them into Qdrant.

    Args:
        docs: List of dicts with `page_content` and optional `metadata`.
        file_name: Logical source name stored on every point (used for delete).
        tenant_id: Tenant filter value (default: `TENANT_ID`).
        collection: Target Qdrant collection.

    Returns:
        Number of points upserted.
    """
    ensure_collection(collection)
    texts = [normalize_for_embed(d["page_content"]) for d in docs]
    vectors = embed_texts(texts)
    points = []
    for doc, vec in zip(docs, vectors):
        points.append(
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "content": doc["page_content"],
                    "fileName": file_name,
                    "source": doc.get("metadata", {}).get("source", file_name),
                    "tenant_id": tenant_id,
                },
            )
        )
    get_qdrant().upsert(collection_name=collection, points=points)
    return len(points)



def search(
    query: str,
    tenant_id: str = TENANT_ID,
    limit: int = 3,
    file_names: list[str] | None = None,
):
    """Dense vector search over indexed chunks for a natural-language query.

    Args:
        query: User question or search phrase.
        tenant_id: Restrict hits to this tenant.
        limit: Max number of points to return.
        file_names: Optional list of `fileName` values to restrict search to.
            If None or empty, all documents in the tenant are searched.

    Returns:
        List of Qdrant scored points (`id`, `score`, `payload`).
    """
    must = [
        qmodels.FieldCondition(
            key="tenant_id", match=qmodels.MatchValue(value=tenant_id)
        )
    ]
    names = [n.strip() for n in (file_names or []) if n and str(n).strip()]
    if names:
        must.append(
            qmodels.FieldCondition(
                key="fileName",
                match=qmodels.MatchAny(any=names),
            )
        )

    vec = embed_texts([normalize_for_embed(query)])[0]
    result = get_qdrant().query_points(
        collection_name=COLLECTION,
        query=vec,
        query_filter=qmodels.Filter(must=must),
        limit=limit,
        with_payload=True,
    )
    return result.points


def list_filenames(tenant_id: str = TENANT_ID) -> list[str]:
    """Return distinct `fileName` values indexed for a tenant.

    Useful for UIs that let the user pick which documents to query.

    Args:
        tenant_id: Tenant scope (default: `TENANT_ID`).

    Returns:
        Sorted list of unique file names.
    """
    client = get_qdrant()
    names: set[str] = set()
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="tenant_id",
                        match=qmodels.MatchValue(value=tenant_id),
                    )
                ]
            ),
            limit=256,
            offset=next_offset,
            with_payload=["fileName"],
            with_vectors=False,
        )
        for p in points:
            fn = (p.payload or {}).get("fileName")
            if fn:
                names.add(str(fn))
        if next_offset is None:
            break
    return sorted(names)


def delete_by_filename(file_name: str, tenant_id: str = TENANT_ID) -> None:
    """Delete all Qdrant points for a given file within a tenant.

    Args:
        file_name: Exact `fileName` payload value used at upsert time.
        tenant_id: Tenant scope (default: `TENANT_ID`).
    """
    get_qdrant().delete(
        collection_name=COLLECTION,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="fileName", match=qmodels.MatchValue(value=file_name)
                    ),
                    qmodels.FieldCondition(
                        key="tenant_id", match=qmodels.MatchValue(value=tenant_id)
                    ),
                ]
            )
        ),
    )
    print(f"Deleted points where fileName={file_name!r} tenant={tenant_id!r}")