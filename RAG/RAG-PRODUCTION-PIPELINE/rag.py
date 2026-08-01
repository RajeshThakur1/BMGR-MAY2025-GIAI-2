import json
import litellm

from config import LLM_MODEL, TENANT_ID
from store import search


RAG_SYSTEM_PROMPT = """You are a helpful knowledge-base assistant.
Answer the user's question using ONLY the provided context chunks.
Rules:
- If the context is insufficient, say you don't know based on the knowledge base.
- Cite sources using the file names mentioned in the context when possible.
- Be concise and factual. Do not invent policies or numbers."""


def format_hit_context(hit, rank: int) -> str:
    """Turn one Qdrant hit into a labeled context block for the LLM prompt.

    Args:
        hit: Qdrant scored point with `.score` and `.payload`.
        rank: 1-based rank in the retrieved list (for display).

    Returns:
        Multi-line string including rank, score, fileName, and body text.
    """

    payload = hit.payload or {}
    raw = payload.get("content", "") or ""
    file_name = payload.get("fileName", "unknown")
    source = payload.get("source", file_name)

    title, summary, body = "", "", raw
    try:
        parsed = (
            json.loads(raw)
            if isinstance(raw, str) and raw.strip().startswith("{")
            else None
        )
        if isinstance(parsed, dict):
            title = parsed.get("title") or ""
            summary = parsed.get("sum") or ""
            content = parsed.get("content", [])
            if isinstance(content, list):
                body = "\n".join(str(c) for c in content)
            elif content:
                body = str(content)
            imgs = parsed.get("imgs")
            if imgs:
                body = f"{body}\n{imgs}".strip()
    except Exception:
        pass

    header = f"[Hit {rank}] score={hit.score:.4f} | file={file_name} | source={source}"
    parts = [header]
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary}")
    parts.append(f"Content:\n{body[:2000]}")
    return "\n".join(parts)



def retrieve_context(
    question: str,
    limit: int = 4,
    tenant_id: str = TENANT_ID,
    file_names: list[str] | None = None,
) -> list:
    """Retrieve top-k relevant chunks from Qdrant for a question.

    Args:
        question: Natural-language user question.
        limit: Number of chunks to retrieve.
        tenant_id: Tenant filter for multi-tenant collections.
        file_names: Optional document filter (`fileName` list). Empty/None = all.

    Returns:
        List of Qdrant scored points (highest similarity first).
    """
    return search(
        question,
        tenant_id=tenant_id,
        limit=limit,
        file_names=file_names,
    )


def generate_answer(question: str, hits: list) -> str:
    """Generate a grounded answer from retrieved hits via LiteLLM.

    Args:
        question: Original user question.
        hits: Retrieved Qdrant points from `retrieve_context`.

    Returns:
        Model answer string. If there are no hits, returns a fixed message.
    """
    if not hits:
        return "I could not find relevant context in the knowledge base."

    context_block = "\n\n---\n\n".join(
        format_hit_context(h, i) for i, h in enumerate(hits, start=1)
    )
    resp = litellm.completion(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_block}\n\n"
                    f"Question: {question}\n\n"
                    "Answer using only the context above."
                ),
            },
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()



def rag_answer(
    question: str,
    limit: int = 4,
    tenant_id: str = TENANT_ID,
    show_sources: bool = True,
    file_names: list[str] | None = None,
) -> dict:
    """Full RAG pipeline: retrieve → augment → generate.

    Args:
        question: User question string.
        limit: How many chunks to retrieve from Qdrant.
        tenant_id: Tenant scope for search.
        show_sources: If True, print ranked hits to stdout.
        file_names: Optional list of document `fileName`s to search within.
            If None or empty, searches the whole tenant knowledge base.

    Returns:
        Dict with `question`, `answer`, `sources`, `hits`, and `file_names`.
    """
    names = [n.strip() for n in (file_names or []) if n and str(n).strip()] or None
    hits = retrieve_context(
        question, limit=limit, tenant_id=tenant_id, file_names=names
    )

    sources = []
    for i, h in enumerate(hits, start=1):
        payload = h.payload or {}
        sources.append(
            {
                "rank": i,
                "score": float(h.score) if h.score is not None else None,
                "fileName": payload.get("fileName"),
                "source": payload.get("source"),
                "preview": (payload.get("content") or "")[:240],
            }
        )

    if show_sources:
        scope = ", ".join(names) if names else "ALL documents"
        print(f"\nQuery: {question!r}")
        print(f"Document filter: {scope}")
        print(f"Retrieved {len(hits)} chunk(s):")
        for s in sources:
            score = s["score"]
            score_s = f"{score:.3f}" if score is not None else "n/a"
            print(f"  [{s['rank']}] score={score_s}  file={s['fileName']}")

    answer = generate_answer(question, hits)
    print("\nAnswer:\n", answer)
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "hits": hits,
        "file_names": names or [],
    }



def ask(
    question: str,
    limit: int = 4,
    file_names: list[str] | None = None,
) -> str:
    """Convenience wrapper that returns only the answer string.

    Args:
        question: User question.
        limit: Retrieval depth.
        file_names: Optional document filter.

    Returns:
        Generated answer text.
    """
    return rag_answer(question, limit=limit, file_names=file_names)["answer"]
