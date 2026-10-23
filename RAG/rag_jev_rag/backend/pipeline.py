"""Agentic RAG query pipeline with decision gate."""

from __future__ import annotations

import base64
from typing import Any

import httpx

from backend.store import Candidate, DataStore
from shared.config import settings
from shared.decisions import score_combined_evidence


class ModelServerClient:
    def __init__(self) -> None:
        self.base = settings.model_server_url.rstrip("/")
        self.headers = {"X-API-Key": settings.model_server_api_key}

    def embed_query(self, text: str) -> list[float]:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                f"{self.base}/embed/query",
                headers=self.headers,
                json={"texts": [text]},
            )
            r.raise_for_status()
            return r.json()["vectors"][0]

    def rerank(self, query: str, candidates: list[Candidate]) -> list[dict[str, Any]]:
        payload = {
            "query": query,
            "candidates": [
                {
                    "id": c.id,
                    "text": c.content,
                    "modality": c.modality,
                    "page_number": c.page_number,
                }
                for c in candidates
            ],
        }
        with httpx.Client(timeout=120.0) as client:
            r = client.post(f"{self.base}/rerank", headers=self.headers, json=payload)
            r.raise_for_status()
            return r.json()["results"]


def _merge_by_page(candidates: list[Candidate]) -> list[Candidate]:
    """Keep best vector hit per (document_id, page_number, modality)."""
    best: dict[tuple[str, int, str], Candidate] = {}
    for c in candidates:
        key = (c.document_id, c.page_number, c.modality)
        if key not in best or c.distance < best[key].distance:
            best[key] = c
    return list(best.values())


def retrieve_candidates(store: DataStore, query_vector: list[float], *, top_k: int) -> list[Candidate]:
    text_hits = store.search(query_vector, n=top_k, modality="text")
    img_hits = store.search(query_vector, n=top_k, modality="image")
    return _merge_by_page(text_hits + img_hits)


def run_query_pipeline(
    store: DataStore,
    *,
    conversation_id: str,
    question: str,
    screenshot_b64: str | None = None,
) -> dict[str, Any]:
    ms = ModelServerClient()
    query_text = question
    if screenshot_b64:
        query_text = f"{question}\n[User attached a screenshot for context.]"

    q_vec = ms.embed_query(query_text)

    for attempt, top_k in enumerate([8, 16], start=1):
        candidates = retrieve_candidates(store, q_vec, top_k=top_k)
        if not candidates:
            store.log_decision(
                conversation_id,
                question,
                "escalate_no_index",
                {"attempt": attempt},
            )
            tid = store.open_ticket(conversation_id, question, [])
            return {
                "type": "ticket",
                "ticket_id": tid,
                "message": "No indexed documentation found. A support ticket was opened.",
            }

        reranked = ms.rerank(question, candidates)
        by_id = {c.id: c for c in candidates}
        survivors: list[Candidate] = []
        rejected: list[dict] = []
        for row in reranked:
            c = by_id[row["id"]]
            item = {
                "id": c.id,
                "page_number": c.page_number,
                "modality": c.modality,
                "content_preview": c.content[:400],
                "relevance": row["relevance"],
                "answerable": row["answerable"],
                "passed": row["passed"],
            }
            if row["passed"]:
                survivors.append(c)
            else:
                rejected.append(item)

        if not survivors:
            text_ranked = [
                by_id[row["id"]]
                for row in reranked
                if by_id[row["id"]].modality == "text"
            ]
            if text_ranked:
                best_rel = max(
                    row["relevance"]
                    for row in reranked
                    if by_id[row["id"]].modality == "text"
                )
                # Prefer highest decision relevance, not just vector distance.
                rel_sorted = sorted(
                    (row for row in reranked if by_id[row["id"]].modality == "text"),
                    key=lambda r: r["relevance"],
                    reverse=True,
                )
                top_text = [by_id[row["id"]] for row in rel_sorted[:5]]
                combined = "\n\n".join(
                    f"[Page {c.page_number}]\n{c.content}" for c in top_text
                )
                combo = score_combined_evidence(question, combined)
                if (
                    best_rel >= settings.text_relevance_threshold
                    and combo.answerable >= settings.text_answerable_threshold
                ):
                    survivors = top_text

        if survivors:
            answer = generate_from_evidence(question, survivors)
            store.log_decision(
                conversation_id,
                question,
                "answer",
                {"attempt": attempt, "survivors": len(survivors), "rejected": len(rejected)},
            )
            store.add_message(conversation_id, "assistant", answer["text"], meta=answer.get("citations"))
            return {"type": "answer", "answer": answer["text"], "citations": answer.get("citations", [])}

        if attempt == 1:
            continue

        store.log_decision(
            conversation_id,
            question,
            "escalate_low_evidence",
            {"rejected": rejected},
        )
        tid = store.open_ticket(conversation_id, question, rejected)
        return {
            "type": "ticket",
            "ticket_id": tid,
            "message": "Evidence was insufficient. Your question was escalated to a human agent.",
            "rejected": rejected,
        }

    raise RuntimeError("unreachable")


def generate_from_evidence(question: str, evidence: list[Candidate]) -> dict[str, Any]:
    citations = [
        {"page": c.page_number, "modality": c.modality, "snippet": c.content[:300]}
        for c in evidence[:5]
    ]
    context = "\n\n".join(
        f"[Page {c.page_number} ({c.modality})]\n{c.content}" for c in evidence[:5]
    )

    if not settings.openai_api_key:
        lines = [
            "**Demo answer (no GENERATION model key configured)**",
            "",
            f"Based on {len(evidence)} evidence item(s) that passed the Jev/d1 decision gate:",
            "",
        ]
        for cit in citations:
            lines.append(f"- Page {cit['page']} ({cit['modality']}): {cit['snippet'][:120]}…")
        lines.append("")
        lines.append(f"_Question:_ {question}")
        return {"text": "\n".join(lines), "citations": citations}

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = (
        "Answer using only the evidence below. Cite page numbers. "
        "You may synthesize a comparison when the evidence describes concepts "
        "with different wording (e.g. 'always call the LLM' vs 'decision gate'). "
        "If the evidence truly cannot support any answer, say so briefly.\n\n"
        f"Evidence:\n{context}\n\nQuestion: {question}"
    )
    completion = client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": "You are a support assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = completion.choices[0].message.content or ""
    return {"text": text, "citations": citations}
