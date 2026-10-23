"""Structured judgments via OpenAI Decisions API (or optional Liquid d1) — with mock fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from shared.config import settings

Provider = Literal["mock", "openai", "liquid"]


@dataclass
class RerankScores:
    relevance: float
    answerable: float
    provider: str
    raw: dict[str, Any] | None = None


def _mock_scores(query: str, passage: str, modality: str) -> RerankScores:
    """Keyword overlap heuristic so the demo runs without API keys."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    p = set(re.findall(r"[a-z0-9]+", passage.lower()))
    if not q:
        return RerankScores(0.2, 0.15, "mock")
    overlap = len(q & p) / max(len(q), 1)
    relevance = min(0.95, 0.25 + overlap * 0.9)
    answerable = min(0.95, 0.2 + overlap * 0.85)
    if modality == "image":
        answerable *= 0.85
    return RerankScores(relevance, answerable, "mock")


def _predicate_probability(answers: list[Any], name: str) -> float:
    for ans in answers:
        ans_name = getattr(ans, "name", None) if not isinstance(ans, dict) else ans.get("name")
        if ans_name != name:
            continue
        ans_type = getattr(ans, "type", None) if not isinstance(ans, dict) else ans.get("type")
        if ans_type == "refusal":
            return 0.0
        if isinstance(ans, dict):
            return float(ans.get("probability", 0.0))
        return float(getattr(ans, "probability", 0.0))
    return 0.0


def _openai_decision(input_text: str, questions: list[dict[str, str]]) -> tuple[float, float, str]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.decisions.create(
        model=settings.decision_model,
        input=input_text,
        questions=questions,
    )
    rel = _predicate_probability(response.answers, "relevant") if any(
        q.get("name") == "relevant" for q in questions
    ) else 0.0
    ans = _predicate_probability(response.answers, "answerable") if any(
        q.get("name") == "answerable" for q in questions
    ) else 0.0
    return rel, ans, response.model


def _openai_scores(state: str) -> RerankScores:
    rel, ans, model = _openai_decision(
        state,
        [
            {
                "type": "predicate",
                "name": "relevant",
                "instructions": (
                    "Is this passage relevant to the customer's query "
                    "(same topic, terminology, or comparison)?"
                ),
            },
            {
                "type": "predicate",
                "name": "answerable",
                "instructions": (
                    "Does this passage contain useful evidence that could support "
                    "an answer to the query (it need not fully answer alone)?"
                ),
            },
        ],
    )
    return RerankScores(rel, ans, "openai", raw={"model": model})


def _liquid_scores(state: str) -> RerankScores:
    payload = {
        "model": "d1",
        "state": state,
        "questions": {
            "relevant": {
                "type": "noul",
                "instructions": "Is this passage relevant to answering the customer's query?",
            },
            "answerable": {
                "type": "noul",
                "instructions": "Does this passage contain enough information to fully answer the query?",
            },
        },
    }
    with httpx.Client(timeout=60.0) as client:
        r = client.post(
            "https://api.liquid.ai/decisions/v1/systemone",
            headers={
                "Authorization": f"Bearer {settings.liquid_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
    answers = data.get("answers") or {}

    def _noul(key: str) -> float:
        val = answers[key]
        if isinstance(val, dict):
            return float(val.get("noul", 0.0))
        return float(val)

    rel = _noul("relevant")
    ans = _noul("answerable")
    return RerankScores(rel, ans, "liquid", raw=data)


def score_candidate(
    query: str,
    candidate_text: str,
    *,
    modality: Literal["text", "image"] = "text",
    page_number: int | None = None,
    provider: Provider | None = None,
) -> RerankScores:
    provider = provider or settings.decision_provider  # type: ignore[assignment]
    if provider == "mock":
        return _mock_scores(query, candidate_text, modality)
    if provider == "openai" and not settings.openai_api_key:
        return _mock_scores(query, candidate_text, modality)
    if provider == "liquid" and not settings.liquid_api_key:
        return _mock_scores(query, candidate_text, modality)

    state = {
        "query": query,
        "modality": modality,
        "page_number": page_number,
        "passage": candidate_text[:8000],
    }
    state_str = json.dumps(state, ensure_ascii=False)

    if provider == "liquid":
        return _liquid_scores(state_str)
    return _openai_scores(state_str)


def score_combined_evidence(query: str, combined_passages: str) -> RerankScores:
    """Gate on merged top-k chunks (conceptual answers rarely fit one chunk)."""
    provider = settings.decision_provider
    if provider == "mock" or (provider == "openai" and not settings.openai_api_key):
        return _mock_scores(query, combined_passages, "text")
    if provider == "liquid" and not settings.liquid_api_key:
        return _mock_scores(query, combined_passages, "text")

    state = json.dumps(
        {"query": query, "passages": combined_passages[:24000]},
        ensure_ascii=False,
    )
    if provider == "liquid":
        return _liquid_scores(state)

    _, ans, model = _openai_decision(
        state,
        [
            {
                "type": "predicate",
                "name": "answerable",
                "instructions": (
                    "Taken together, do these passages contain enough information "
                    "to answer the customer's query?"
                ),
            },
        ],
    )
    rel = 1.0
    return RerankScores(rel, ans, "openai", raw={"model": model, "combined": True})
