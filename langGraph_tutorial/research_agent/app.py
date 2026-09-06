"""
Gradio test harness for the Team Research Bot (Researcher -> Writer -> Fact-Checker -> Editor).

This file ports the pipeline built in team_research_bot.ipynb into a standalone module
(schemas, prompts, source loader, LLM client, agent functions, LangGraph state/nodes/graph)
and wraps it with a Gradio UI so every piece of the pipeline can be exercised interactively:

- Sources tab      -> load/browse the local source pack
- Full Pipeline tab -> run the compiled LangGraph end-to-end (incl. the revision loop)
- Step-by-Step tab  -> call each agent individually and inspect the Pydantic contract it returns
- Trace tab         -> inspect AgentMessage handoffs from the last run and saved trace files
- Graph tab         -> view the compiled graph structure
- Prompts tab       -> edit each agent's system prompt and re-run with the changes

Run with:  python app.py
"""

from __future__ import annotations

import json
import operator
import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict, TypeVar
from uuid import uuid4

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_DIR / "source_pack"
TRACE_DIR = PROJECT_ROOT / "traces"
TRACE_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))

DEFAULT_QUERY = "What should an organization consider before using AI agents in customer support?"


# ---------------------------------------------------------------------------
# 2. Pydantic schemas (inter-agent contracts) — identical to the notebook
# ---------------------------------------------------------------------------

class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    WRITER = "writer"
    FACT_CHECKER = "fact_checker"
    EDITOR = "editor"


class MessageType(str, Enum):
    TASK = "task"
    RESULT = "result"
    CRITIQUE = "critique"
    FINAL = "final"


class Status(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    NEED_REVISION = "need_revision"
    COMPLETED = "completed"


class StrictBaseModel(BaseModel):
    """Rejects unknown fields — enforces schema discipline on LLM output."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvidenceItem(StrictBaseModel):
    source_id: str
    title: str
    snippet: str = Field(..., min_length=30)
    relevance_score: float = Field(..., ge=0, le=1)


class ResearchFinding(StrictBaseModel):
    claim: str = Field(..., min_length=20)
    evidence: list[EvidenceItem] = Field(..., min_length=1)
    confidence: float = Field(..., ge=0, le=1)
    limitations: str


class ResearchReport(StrictBaseModel):
    topic: str
    findings: list[ResearchFinding] = Field(..., min_length=2)
    unresolved_questions: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(..., ge=0, le=1)


class DraftSection(StrictBaseModel):
    heading: str
    content: str = Field(..., min_length=80)
    claim_ids: list[int]


class DraftReport(StrictBaseModel):
    title: str
    executive_summary: str = Field(..., min_length=80)
    sections: list[DraftSection] = Field(..., min_length=2)
    risks_or_uncertainties: list[str] = Field(default_factory=list)
    source_ids_used: list[str] = Field(default_factory=list)


class ClaimCheck(StrictBaseModel):
    claim: str
    verdict: Literal["supported", "partially_supported", "unsupported"]
    evidence_refs: list[str] = Field(default_factory=list)
    issue: str
    recommendation: str


class FactCheckReport(StrictBaseModel):
    checks: list[ClaimCheck] = Field(..., min_length=1)
    summary: str
    revision_required: bool
    overall_reliability: float = Field(..., ge=0, le=1)


class FinalReport(StrictBaseModel):
    title: str
    final_answer: str = Field(..., min_length=150)
    key_takeaways: list[str] = Field(..., min_length=3)
    caveats: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    editor_notes: str


class AgentMessage(StrictBaseModel):
    trace_id: str
    sender: AgentRole
    receiver: AgentRole
    message_type: MessageType
    task: str
    payload: dict[str, Any]
    confidence: float = Field(..., ge=0, le=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("trace_id")
    @classmethod
    def trace_id_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("trace_id cannot be blank")
        return value


def new_trace_id(prefix: str = "trace") -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


def make_message(
    sender: AgentRole,
    receiver: AgentRole,
    message_type: MessageType,
    task: str,
    payload_model: BaseModel,
    confidence: float,
) -> AgentMessage:
    return AgentMessage(
        trace_id=new_trace_id("msg"),
        sender=sender,
        receiver=receiver,
        message_type=message_type,
        task=task,
        payload=payload_model.model_dump(),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# 3. Prompts — kept in a mutable dict so the "Prompts" tab can edit them live
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS: dict[str, str] = {
    "researcher": """
You are the Researcher agent in a multi-agent research team.
Your job is to extract evidence-backed findings from the provided local source pack.

Rules:
- Use only the provided sources.
- Every finding must include at least one evidence item.
- Be honest about limitations.
- Return valid JSON that conforms exactly to the provided schema.
""".strip(),
    "writer": """
You are the Writer agent in a multi-agent research team.
Your job is to turn structured research findings into a concise, readable draft.

Rules:
- Do not invent claims beyond the research report.
- Keep the draft clear for a mixed technical/business audience.
- Mark uncertainty where the evidence is incomplete.
- Return valid JSON that conforms exactly to the provided schema.
""".strip(),
    "fact_checker": """
You are the Fact-Checker agent in a multi-agent research team.
Your job is to compare the draft against the research evidence.

Rules:
- Verify whether each major claim is supported by the Researcher output.
- Mark claims as supported, partially_supported, or unsupported.
- Recommend concrete revisions, not vague criticism.
- Return valid JSON that conforms exactly to the provided schema.
""".strip(),
    "editor": """
You are the Editor agent in a multi-agent research team.
Your job is to produce the final answer using the draft and fact-check report.

Rules:
- Remove or soften unsupported claims.
- Preserve useful nuance and caveats.
- Make the answer polished but evidence-grounded.
- Return valid JSON that conforms exactly to the provided schema.
""".strip(),
}

# Live copy the agents actually read. The Prompts tab mutates this dict in place.
PROMPTS: dict[str, str] = dict(DEFAULT_PROMPTS)


def schema_instruction(schema_json: str) -> str:
    return (
        "Return only JSON. Do not include markdown fences. "
        "The JSON must match this schema:\n"
        f"{schema_json}"
    )


# ---------------------------------------------------------------------------
# 4. Source loader
# ---------------------------------------------------------------------------

@dataclass
class SourceDocument:
    source_id: str
    title: str
    path: Path
    text: str


def _extract_title(markdown_text: str, fallback: str) -> str:
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            return line.replace("# ", "").strip()
    return fallback


def load_source_pack(source_dir: str | Path = SOURCE_DIR) -> list[SourceDocument]:
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source pack directory not found: {source_path}")

    index_path = source_path / "source_index.json"
    documents: list[SourceDocument] = []
    if index_path.exists():
        index_rows = json.loads(index_path.read_text(encoding="utf-8"))
        for row in index_rows:
            path = source_path / row["file"]
            text = path.read_text(encoding="utf-8")
            documents.append(
                SourceDocument(
                    source_id=row["source_id"],
                    title=_extract_title(text, path.stem.replace("_", " ").title()),
                    path=path,
                    text=text,
                )
            )
    else:
        for index, path in enumerate(sorted(source_path.glob("*.md")), start=1):
            text = path.read_text(encoding="utf-8")
            documents.append(
                SourceDocument(
                    source_id=f"SRC-{index:03d}",
                    title=_extract_title(text, path.stem.replace("_", " ").title()),
                    path=path,
                    text=text,
                )
            )
    if not documents:
        raise ValueError(f"No markdown sources found in {source_path}")
    return documents


def render_sources_for_prompt(documents: list[SourceDocument], max_chars_per_doc: int = 1800) -> str:
    blocks = []
    for doc in documents:
        clipped_text = doc.text[:max_chars_per_doc]
        blocks.append(f"Source ID: {doc.source_id}\nTitle: {doc.title}\nText:\n{clipped_text}\n")
    return "\n---\n".join(blocks)


# ---------------------------------------------------------------------------
# 5. LLM client (OpenAI structured JSON)
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


class StructuredLLMClient:
    """Calls OpenAI and validates every response against a Pydantic schema."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or OPENAI_MODEL
        self.api_key = OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required. Add it to .env.")
        self.client = OpenAI(api_key=self.api_key)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[T],
        temperature: float = 0.2,
    ) -> T:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_content = response.choices[0].message.content or "{}"
        try:
            return response_schema.model_validate_json(raw_content)
        except ValidationError as original_error:
            return self._repair_json(raw_content, response_schema, str(original_error))

    def _repair_json(self, raw_content: str, response_schema: type[T], original_error: str) -> T:
        repair_prompt = (
            "The previous output failed validation. Repair it to match the schema. "
            "Return only JSON.\n\n"
            f"Validation error:\n{original_error}\n\n"
            f"Target schema:\n{json.dumps(response_schema.model_json_schema(), indent=2)}\n\n"
            f"Invalid output:\n{raw_content}"
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You repair JSON to match schemas."},
                {"role": "user", "content": repair_prompt},
            ],
        )
        repaired_content = response.choices[0].message.content or "{}"
        return response_schema.model_validate_json(repaired_content)


@lru_cache(maxsize=8)
def get_llm_client(model_name: str) -> StructuredLLMClient:
    return StructuredLLMClient(model_name=model_name)


# ---------------------------------------------------------------------------
# 6. Agent functions
# ---------------------------------------------------------------------------

def run_researcher(query: str, docs: list[SourceDocument], model_name: str = OPENAI_MODEL) -> ResearchReport:
    sources_block = render_sources_for_prompt(docs)
    schema_json = json.dumps(ResearchReport.model_json_schema(), indent=2)
    user_prompt = (
        f"Research question:\n{query}\n\n"
        f"Local source pack:\n{sources_block}\n\n"
        f"{schema_instruction(schema_json)}"
    )
    return get_llm_client(model_name).complete_json(
        system_prompt=PROMPTS["researcher"],
        user_prompt=user_prompt,
        response_schema=ResearchReport,
    )


def run_writer(
    query: str,
    research_report: ResearchReport,
    fact_check_report: FactCheckReport | None,
    model_name: str = OPENAI_MODEL,
) -> DraftReport:
    schema_json = json.dumps(DraftReport.model_json_schema(), indent=2)
    research_json = json.dumps(research_report.model_dump(), indent=2)

    if fact_check_report is None:
        user_prompt = (
            f"Original question:\n{query}\n\n"
            f"Research report (Researcher output):\n{research_json}\n\n"
            "Write the first draft report grounded strictly in the findings above.\n\n"
            f"{schema_instruction(schema_json)}"
        )
    else:
        fact_check_json = json.dumps(fact_check_report.model_dump(), indent=2)
        user_prompt = (
            f"Original question:\n{query}\n\n"
            f"Research report (Researcher output):\n{research_json}\n\n"
            f"Fact-Checker critique of your previous draft:\n{fact_check_json}\n\n"
            "Revise the draft to address every 'unsupported' or 'partially_supported' "
            "claim above while staying grounded in the research report.\n\n"
            f"{schema_instruction(schema_json)}"
        )

    return get_llm_client(model_name).complete_json(
        system_prompt=PROMPTS["writer"],
        user_prompt=user_prompt,
        response_schema=DraftReport,
    )


def run_fact_checker(
    draft_report: DraftReport,
    research_report: ResearchReport,
    model_name: str = OPENAI_MODEL,
) -> FactCheckReport:
    schema_json = json.dumps(FactCheckReport.model_json_schema(), indent=2)
    draft_json = json.dumps(draft_report.model_dump(), indent=2)
    research_json = json.dumps(research_report.model_dump(), indent=2)
    user_prompt = (
        f"Draft report to verify (Writer output):\n{draft_json}\n\n"
        f"Research report evidence to check against (Researcher output):\n{research_json}\n\n"
        "Check every claim in the draft against the research evidence. Set "
        "revision_required=True if any claim is 'partially_supported' or 'unsupported'.\n\n"
        f"{schema_instruction(schema_json)}"
    )
    return get_llm_client(model_name).complete_json(
        system_prompt=PROMPTS["fact_checker"],
        user_prompt=user_prompt,
        response_schema=FactCheckReport,
    )


def run_editor(
    draft_report: DraftReport,
    fact_check_report: FactCheckReport,
    model_name: str = OPENAI_MODEL,
) -> FinalReport:
    schema_json = json.dumps(FinalReport.model_json_schema(), indent=2)
    draft_json = json.dumps(draft_report.model_dump(), indent=2)
    fact_check_json = json.dumps(fact_check_report.model_dump(), indent=2)
    user_prompt = (
        f"Latest draft report (Writer output):\n{draft_json}\n\n"
        f"Fact-Checker report (verdicts and recommendations):\n{fact_check_json}\n\n"
        "Produce the final answer for the user: remove or soften any claim marked "
        "'unsupported', keep caveats from the fact-check visible, and preserve "
        "useful nuance from the draft.\n\n"
        f"{schema_instruction(schema_json)}"
    )
    return get_llm_client(model_name).complete_json(
        system_prompt=PROMPTS["editor"],
        user_prompt=user_prompt,
        response_schema=FinalReport,
    )


# ---------------------------------------------------------------------------
# 7. LangGraph state and nodes
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    run_id: str
    user_query: str
    documents: list[SourceDocument]
    source_ids_consulted: list[str]
    research_report: ResearchReport | None
    draft_report: DraftReport | None
    fact_check_report: FactCheckReport | None
    final_report: FinalReport | None
    revision_count: int
    max_revisions: int
    model_name: str
    status: str
    messages: Annotated[list[AgentMessage], operator.add]
    errors: Annotated[list[str], operator.add]


def researcher_node(state: GraphState) -> dict[str, Any]:
    research = run_researcher(state["user_query"], state["documents"], state.get("model_name", OPENAI_MODEL))
    source_ids = sorted(
        {evidence.source_id for finding in research.findings for evidence in finding.evidence}
    )
    message = make_message(
        sender=AgentRole.RESEARCHER,
        receiver=AgentRole.WRITER,
        message_type=MessageType.RESULT,
        task="use these evidence-based findings to draft a research brief",
        payload_model=research,
        confidence=research.overall_confidence,
    )
    return {
        "research_report": research,
        "source_ids_consulted": source_ids,
        "messages": [message],
        "status": Status.RUNNING.value,
    }


def writer_node(state: GraphState) -> dict[str, Any]:
    assert state["research_report"] is not None
    updates: dict[str, Any] = {}
    fact_check = state.get("fact_check_report")
    revision_count = state.get("revision_count", 0)

    if (
        state.get("draft_report")
        and fact_check
        and fact_check.revision_required
        and revision_count < state["max_revisions"]
    ):
        updates["revision_count"] = revision_count + 1
        updates["status"] = Status.NEED_REVISION.value

    draft = run_writer(
        state["user_query"], state["research_report"], fact_check, state.get("model_name", OPENAI_MODEL)
    )
    message = make_message(
        sender=AgentRole.WRITER,
        receiver=AgentRole.FACT_CHECKER,
        message_type=MessageType.RESULT,
        task="verify the draft report against the research evidence",
        payload_model=draft,
        confidence=0.78 if fact_check else 0.72,
    )
    updates.update({"draft_report": draft, "messages": [message]})
    return updates


def fact_checker_node(state: GraphState) -> dict[str, Any]:
    assert state["draft_report"] is not None and state["research_report"] is not None
    fact_check = run_fact_checker(
        state["draft_report"], state["research_report"], state.get("model_name", OPENAI_MODEL)
    )
    receiver = AgentRole.WRITER if fact_check.revision_required else AgentRole.EDITOR
    message = make_message(
        sender=AgentRole.FACT_CHECKER,
        receiver=receiver,
        message_type=MessageType.CRITIQUE,
        task="Revise the draft if needed; otherwise prepare final editing.",
        payload_model=fact_check,
        confidence=fact_check.overall_reliability,
    )
    return {"fact_check_report": fact_check, "messages": [message]}


def editor_node(state: GraphState) -> dict[str, Any]:
    assert state["draft_report"] is not None and state["fact_check_report"] is not None
    final = run_editor(state["draft_report"], state["fact_check_report"], state.get("model_name", OPENAI_MODEL))
    message = make_message(
        sender=AgentRole.EDITOR,
        receiver=AgentRole.ORCHESTRATOR,
        message_type=MessageType.FINAL,
        task="Return the final answer to the user.",
        payload_model=final,
        confidence=state["fact_check_report"].overall_reliability,
    )
    return {"final_report": final, "messages": [message], "status": Status.COMPLETED.value}


def route_after_fact_check(state: GraphState) -> Literal["writer", "editor"]:
    fact_check = state.get("fact_check_report")
    if (
        fact_check
        and fact_check.revision_required
        and state.get("revision_count", 0) < state.get("max_revisions", 1)
    ):
        return "writer"
    return "editor"


# ---------------------------------------------------------------------------
# 8. Build and compile the graph (once, at import time)
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("editor", editor_node)

    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "fact_checker")
    workflow.add_conditional_edges(
        "fact_checker",
        route_after_fact_check,
        {"writer": "writer", "editor": "editor"},
    )
    workflow.add_edge("editor", END)
    return workflow.compile()


research_graph = build_graph()


# ---------------------------------------------------------------------------
# 9. Helpers shared by the Gradio callbacks
# ---------------------------------------------------------------------------

def to_jsonable(model: BaseModel | None) -> dict[str, Any]:
    """Convert a Pydantic model (Enums included) into a plain JSON-safe dict."""
    if model is None:
        return {}
    return json.loads(model.model_dump_json())


def messages_to_dataframe(messages: list[AgentMessage]) -> pd.DataFrame:
    if not messages:
        return pd.DataFrame(
            columns=["#", "trace_id", "sender", "receiver", "type", "task", "confidence", "created_at", "payload"]
        )
    rows = []
    for i, m in enumerate(messages, start=1):
        rows.append(
            {
                "#": i,
                "trace_id": m.trace_id,
                "sender": m.sender.value,
                "receiver": m.receiver.value,
                "type": m.message_type.value,
                "task": m.task,
                "confidence": round(m.confidence, 3),
                "created_at": m.created_at,
                "payload": json.dumps(m.payload, indent=None)[:400],
            }
        )
    return pd.DataFrame(rows)


def check_setup() -> str:
    lines = []
    lines.append(f"**OPENAI_API_KEY**: {'set ✅' if OPENAI_API_KEY else 'MISSING ❌ — add it to .env'}")
    lines.append(f"**OPENAI_MODEL**: `{OPENAI_MODEL}`")
    lines.append(f"**Source pack**: `{SOURCE_DIR}` {'✅ found' if SOURCE_DIR.exists() else '❌ missing'}")
    lines.append(f"**Trace dir**: `{TRACE_DIR}`")
    return "\n\n".join(lines)


def require_api_key():
    if not OPENAI_API_KEY:
        raise gr.Error("OPENAI_API_KEY is not set. Add it to your .env file and restart the app.")


def require_documents(documents: list[SourceDocument] | None) -> list[SourceDocument]:
    if not documents:
        documents = load_source_pack()
    return documents


# ---------------------------------------------------------------------------
# 10. Gradio callbacks — Sources tab
# ---------------------------------------------------------------------------

def ui_load_sources():
    docs = load_source_pack()
    df = pd.DataFrame(
        [{"source_id": d.source_id, "title": d.title, "chars": len(d.text), "file": d.path.name} for d in docs]
    )
    choices = [d.source_id for d in docs]
    status = f"Loaded {len(docs)} source document(s) from `{SOURCE_DIR}`."
    return docs, df, gr.update(choices=choices, value=choices[0] if choices else None), status


def ui_preview_source(source_id: str, documents: list[SourceDocument] | None):
    if not documents or not source_id:
        return ""
    for d in documents:
        if d.source_id == source_id:
            return f"# {d.title}  ({d.source_id})\n\n{d.text}"
    return "Source not found — reload the source pack."


# ---------------------------------------------------------------------------
# 11. Gradio callbacks — Full pipeline tab
# ---------------------------------------------------------------------------

def ui_run_pipeline(
    query: str,
    max_revisions: int,
    model_name: str,
    save_trace: bool,
    documents: list[SourceDocument] | None,
):
    require_api_key()
    if not query or not query.strip():
        raise gr.Error("Enter a research question first.")

    documents = require_documents(documents)
    run_id = f"run_{uuid4().hex[:10]}"

    initial_state: GraphState = {
        "run_id": run_id,
        "user_query": query.strip(),
        "documents": documents,
        "source_ids_consulted": [],
        "research_report": None,
        "draft_report": None,
        "fact_check_report": None,
        "final_report": None,
        "revision_count": 0,
        "max_revisions": int(max_revisions),
        "model_name": model_name.strip() or OPENAI_MODEL,
        "status": Status.CREATED.value,
        "messages": [],
        "errors": [],
    }

    try:
        final_state = research_graph.invoke(initial_state)
    except Exception as exc:  # noqa: BLE001 — surface any pipeline failure to the UI
        raise gr.Error(f"Pipeline failed: {exc}\n\n{traceback.format_exc(limit=3)}")

    final_report: FinalReport | None = final_state.get("final_report")
    messages: list[AgentMessage] = final_state.get("messages", [])

    status_md = (
        f"**Run ID:** `{final_state['run_id']}`  \n"
        f"**Status:** `{final_state['status']}`  \n"
        f"**Revisions used:** {final_state['revision_count']} / {final_state['max_revisions']}  \n"
        f"**Messages passed:** {len(messages)}  \n"
        f"**Sources consulted:** {', '.join(final_state.get('source_ids_consulted', [])) or '—'}"
    )

    if final_report:
        final_md = f"## {final_report.title}\n\n{final_report.final_answer}"
        takeaways = "\n".join(f"- {t}" for t in final_report.key_takeaways) or "_none_"
        caveats = "\n".join(f"- {c}" for c in final_report.caveats) or "_none_"
        references = "\n".join(f"- {r}" for r in final_report.references) or "_none_"
        editor_notes = final_report.editor_notes
    else:
        final_md, takeaways, caveats, references, editor_notes = "_No final report produced._", "", "", "", ""

    trace_path_msg = ""
    if save_trace:
        trace_file = TRACE_DIR / f"{run_id}.json"
        trace_payload = {
            "run_id": final_state["run_id"],
            "user_query": final_state["user_query"],
            "status": final_state["status"],
            "revision_count": final_state["revision_count"],
            "max_revisions": final_state["max_revisions"],
            "model_name": final_state["model_name"],
            "source_ids_consulted": final_state.get("source_ids_consulted", []),
            "research_report": to_jsonable(final_state.get("research_report")),
            "draft_report": to_jsonable(final_state.get("draft_report")),
            "fact_check_report": to_jsonable(final_state.get("fact_check_report")),
            "final_report": to_jsonable(final_state.get("final_report")),
            "messages": [json.loads(m.model_dump_json()) for m in messages],
        }
        trace_file.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
        trace_path_msg = f"Trace saved to `{trace_file}`"

    return (
        status_md,
        final_md,
        takeaways,
        caveats,
        references,
        editor_notes,
        to_jsonable(final_state.get("research_report")),
        to_jsonable(final_state.get("draft_report")),
        to_jsonable(final_state.get("fact_check_report")),
        to_jsonable(final_state.get("final_report")),
        messages_to_dataframe(messages),
        trace_path_msg,
        messages,  # kept in a State for the Trace tab
    )


# ---------------------------------------------------------------------------
# 12. Gradio callbacks — Step-by-step tab
# ---------------------------------------------------------------------------

def ui_step_researcher(query: str, model_name: str, documents: list[SourceDocument] | None):
    require_api_key()
    if not query or not query.strip():
        raise gr.Error("Enter a research question first.")
    documents = require_documents(documents)
    try:
        research = run_researcher(query.strip(), documents, model_name.strip() or OPENAI_MODEL)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Researcher failed: {exc}")
    summary = f"✅ Researcher produced {len(research.findings)} finding(s), overall_confidence={research.overall_confidence:.2f}"
    return research, to_jsonable(research), summary, documents


def ui_step_writer_initial(query: str, model_name: str, research: ResearchReport | None):
    require_api_key()
    if research is None:
        raise gr.Error("Run the Researcher step first.")
    try:
        draft = run_writer(query.strip(), research, None, model_name.strip() or OPENAI_MODEL)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Writer failed: {exc}")
    summary = f"✅ Writer produced draft '{draft.title}' with {len(draft.sections)} section(s)"
    return draft, to_jsonable(draft), summary


def ui_step_fact_checker(draft: DraftReport | None, research: ResearchReport | None, model_name: str):
    require_api_key()
    if draft is None or research is None:
        raise gr.Error("Run the Researcher and Writer steps first.")
    try:
        fact_check = run_fact_checker(draft, research, model_name.strip() or OPENAI_MODEL)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Fact-Checker failed: {exc}")
    verdict = "🔁 REVISION REQUIRED" if fact_check.revision_required else "✅ No revision needed"
    summary = f"{verdict} — overall_reliability={fact_check.overall_reliability:.2f}, {len(fact_check.checks)} claim(s) checked"
    return fact_check, to_jsonable(fact_check), summary


def ui_step_writer_revise(
    query: str, model_name: str, research: ResearchReport | None, fact_check: FactCheckReport | None
):
    require_api_key()
    if research is None or fact_check is None:
        raise gr.Error("Run the Researcher and Fact-Checker steps first.")
    try:
        draft = run_writer(query.strip(), research, fact_check, model_name.strip() or OPENAI_MODEL)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Writer (revision) failed: {exc}")
    summary = f"✅ Revised draft '{draft.title}' with {len(draft.sections)} section(s)"
    return draft, to_jsonable(draft), summary


def ui_step_editor(draft: DraftReport | None, fact_check: FactCheckReport | None, model_name: str):
    require_api_key()
    if draft is None or fact_check is None:
        raise gr.Error("Run the Writer and Fact-Checker steps first.")
    try:
        final = run_editor(draft, fact_check, model_name.strip() or OPENAI_MODEL)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Editor failed: {exc}")
    final_md = f"## {final.title}\n\n{final.final_answer}"
    summary = "✅ Final report produced"
    return final, to_jsonable(final), final_md, summary


def ui_step_reset():
    return None, None, None, None, {}, {}, {}, {}, "", "", "", "State cleared."


# ---------------------------------------------------------------------------
# 13. Gradio callbacks — Trace / Prompts / Graph tabs
# ---------------------------------------------------------------------------

def ui_list_trace_files():
    files = sorted(TRACE_DIR.glob("*.json"), reverse=True)
    return gr.update(choices=[f.name for f in files], value=files[0].name if files else None)


def ui_load_trace_file(filename: str):
    if not filename:
        return {}
    path = TRACE_DIR / filename
    if not path.exists():
        raise gr.Error(f"Trace file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ui_apply_prompts(researcher_p: str, writer_p: str, fact_checker_p: str, editor_p: str):
    PROMPTS["researcher"] = researcher_p
    PROMPTS["writer"] = writer_p
    PROMPTS["fact_checker"] = fact_checker_p
    PROMPTS["editor"] = editor_p
    return "✅ Prompts applied — new runs (full pipeline and step-by-step) will use these."


def ui_reset_prompts():
    PROMPTS.update(DEFAULT_PROMPTS)
    return (
        DEFAULT_PROMPTS["researcher"],
        DEFAULT_PROMPTS["writer"],
        DEFAULT_PROMPTS["fact_checker"],
        DEFAULT_PROMPTS["editor"],
        "↩️ Prompts reset to defaults.",
    )


def ui_graph_mermaid():
    try:
        return research_graph.get_graph().draw_mermaid()
    except Exception as exc:  # noqa: BLE001
        return f"Could not render graph structure: {exc}"


# ---------------------------------------------------------------------------
# 14. Build the Gradio app
# ---------------------------------------------------------------------------

with gr.Blocks(title="Team Research Bot — Test Console") as demo:
    gr.Markdown("# 🧑‍🔬 Team Research Bot — Test Console")
    gr.Markdown(
        "Interactively test the Researcher → Writer → Fact-Checker → Editor LangGraph pipeline "
        "from `team_research_bot.ipynb`."
    )
    with gr.Accordion("⚙️ Environment / setup check", open=False):
        gr.Markdown(check_setup())

    documents_state = gr.State(None)
    messages_state = gr.State([])
    research_state = gr.State(None)
    draft_state = gr.State(None)
    fact_check_state = gr.State(None)
    final_state_step = gr.State(None)

    # ------------------------------------------------------------------ Sources
    with gr.Tab("📂 Sources"):
        gr.Markdown("Load and inspect the local source pack the Researcher agent reads from.")
        load_btn = gr.Button("Load / Reload Source Pack", variant="primary")
        sources_status = gr.Markdown()
        sources_df = gr.Dataframe(headers=["source_id", "title", "chars", "file"], interactive=False)
        with gr.Row():
            source_picker = gr.Dropdown(label="Preview source", choices=[])
        source_preview = gr.Markdown()

        load_btn.click(
            ui_load_sources,
            inputs=[],
            outputs=[documents_state, sources_df, source_picker, sources_status],
        )
        source_picker.change(ui_preview_source, inputs=[source_picker, documents_state], outputs=[source_preview])
        demo.load(ui_load_sources, inputs=[], outputs=[documents_state, sources_df, source_picker, sources_status])

    # ------------------------------------------------------------- Full pipeline
    with gr.Tab("▶️ Full Pipeline"):
        gr.Markdown("Runs the compiled LangGraph end-to-end, including the Fact-Checker → Writer revision loop.")
        with gr.Row():
            query_box = gr.Textbox(label="Research question", value=DEFAULT_QUERY, lines=2, scale=3)
            model_box = gr.Textbox(label="Model", value=OPENAI_MODEL, scale=1)
        with gr.Row():
            max_rev_slider = gr.Slider(0, 5, value=DEFAULT_MAX_REVISIONS, step=1, label="Max revisions")
            save_trace_checkbox = gr.Checkbox(label="Save trace to /traces", value=True)
        run_btn = gr.Button("Run Pipeline", variant="primary")

        pipeline_status = gr.Markdown()
        final_answer_md = gr.Markdown(label="Final answer")
        with gr.Row():
            takeaways_md = gr.Markdown(label="Key takeaways")
            caveats_md = gr.Markdown(label="Caveats")
            references_md = gr.Markdown(label="References")
        editor_notes_box = gr.Textbox(label="Editor notes", interactive=False)

        with gr.Accordion("Research Report (Researcher output)", open=False):
            research_json = gr.JSON()
        with gr.Accordion("Draft Report (final Writer output)", open=False):
            draft_json = gr.JSON()
        with gr.Accordion("Fact-Check Report (final)", open=False):
            fact_check_json = gr.JSON()
        with gr.Accordion("Final Report (raw)", open=False):
            final_json = gr.JSON()

        gr.Markdown("### Message trace (AgentMessage handoffs)")
        messages_df = gr.Dataframe(interactive=False, wrap=True)
        trace_saved_msg = gr.Markdown()

        run_btn.click(
            ui_run_pipeline,
            inputs=[query_box, max_rev_slider, model_box, save_trace_checkbox, documents_state],
            outputs=[
                pipeline_status,
                final_answer_md,
                takeaways_md,
                caveats_md,
                references_md,
                editor_notes_box,
                research_json,
                draft_json,
                fact_check_json,
                final_json,
                messages_df,
                trace_saved_msg,
                messages_state,
            ],
        )

    # ------------------------------------------------------------- Step-by-step
    with gr.Tab("🧪 Step-by-Step Agents"):
        gr.Markdown(
            "Call each agent individually and inspect the exact Pydantic payload it hands to the next agent. "
            "Useful for debugging one contract at a time."
        )
        with gr.Row():
            step_query_box = gr.Textbox(label="Research question", value=DEFAULT_QUERY, lines=2, scale=3)
            step_model_box = gr.Textbox(label="Model", value=OPENAI_MODEL, scale=1)
        reset_btn = gr.Button("Reset step-by-step state")
        step_status = gr.Markdown()

        gr.Markdown("#### 1) Researcher")
        step1_btn = gr.Button("Run Researcher")
        step1_summary = gr.Markdown()
        step1_json = gr.JSON(label="ResearchReport")

        gr.Markdown("#### 2) Writer — initial draft")
        step2_btn = gr.Button("Run Writer (initial draft)")
        step2_summary = gr.Markdown()
        step2_json = gr.JSON(label="DraftReport")

        gr.Markdown("#### 3) Fact-Checker")
        step3_btn = gr.Button("Run Fact-Checker")
        step3_summary = gr.Markdown()
        step3_json = gr.JSON(label="FactCheckReport")

        gr.Markdown("#### 4) Writer — revise (only needed if Fact-Checker flagged revision_required)")
        step4_btn = gr.Button("Run Writer (revise draft)")
        step4_summary = gr.Markdown()
        step4_json = gr.JSON(label="DraftReport (revised)")

        gr.Markdown("#### 5) Editor — finalize")
        step5_btn = gr.Button("Run Editor")
        step5_summary = gr.Markdown()
        step5_final_md = gr.Markdown()
        step5_json = gr.JSON(label="FinalReport")

        step1_btn.click(
            ui_step_researcher,
            inputs=[step_query_box, step_model_box, documents_state],
            outputs=[research_state, step1_json, step1_summary, documents_state],
        )
        step2_btn.click(
            ui_step_writer_initial,
            inputs=[step_query_box, step_model_box, research_state],
            outputs=[draft_state, step2_json, step2_summary],
        )
        step3_btn.click(
            ui_step_fact_checker,
            inputs=[draft_state, research_state, step_model_box],
            outputs=[fact_check_state, step3_json, step3_summary],
        )
        step4_btn.click(
            ui_step_writer_revise,
            inputs=[step_query_box, step_model_box, research_state, fact_check_state],
            outputs=[draft_state, step4_json, step4_summary],
        )
        step5_btn.click(
            ui_step_editor,
            inputs=[draft_state, fact_check_state, step_model_box],
            outputs=[final_state_step, step5_json, step5_final_md, step5_summary],
        )
        reset_btn.click(
            ui_step_reset,
            inputs=[],
            outputs=[
                research_state,
                draft_state,
                fact_check_state,
                final_state_step,
                step1_json,
                step2_json,
                step3_json,
                step5_json,
                step1_summary,
                step2_summary,
                step3_summary,
                step_status,
            ],
        )

    # ------------------------------------------------------------------- Trace
    with gr.Tab("📜 Trace Inspector"):
        gr.Markdown("Inspect AgentMessage handoffs from the last Full Pipeline run, or load a saved trace file.")
        gr.Markdown("#### Last run in this session")
        last_run_df = gr.Dataframe(interactive=False, wrap=True)
        refresh_last_run_btn = gr.Button("Refresh from last run")
        refresh_last_run_btn.click(
            lambda msgs: messages_to_dataframe(msgs), inputs=[messages_state], outputs=[last_run_df]
        )

        gr.Markdown("#### Saved trace files (`/traces`)")
        with gr.Row():
            trace_file_picker = gr.Dropdown(label="Trace file", choices=[])
            refresh_files_btn = gr.Button("Refresh list")
        trace_file_json = gr.JSON()

        refresh_files_btn.click(ui_list_trace_files, inputs=[], outputs=[trace_file_picker])
        trace_file_picker.change(ui_load_trace_file, inputs=[trace_file_picker], outputs=[trace_file_json])
        demo.load(ui_list_trace_files, inputs=[], outputs=[trace_file_picker])

    # ------------------------------------------------------------------- Graph
    with gr.Tab("🕸️ Graph Structure"):
        gr.Markdown("Mermaid definition of the compiled LangGraph `StateGraph` (nodes + fixed/conditional edges).")
        graph_code = gr.Code(value=ui_graph_mermaid(), language=None, label="mermaid")

    # ----------------------------------------------------------------- Prompts
    with gr.Tab("✍️ Prompts"):
        gr.Markdown("Edit each agent's system prompt, apply, then re-run the Full Pipeline or Step-by-Step tabs.")
        researcher_prompt_box = gr.Textbox(value=DEFAULT_PROMPTS["researcher"], label="Researcher system prompt", lines=8)
        writer_prompt_box = gr.Textbox(value=DEFAULT_PROMPTS["writer"], label="Writer system prompt", lines=8)
        fact_checker_prompt_box = gr.Textbox(
            value=DEFAULT_PROMPTS["fact_checker"], label="Fact-Checker system prompt", lines=8
        )
        editor_prompt_box = gr.Textbox(value=DEFAULT_PROMPTS["editor"], label="Editor system prompt", lines=8)
        with gr.Row():
            apply_prompts_btn = gr.Button("Apply prompts", variant="primary")
            reset_prompts_btn = gr.Button("Reset to defaults")
        prompts_status = gr.Markdown()

        apply_prompts_btn.click(
            ui_apply_prompts,
            inputs=[researcher_prompt_box, writer_prompt_box, fact_checker_prompt_box, editor_prompt_box],
            outputs=[prompts_status],
        )
        reset_prompts_btn.click(
            ui_reset_prompts,
            inputs=[],
            outputs=[researcher_prompt_box, writer_prompt_box, fact_checker_prompt_box, editor_prompt_box, prompts_status],
        )


if __name__ == "__main__":
    demo.queue().launch()
