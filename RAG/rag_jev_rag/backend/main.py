"""Support desk API — local stand-in for InsForge edge functions + Postgres."""

from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.indexing import index_pdf_bytes
from backend.pipeline import run_query_pipeline
from backend.store import DataStore
from shared.config import settings

app = FastAPI(title="Jev Agentic RAG Support Desk", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = DataStore()
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1)
    screenshot_base64: str | None = None


class TicketReplyRequest(BaseModel):
    reply: str
    save_to_knowledge: bool = True


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "company_id": settings.company_id,
        "decision_provider": settings.decision_provider,
        "model_server": settings.model_server_url,
    }


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF uploads are supported in this demo.")
    raw = await file.read()
    doc_id = store.add_document(file.filename)
    docs_dir = settings.data_dir / "uploads"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / f"{doc_id}.pdf").write_bytes(raw)
    stats = index_pdf_bytes(store, doc_id, raw)
    return {"document_id": doc_id, "filename": file.filename, **stats}


@app.post("/api/chat")
def chat(body: ChatRequest) -> dict:
    cid = body.conversation_id or store.create_conversation()
    store.add_message(cid, "user", body.message)
    result = run_query_pipeline(
        store,
        conversation_id=cid,
        question=body.message,
        screenshot_b64=body.screenshot_base64,
    )
    return {"conversation_id": cid, **result}


@app.get("/api/tickets")
def tickets() -> dict:
    return {"tickets": store.list_open_tickets()}


@app.post("/api/tickets/{ticket_id}/reply")
def reply_ticket(ticket_id: str, body: TicketReplyRequest) -> dict:
    store.resolve_ticket(ticket_id, body.reply, body.save_to_knowledge)
    return {"status": "resolved", "ticket_id": ticket_id}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
