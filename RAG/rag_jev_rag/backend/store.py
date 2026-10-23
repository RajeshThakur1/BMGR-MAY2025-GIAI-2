"""SQLite metadata + Chroma for vectors (InsForge + pgvector in production)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from shared.config import settings


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Candidate:
    id: str
    document_id: str
    page_number: int
    modality: str
    content: str
    distance: float


class DataStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.data_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "support.db"
        self._init_sqlite()
        self.chroma = chromadb.PersistentClient(
            path=str(self.root / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.text_col = self.chroma.get_or_create_collection("text_chunks")
        self.image_col = self.chroma.get_or_create_collection("page_images")

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    company_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS indexing_jobs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    total_pages INTEGER NOT NULL,
                    completed_pages INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    company_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    meta_json TEXT
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    company_id TEXT NOT NULL,
                    conversation_id TEXT,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rejected_json TEXT,
                    created_at TEXT NOT NULL,
                    rep_reply TEXT,
                    resolved_at TEXT
                );
                CREATE TABLE IF NOT EXISTS decision_log (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    query TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def add_document(self, filename: str) -> str:
        doc_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?)",
                (doc_id, settings.company_id, filename, "indexing", _utcnow()),
            )
        return doc_id

    def update_document_status(self, doc_id: str, status: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE documents SET status=? WHERE id=?", (status, doc_id))

    def upsert_job(self, job_id: str, document_id: str, total: int, done: int, status: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO indexing_jobs VALUES (?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  completed_pages=excluded.completed_pages,
                  status=excluded.status,
                  updated_at=excluded.updated_at
                """,
                (job_id, document_id, total, done, status, _utcnow()),
            )

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        text_ids, text_docs, text_meta, text_emb = [], [], [], []
        img_ids, img_docs, img_meta, img_emb = [], [], [], []
        for c in chunks:
            meta = {
                "document_id": c["document_id"],
                "page_number": c["page_number"],
                "modality": c["modality"],
            }
            if c["modality"] == "text":
                text_ids.append(c["id"])
                text_docs.append(c["content"])
                text_meta.append(meta)
                text_emb.append(c["vector"])
            else:
                img_ids.append(c["id"])
                img_docs.append(c["content"])
                img_meta.append(meta)
                img_emb.append(c["vector"])
        if text_ids:
            self.text_col.upsert(ids=text_ids, documents=text_docs, metadatas=text_meta, embeddings=text_emb)
        if img_ids:
            self.image_col.upsert(ids=img_ids, documents=img_docs, metadatas=img_meta, embeddings=img_emb)

    def search(self, query_vector: list[float], *, n: int, modality: str) -> list[Candidate]:
        col = self.text_col if modality == "text" else self.image_col
        if col.count() == 0:
            return []
        res = col.query(query_embeddings=[query_vector], n_results=min(n, col.count()))
        out: list[Candidate] = []
        for i, cid in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            out.append(
                Candidate(
                    id=cid,
                    document_id=str(meta["document_id"]),
                    page_number=int(meta["page_number"]),
                    modality=str(meta["modality"]),
                    content=res["documents"][0][i] or "",
                    distance=float(res["distances"][0][i]),
                )
            )
        return out

    def create_conversation(self) -> str:
        cid = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO conversations VALUES (?,?,?)",
                (cid, settings.company_id, _utcnow()),
            )
        return cid

    def add_message(self, conversation_id: str, role: str, content: str, meta: dict | None = None) -> str:
        mid = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?)",
                (mid, conversation_id, role, content, _utcnow(), json.dumps(meta or {})),
            )
        return mid

    def log_decision(self, conversation_id: str, query: str, decision: str, details: dict) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO decision_log VALUES (?,?,?,?,?,?)",
                (str(uuid.uuid4()), conversation_id, query, decision, json.dumps(details), _utcnow()),
            )

    def open_ticket(self, conversation_id: str, question: str, rejected: list[dict]) -> str:
        tid = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    tid,
                    settings.company_id,
                    conversation_id,
                    question,
                    "open",
                    json.dumps(rejected),
                    _utcnow(),
                    None,
                    None,
                ),
            )
        return tid

    def list_open_tickets(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM tickets WHERE status='open' ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def resolve_ticket(self, ticket_id: str, reply: str, save_to_kb: bool) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE tickets SET status='resolved', rep_reply=?, resolved_at=? WHERE id=?",
                (reply, _utcnow(), ticket_id),
            )
            if save_to_kb:
                row = conn.execute(
                    "SELECT question FROM tickets WHERE id=?", (ticket_id,)
                ).fetchone()
                if row:
                    q = row[0]
                    chunk_id = f"kb-{uuid.uuid4()}"
                    content = f"Q: {q}\nA: {reply}"
                    from shared.embeddings import embed_texts

                    vec = embed_texts([content])[0]
                    self.text_col.upsert(
                        ids=[chunk_id],
                        documents=[content],
                        metadatas=[
                            {
                                "document_id": "human-kb",
                                "page_number": 0,
                                "modality": "text",
                            }
                        ],
                        embeddings=[vec],
                    )
