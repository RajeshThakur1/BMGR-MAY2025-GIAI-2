#!/usr/bin/env python3
"""Gradio UI that calls the FastAPI RAG services for end-to-end testing.

Prerequisites:
    1. Start the API:  python main.py
    2. Start this UI:  python gradio_app.py

Default API: http://127.0.0.1:8001
"""


import json
from pathlib import Path
from typing import Any

import gradio as gr
import requests

DEFAULT_API = "http://127.0.0.1:8001"
TIMEOUT = 300  # indexing / LLM can be slow

def _base(api_url: str) -> str:
    """Normalize API base URL (strip trailing slash)."""
    return (api_url or DEFAULT_API).rstrip("/")


def _fmt(data: Any) -> str:
    """Pretty-print JSON-like responses for the UI."""
    if isinstance(data, (dict, list)):
        return json.dumps(data, indent=2, ensure_ascii=False)
    return str(data)


def _request(
    method: str,
    url: str,
    *,
    json_body: dict | None = None,
    files: dict | None = None,
    data: dict | None = None,
    timeout: int = TIMEOUT,
) -> str:
    """Call the FastAPI backend and return a readable status + body string."""
    try:
        resp = requests.request(
            method,
            url,
            json=json_body,
            files=files,
            data=data,
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError:
        return (
            "ERROR: Cannot reach the API.\n"
            f"Is the server running? Try: python main.py\n"
            f"Expected URL: {url}"
        )
    except requests.exceptions.Timeout:
        return f"ERROR: Request timed out after {timeout}s → {url}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

    try:
        body = resp.json()
        body_text = _fmt(body)
    except Exception:
        body_text = resp.text

    return f"HTTP {resp.status_code}\n\n{body_text}"

def do_health(api_url: str) -> str:
    """GET /health and GET /collection."""
    base = _base(api_url)
    health = _request("GET", f"{base}/health", timeout=30)
    collection = _request("GET", f"{base}/collection", timeout=30)
    return f"=== /health ===\n{health}\n\n=== /collection ===\n{collection}"

def do_index_upload(api_url: str, file_obj, use_llm: bool) -> str:
    """POST /index/upload with a Gradio file."""
    if file_obj is None:
        return "Please upload a file (.txt, .md, .csv, .html, .pdf)."
    path = Path(file_obj if isinstance(file_obj, str) else file_obj.name)
    with path.open("rb") as f:
        return _request(
            "POST",
            f"{_base(api_url)}/index/upload",
            files={"file": (path.name, f)},
            data={"use_llm": str(use_llm).lower()},
        )


def do_list_sources(api_url: str) -> gr.CheckboxGroup:
    """GET /sources → refresh document multiselect choices."""
    try:
        resp = requests.get(f"{_base(api_url)}/sources", timeout=60)
        resp.raise_for_status()
        names = resp.json().get("file_names", [])
    except Exception as e:
        return gr.CheckboxGroup(
            choices=[],
            value=[],
            label=f"Documents (failed to load: {e})",
        )
    return gr.CheckboxGroup(
        choices=names,
        value=[],
        label=f"Restrict to documents ({len(names)} available) — leave empty = all",
    )


def do_ask(
    api_url: str,
    question: str,
    limit: int,
    file_names: list[str] | None,
) -> tuple[str, str]:
    """POST /ask → answer text + sources JSON."""
    if not question.strip():
        return "Enter a question.", ""
    selected = [n for n in (file_names or []) if n]
    body: dict[str, Any] = {
        "question": question.strip(),
        "limit": int(limit),
        "show_sources": True,
    }
    if selected:
        body["file_names"] = selected
    raw = _request(
        "POST",
        f"{_base(api_url)}/ask",
        json_body=body,
    )
    try:
        json_part = raw.split("\n\n", 1)[1]
        data = json.loads(json_part)
        answer = data.get("answer", "")
        meta = {
            "file_names_filter": data.get("file_names", []),
            "sources": data.get("sources", []),
        }
        return answer, _fmt(meta)
    except Exception:
        return raw, ""


def do_search(
    api_url: str,
    query: str,
    limit: int,
    file_names: list[str] | None,
) -> str:
    """POST /search (retrieve only)."""
    if not query.strip():
        return "Enter a search query."
    selected = [n for n in (file_names or []) if n]
    body: dict[str, Any] = {"query": query.strip(), "limit": int(limit)}
    if selected:
        body["file_names"] = selected
    return _request(
        "POST",
        f"{_base(api_url)}/search",
        json_body=body,
        timeout=60,
    )

def do_crawl(
    api_url: str,
    seed: str,
    max_pages: int,
    max_depth: int,
    index: bool,
    use_llm: bool,
) -> str:
    """POST /crawl."""
    if not seed.strip():
        return "Enter a seed URL (e.g. https://example.com)."
    return _request(
        "POST",
        f"{_base(api_url)}/crawl",
        json_body={
            "seed": seed.strip(),
            "max_pages": int(max_pages),
            "max_depth": int(max_depth),
            "index": index,
            "use_llm": use_llm,
        },
    )



def do_delete(api_url: str, file_name: str) -> str:
    """DELETE /sources/{file_name}."""
    if not file_name.strip():
        return "Enter a file name (e.g. catalog.csv or a crawled URL)."
    from urllib.parse import quote

    name = quote(file_name.strip(), safe="")
    return _request("DELETE", f"{_base(api_url)}/sources/{name}", timeout=60)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Build the Gradio Blocks layout wired to FastAPI endpoints."""
    with gr.Blocks(title="RAG API Tester") as demo:
        gr.Markdown(
            """
# Multi-Modal RAG — API Tester

Talks to your FastAPI server (`python main.py`).  
Use the tabs below to walk the full flow: **prepare → index → ask**.
"""
        )

        api_url = gr.Textbox(
            label="API base URL",
            value=DEFAULT_API,
            info="Must match the running FastAPI server",
        )

        with gr.Tab("1. Health"):
            gr.Markdown("Check that the API and Qdrant are reachable.")
            health_btn = gr.Button("Check health + collection", variant="primary")
            health_out = gr.Code(label="Response", language="json")
            health_btn.click(do_health, inputs=[api_url], outputs=[health_out])

        with gr.Tab("2. Index"):
            # gr.Markdown("Index sample files **or** upload your own document into Qdrant.")
            # with gr.Row():
            #     use_llm = gr.Checkbox(label="Use LLM extraction", value=True)
            #     pdf_images = gr.Checkbox(label="Process PDF images", value=False)
            # index_btn = gr.Button("Index sample_data/", variant="primary")
            # index_out = gr.Code(label="Index samples response", language="json")
            # index_btn.click(
            #     do_index_upload,
            #     inputs=[api_url, use_llm, pdf_images],
            #     outputs=[index_out],
            # )

            gr.Markdown("---")
            upload = gr.File(
                label="Upload file (.txt .md .csv .html .pdf)",
                file_types=[".txt", ".md", ".csv", ".html", ".htm", ".pdf"],
            )
            upload_llm = gr.Checkbox(label="Use LLM for upload", value=True)
            upload_btn = gr.Button("Upload & index")
            upload_out = gr.Code(label="Upload response", language="json")
            upload_btn.click(
                do_index_upload,
                inputs=[api_url, upload, upload_llm],
                outputs=[upload_out],
            )

        with gr.Tab("3. Ask (RAG)"):
            gr.Markdown(
                "Full RAG: retrieve from Qdrant → LiteLLM grounded answer.\n\n"
                "Optionally pick specific documents; leave empty to search **all**."
            )
            refresh_docs_btn = gr.Button("Refresh document list")
            doc_picker = gr.CheckboxGroup(
                choices=[],
                value=[],
                label="Restrict to documents (leave empty = all)",
            )
            refresh_docs_btn.click(
                do_list_sources, inputs=[api_url], outputs=[doc_picker]
            )
            question = gr.Textbox(
                label="Question",
                placeholder="What is the refund policy?",
                lines=2,
            )
            ask_limit = gr.Slider(1, 10, value=4, step=1, label="Top-k chunks")
            ask_btn = gr.Button("Ask", variant="primary")
            answer_out = gr.Markdown(label="Answer")
            sources_out = gr.Code(label="Sources + filter applied", language="json")
            ask_btn.click(
                do_ask,
                inputs=[api_url, question, ask_limit, doc_picker],
                outputs=[answer_out, sources_out],
            )
            question.submit(
                do_ask,
                inputs=[api_url, question, ask_limit, doc_picker],
                outputs=[answer_out, sources_out],
            )

        with gr.Tab("4. Search only"):
            gr.Markdown(
                "Retrieve chunks only (no LLM answer). "
                "Same document filter as Ask."
            )
            refresh_search_docs_btn = gr.Button("Refresh document list")
            search_doc_picker = gr.CheckboxGroup(
                choices=[],
                value=[],
                label="Restrict to documents (leave empty = all)",
            )
            refresh_search_docs_btn.click(
                do_list_sources, inputs=[api_url], outputs=[search_doc_picker]
            )
            search_q = gr.Textbox(label="Query", placeholder="refund policy")
            search_limit = gr.Slider(1, 10, value=4, step=1, label="Top-k")
            search_btn = gr.Button("Search", variant="primary")
            search_out = gr.Code(label="Hits", language="json")
            search_btn.click(
                do_search,
                inputs=[api_url, search_q, search_limit, search_doc_picker],
                outputs=[search_out],
            )

        with gr.Tab("5. Crawl"):
            gr.Markdown(
                "Discover URLs from a seed. Set **Index into Qdrant** to write vectors "
                "(sites like Medium may block crawlers — try https://example.com first)."
            )
            seed = gr.Textbox(label="Seed URL", value="https://example.com")
            with gr.Row():
                max_pages = gr.Slider(1, 20, value=5, step=1, label="Max pages")
                max_depth = gr.Slider(0, 3, value=1, step=1, label="Max depth")
            with gr.Row():
                crawl_index = gr.Checkbox(label="Index into Qdrant", value=True)
                crawl_llm = gr.Checkbox(label="Use LLM", value=True)
            crawl_btn = gr.Button("Crawl", variant="primary")
            crawl_out = gr.Code(label="Response", language="json")
            crawl_btn.click(
                do_crawl,
                inputs=[api_url, seed, max_pages, max_depth, crawl_index, crawl_llm],
                outputs=[crawl_out],
            )

        with gr.Tab("6. Delete source"):
            gr.Markdown("Remove all Qdrant points for a given `fileName`.")
            del_name = gr.Textbox(
                label="fileName",
                placeholder="product_faq.txt",
            )
            del_btn = gr.Button("Delete", variant="stop")
            del_out = gr.Code(label="Response", language="json")
            del_btn.click(do_delete, inputs=[api_url, del_name], outputs=[del_out])

    gr.Markdown(
        """
---
**Suggested test flow:** Health → Prepare samples → Index → Ask  
API docs: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)
"""
    )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="localhost", server_port=7860, share=True)

