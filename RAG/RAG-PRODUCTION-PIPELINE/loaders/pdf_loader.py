"""PDF loader with optional vision captions and deferred image processing."""

from __future__ import annotations

import base64
from pathlib import Path

import litellm
import pymupdf as fitz

from config import VISION_MODEL
from llm import build_page_payload, extract_content_chunks, extract_metadata


def extract_pdf_pages(path: Path) -> list[dict]:
    """Extract per-page text and embedded images from a PDF.

    Args:
        path: Path to a PDF file.

    Returns:
        List of `{"page_number", "text", "images"}` dicts. `images` is a list
        of PNG `bytes` (may be empty).
    """
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        images = []
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                images.append(pix.tobytes("png"))
            except Exception:
                continue
        pages.append({"page_number": i, "text": text, "images": images})
    doc.close()
    return pages


def describe_image_bytes(png_bytes: bytes) -> str:
    """Describe a PDF image with a vision LLM for retrieval text.

    Args:
        png_bytes: Raw PNG image bytes.

    Returns:
        Short natural-language description string.
    """
    b64 = base64.b64encode(png_bytes).decode("ascii")
    resp = litellm.completion(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in under 20 words for search retrieval.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=40,
    )
    return resp.choices[0].message.content.strip()


def process_pdf(
    path: Path,
    use_llm: bool = True,
    process_images: bool = False,
) -> list[dict]:
    """Index a PDF one page at a time (optional image captions).

    Args:
        path: PDF file path.
        use_llm: Call LiteLLM for metadata/chunks/captions when True.
        process_images: If True, describe embedded images and set `imgs`.

    Returns:
        One indexable document dict per PDF page.
    """
    pages = extract_pdf_pages(path)
    docs = []
    for page in pages:
        text = page["text"] or f"(empty page {page['page_number']})"
        if use_llm:
            meta = extract_metadata(text, page_number=page["page_number"])
            content = extract_content_chunks(text)
        else:
            meta = {
                "page_number": page["page_number"],
                "page_title": (
                    text.splitlines()[0][:80]
                    if text
                    else f"Page {page['page_number']}"
                ),
                "summary": text[:120],
                "keywords": [],
            }
            content = [text]

        images_line = None
        if process_images and page["images"]:
            descs = []
            for idx, png in enumerate(page["images"], start=1):
                desc = describe_image_bytes(png) if use_llm else f"image-{idx}"
                descs.append(f"local://page{page['page_number']}/img{idx} - {desc}")
            images_line = f"Imgs({len(descs)}): " + " | ".join(descs)

        docs.append(
            {
                "page_content": build_page_payload(meta, content, images_line=images_line),
                "metadata": {"source": str(path), "page_number": page["page_number"]},
            }
        )
    return docs


def process_pdf_deferred(path: Path, use_llm: bool = True):
    """Split PDF indexing into immediate (text-only) and deferred (image) pages.

    Args:
        path: PDF file path.
        use_llm: Whether metadata/content/captions use LiteLLM.

    Returns:
        Tuple `(immediate_docs, deferred_docs)`.
    """
    pages = extract_pdf_pages(path)
    immediate, deferred = [], []
    for page in pages:
        if page["images"]:
            deferred.append(page)
        else:
            immediate.append(page)

    def pages_to_docs(page_list, process_images: bool = False) -> list[dict]:
        """Convert raw page dicts into indexable payloads."""
        docs = []
        for page in page_list:
            text = page["text"] or f"(empty page {page['page_number']})"
            meta = (
                extract_metadata(text, page["page_number"])
                if use_llm
                else {
                    "page_number": page["page_number"],
                    "page_title": f"Page {page['page_number']}",
                    "summary": text[:120],
                    "keywords": [],
                }
            )
            content = extract_content_chunks(text) if use_llm else [text]
            images_line = None
            if process_images and page["images"]:
                descs = [
                    f"local://p{page['page_number']}/i{i} - "
                    f"{(describe_image_bytes(png) if use_llm else 'img')}"
                    for i, png in enumerate(page["images"], start=1)
                ]
                images_line = f"Imgs({len(descs)}): " + " | ".join(descs)
            docs.append(
                {
                    "page_content": build_page_payload(meta, content, images_line),
                    "metadata": {
                        "source": str(path),
                        "page_number": page["page_number"],
                    },
                }
            )
        return docs

    return pages_to_docs(immediate, False), pages_to_docs(deferred, True)