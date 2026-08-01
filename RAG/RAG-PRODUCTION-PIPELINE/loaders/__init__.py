"""Document loaders for TXT, Markdown, CSV, HTML, PDF, and fallback splitting."""

from loaders.txt import process_txt, split_txt
from loaders.csv_loader import process_csv, split_csv
from loaders.fallback import split_text_from_docs

from loaders.html_loader import (
    discover_urls,
    process_html,
    split_html_char_chunks,
)

from loaders.markdown import process_markdown, split_markdown_smart

from loaders.pdf_loader import (
    describe_image_bytes,
    extract_pdf_pages,
    process_pdf,
    process_pdf_deferred,
)

from loaders.txt import process_txt, split_txt




__all__ = [
    "split_text_from_docs",
    "split_txt",
    "process_txt",
    "split_markdown_smart",
    "process_markdown",
    "split_csv",
    "process_csv",
    "split_html_char_chunks",
    "process_html",
    "discover_urls",
    "extract_pdf_pages",
    "describe_image_bytes",
    "process_pdf",
    "process_pdf_deferred",
]
