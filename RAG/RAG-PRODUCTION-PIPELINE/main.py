#!/usr/bin/env python3
"""Start the FastAPI server.

Usage:
    python main.py
    python main.py --host 0.0.0.0 --port 8000 --reload

Swagger UI: http://127.0.0.1:8000/docs

For CLI / function-style RAG commands, use main_rag.py instead.
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



def main() -> None:
    """Parse host/port and run uvicorn against api.app:app."""
    parser = argparse.ArgumentParser(description="Start Multi-Modal RAG FastAPI server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on code changes (dev only)",
    )
    args = parser.parse_args()

    import uvicorn

    print(f"Starting API at http://{args.host}:{args.port}")
    print(f"Swagger docs → http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
