#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p data

echo "Starting model server on :8001 …"
uvicorn model_server.main:app --host 127.0.0.1 --port 8001 &
MS_PID=$!

cleanup() {
  kill "$MS_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 2
echo "Starting support desk API on :8000 …"
echo "Open http://127.0.0.1:8000"
uvicorn backend.main:app --host 127.0.0.1 --port 8000
