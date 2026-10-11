"""Top-level web UI for microphone access and LiveKit session creation."""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import json
from pathlib import Path
from token_utils import create_session_credentials


load_dotenv()

AGENT_NAME= os.getenv("AGENT_NAME", "hotel-booking-agent")
FRONTEND_PATH = Path(__file__).parent / "frontend" / "voice_client.html"

app = FastAPI(title="Hotel booking voice agent")


class SessionResponse(BaseModel):
    url: str
    token: str
    room: str

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    config = {
        "url": os.getenv("LIVEKIT_URL"),
        "token": "",
        "room": "",
        "autoConnect": False,
        "tokenEndpoint": "api/session"
    }
    html = FRONTEND_PATH.read_text(encoding="utf-8").replace("__CONFIG__", json.dumps(config))
    return HTMLResponse(html)


@app.post("/api/session", response_model=SessionResponse)
async def create_session() -> SessionResponse:
    try:
        creds = await create_session_credentials(agent_name=AGENT_NAME)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SessionResponse(url=creds.url, token=creds.token, room=creds.room_name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)