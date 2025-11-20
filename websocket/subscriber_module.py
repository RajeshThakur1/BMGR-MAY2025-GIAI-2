from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from connections import subscribers
import logging

logger = logging.getLogger("backend")
router = APIRouter()

@router.websocket("/ws/subscribe/{channel}/{username}")
async def websocket_subscribe(websocket: WebSocket, channel: str, username: str):
    await websocket.accept()
    subscribers[channel][username] = websocket
    logger.info(f"{username} subscribed to channel {channel}")

    # No history is sent; this is a pure in-memory live subscription

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        subscribers[channel].pop(username, None)
        logger.info(f"{username} unsubscribed from channel {channel}")
