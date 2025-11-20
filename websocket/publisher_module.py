from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from connections import subscribers
from datetime import datetime
import logging

logger = logging.getLogger("backend")
router = APIRouter()

@router.websocket("/ws/publish/{channel}/{sender}")
async def websocket_publish(websocket: WebSocket, channel: str, sender: str):
    await websocket.accept()
    logger.info(f"Publisher {sender} connected to channel {channel}")

    try:
        while True:
            content = await websocket.receive_text()

            # Deliver live to all subscribers in this channel, in-memory only
            msg_timestamp = datetime.utcnow()
            if channel in subscribers:
                dead_users = []
                for username, ws in subscribers[channel].items():
                    try:
                        await ws.send_text(f"{content}")
                        logger.info(f"Delivered {sender} -> {username} in channel {channel}: {content}")
                    except WebSocketDisconnect:
                        dead_users.append(username)

                # Remove dead connections
                for u in dead_users:
                    subscribers[channel].pop(u, None)
    except WebSocketDisconnect:
        logger.info(f"Publisher {sender} disconnected from channel {channel}")
    finally:
        pass
