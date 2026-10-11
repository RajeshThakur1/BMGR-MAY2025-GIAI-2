"""LiveKit access token and agent dispatch helpers."""

from __future__ import annotations

from operator import attrgetter
import os
from re import A
import uuid
from dataclasses import dataclass
from livekit import api


@dataclass
class SessionCredentials:
    url: str
    token: str
    room_name: str
    identity: str


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required enviorent variable: {name}")
    return value


def _build_token(
    *,
    url: str,
    api_key: str,
    api_secret: str,
    room: str,
    participant: str,
):
    return (
        api.AccessToken(api_key, api_secret)
        .with_identity(participant)
        .with_name("Hotel Caller")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )


async def create_session_credentials(
    *,
    room_name: str | None = None,
    identity: str | None = None,
    agent_name: str | None = None,
    ) -> SessionCredentials:


    """Create LiveKit session credentials for a given room and identity."""
    url = _require_env("LIVEKIT_URL")
    api_key = _require_env("LIVEKIT_API_KEY")
    api_secret = _require_env("LIVEKIT_API_SECRET")

    room = room_name or f"hotel-{uuid.uuid4().hex[:8]}"
    participant = identity or f"caller-{uuid.uuid4().hex[:8]}"

    token = _build_token(
        url=url,
        api_key=api_key,
        api_secret=api_secret,
        room=room,
        participant=participant,
    )

    if agent_name:
        await _dispatch_agent(url, api_key, api_secret, agent_name, room)

    return SessionCredentials(url=url, token=token, room_name=room, identity=participant)


async def _dispatch_agent(
    url: str,
    api_key: str,
    api_secret: str,
    agent_name: str,
    room: str
) -> None:
    dispatch_client = api.LiveKitAPI(url, api_key, api_secret)
    try:
        await dispatch_client.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=agent_name,
                room=room,
            )
        )
    finally:
        await dispatch_client.aclose()
