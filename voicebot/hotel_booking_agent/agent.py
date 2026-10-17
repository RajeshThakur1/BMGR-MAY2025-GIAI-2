"""LiveKit voice agent for Northstar Hotels booking changes."""

from __future__ import annotations

import logging
import os
from datetime import date
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, function_tool, inference, room_io
from livekit.plugins import openai, tavus

from hotel_tools import Hotel_DB, HotelDatabase, normalize_reference

from instructions import HOTEL_AGENT_INSTRUCTIONS, GREETING

from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger("hotel-booking-agent")
AGENT_NAME = os.getenv("AGENT_NAME", "hotel-booking-agent")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FISH_VOICE=os.getenv("FISH_AUDIO_VOICE", "fa4c9eb3dccc4806b382b40d61c6b10a")

@function_tool
async def lookup_booking(reference: str) -> dict:
    """Look up an existing reservations or booking by booking reference (  for Example BQ-- 7394)"""
    booking = Hotel_DB.lookup(reference)
    if booking is None:
        return {
            "found": False,
            "reference": normalize_reference(reference),
            "message": f"No booking found for this reference {reference}"
        }
    return {
        "found": True,
        **booking.to_dict(),
    }


@function_tool
async def update_booking_dates(
    reference: str,
    check_in: str,
    check_out: str,
    guest_name: str,
    room_type: str,
    ) -> dict:
    """Update the dates of an existing reservation or booking by booking reference (for Example BQ-- 7394)"""
    try:
        
        booking = Hotel_DB.update_dates(
            reference=reference,
            check_in=date.fromisoformat(check_in),
            check_out=date.fromisoformat(check_out),
            guest_name=guest_name,
            room_type=room_type,
        )

    except ValueError as e:
        return {"updated": False, "error": str(e)}

    return {
        "updated": True,
        "message": "Booking dates updated successfully",
        **booking.to_dict(),
    }


### Creating the Agent

class HotelBookingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=HOTEL_AGENT_INSTRUCTIONS,
            tools=[lookup_booking, update_booking_dates],
        )


server = AgentServer()


@server.rtc_session(agent_name = AGENT_NAME)
async def hotel_booking_agent(ctx: agents.JobContext) -> None:
    session = AgentSession(
        stt=inference.STT(
            model="speechmatics/linden-1",
            language="en",
            extra_kwargs={
                "enable_partials": True,
                "additional_vocab": [
                    {"content": "BQ-7394", "sounds_like": ["B Q seven three nine four"]},
                    {"content": "Northstar"},
                    {"content": "checkout"},
                ],
            }
        ),
        llm=openai.LLM(model=OPENAI_MODEL),
        tts=inference.TTS(
            model="fishaudio/s2.1-pro",
            voice=FISH_VOICE,
            ),

        turn_detection = "stt",
        use_tts_aligned_transcript=True,
    )

    if os.getenv("TAVUS_API_KEY"):
        avatar = tavus.AvatarSession(
            replica_id=os.getenv("TAVUS_REPLICA_ID"),
            persona_id=os.getenv("TAVUS_PERSONA_ID"),

        )
        await avatar.start(session, room=ctx.room)
    else:
        logger.info("TAVUS_API_KEY not set: running audio only")
        
    await session.start(
        room=ctx.room,
        agent=HotelBookingAgent(),
        room_options=room_io.RoomOptions(
            text_output=room_io.TextOutputOptions(
                sync_transcription=False
            ),
        ),
    )

    await session.generate_reply(instructions=f"Say Exactly: {GREETING}")


if __name__ == "__main__":
    agents.cli.run_app(server)



