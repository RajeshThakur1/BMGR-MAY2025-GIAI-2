"""Mock hotel booking backend for the Northstar Hotels voice agent."""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from datetime import date

def normalize_reference(reference: str) -> str:
    """Normalize a reference string to a standard format."""
    ref = reference.strip().upper().replace(" ", "")
    if ref.startswith("BQ") and "-" not in ref and len(ref) > 2:
        ref = f"BQ-{ref[2:]}"
    return ref
 

@dataclass
class Booking:
    reference: str
    guest_name: str
    room_type: str
    check_in: date
    check_out: date


    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "guest_name": self.guest_name,
            "room_type": self.room_type,
            "check_in": self.check_in.isoformat(),
            "check_out": self.check_out.isoformat(),
        }


@dataclass
class HotelDatabase:
    bookings: dict[str, Booking] = field(default_factory=dict)

    def __post_init__(self):
        if not self.bookings:
            self.bookings={
                "BQ-7394": Booking(
                    reference="BQ-7394",
                    guest_name="John Doe",
                    room_type="Suite",
                    check_in=date(2026, 10, 10),
                    check_out=date(2026, 10, 15),
                ),
                "BQ-7395": Booking(
                    reference="BQ-7395",
                    guest_name="Jane Doe",
                    room_type="Deluxe",
                    check_in=date(2026, 10, 15),
                    check_out=date(2026, 10, 20),
                ),
            }

    def lookup(self, reference: str) -> Optional[Booking]:
        return self.bookings.get(normalize_reference(reference))

    def update_dates(
        self,
        reference: str,
        check_in: date,
        check_out: date,
        guest_name: Optional[str] = None,
        room_type: Optional[str] = None,
    ) -> Booking:
        ref = normalize_reference(reference)
        booking = self.bookings.get(ref)
        if booking is None:
            raise ValueError(f"Booking {reference} not found")
        if check_out <= check_in:
            raise ValueError("Check-out date must be after check-in date")
        booking.check_in = check_in
        booking.check_out = check_out
        if guest_name:
            booking.guest_name = guest_name
        if room_type:
            booking.room_type = room_type
        return booking

# Add the functions to cancel a booking and get all bookings


Hotel_DB = HotelDatabase()
