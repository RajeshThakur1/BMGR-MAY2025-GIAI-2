"""Domain dataclasses for travel planning (no LangGraph dependencies)."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class TravelRequest:
    """User's Travel Request Details"""
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str] = None
    travelers: int = 1
    budget: Optional[float] = None
    preferences: Optional[Dict[str, Any]] = None
    trip_type: Optional[str] = None  # e.g., 'leisure', 'business', 'adventure', 'family'


@dataclass
class FlightOption:
    """Details of a flight option"""
    airline: str
    departure_time: str
    arrival_time: str
    duration: str
    price: float
    stops: int

@dataclass
class HotelOption:
    """Hotel search Result"""
    name: str
    rating: float
    price_per_night: float
    amenities: List[str]
    location: str
    distance_from_centre: float


@dataclass
class WeatherInfo:
    """Weather information"""
    temperature_range: str
    conditions: str
    precipitation_chance: int
    recommendations: List[str]


@dataclass
class Attraction:
    """Tourist attraction information"""
    name: str
    category: str
    rating: float
    estimated_time: str
    cost: float
    description: str

@dataclass
class DayPlan:
    """Daily itinerary"""
    day: int
    date: str
    activities: List[Dict[str, Any]]
    estimated_cost: float
    notes: str




