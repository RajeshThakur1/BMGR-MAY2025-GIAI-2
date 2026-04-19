"""Mock flight and hotel search nodes."""

from datetime import datetime, timedelta

from travel_planner_module.models import FlightOption, HotelOption
from travel_planner_module.state import TravelPlannerState


def search_flights(state: TravelPlannerState) -> TravelPlannerState:
    """
    Search for available flights based on travel request.
    
    This node:
    1. Uses travel request details to search flights
    2. Filters results based on budget and preferences
    3. Returns a list of flight options
    """
    # Placeholder implementation - replace with real API calls
    print("Searching for flights based on the travel request...")

    if not state["travel_request"]:
        state["errors"].append("No travel request found. Cannot search for flights.")
        return state
    travel_req = state["travel_request"]

    # Mock flight search (in production, integrate with flight APIs)
    mock_flights = [
        FlightOption(
            airline="Air France",
            departure_time="08:00",
            arrival_time="21:30",
            duration="7h 30m",
            price=650.0,
            stops=0
        ),
        FlightOption(
            airline="Delta",
            departure_time="14:15",
            arrival_time="04:45+1",
            duration="8h 30m",
            price=580.0,
            stops=1
        ),
        FlightOption(
            airline="British Airways",
            departure_time="22:00",
            arrival_time="11:15+1",
            duration="7h 15m",
            price=720.0,
            stops=0
        )
    ]

    # Filter by budget (flight costs for round trip)
    budget_per_person = travel_req.budget / 2  # Rough estimate
    suitable_flights = [f for f in mock_flights if f.price <= budget_per_person]

    print(f"Found {len(suitable_flights)} flight options within budget.")

    for flight in suitable_flights:
        print(f"   ✈️ {flight.airline}: ${flight.price}, {flight.duration}, {flight.stops} stops")


    updated_state = dict(state)
    updated_state.update({
        'flight_options': suitable_flights,
        'planning_stage': "hotel_search",
        'completed_steps': state['completed_steps'] + ["flight_search"]
    })

    return TravelPlannerState(**updated_state)




def search_hotels(state: TravelPlannerState) -> TravelPlannerState:
    """
    Search for hotels based on travel request and selected flight.
    
    This node:
    1. Uses travel request details to search hotels
    2. Filters results based on budget and preferences
    3. Returns a list of hotel options
    """
    # Placeholder implementation - replace with real API calls
    print("Searching for hotels based on the travel request...")

    if not state["travel_request"]:
        state["errors"].append("No travel request found. Cannot search for hotels.")
        return state
    travel_req = state["travel_request"]

    # Mock hotel search (in production, integrate with hotel APIs)
    mock_hotels = [
        HotelOption(
            name="Tokyo Grand Hotel",
            rating=4.5,
            price_per_night=150.0,
            amenities=["Free WiFi", "Breakfast included", "Pool"],
            location="Shinjuku",
            distance_from_centre=2.0
        ),
        HotelOption(
            name="Sakura Inn",
            rating=4.0,
            price_per_night=100.0,
            amenities=["Free WiFi", "Pet-friendly"],
            location="Asakusa",
            distance_from_centre=3.5
        ),
        HotelOption(
            name="Budget Stay Tokyo",
            rating=3.5,
            price_per_night=80.0,
            amenities=["Free WiFi"],
            location="Ueno",
            distance_from_centre=4.0
        )
    ]

     # Calculate trip duration for budget calculation
    departure = datetime.strptime(travel_req.departure_date, "%Y-%m-%d")
    return_date = datetime.strptime(travel_req.return_date, "%Y-%m-%d")
    nights = (return_date - departure).days

    # Filter hotels by budget
    remaining_budget = travel_req.budget - (len(state['flight_options']) * state['flight_options'][0].price * 2 if state['flight_options'] else 0)
    budget_per_night = (remaining_budget * 0.6) / nights  # Allocate 60% of remaining budget to hotels
    suitable_hotels = [h for h in mock_hotels if h.price_per_night <= budget_per_night]
    print(f"Found {len(suitable_hotels)} hotel options within budget.")

    for hotel in suitable_hotels:
        total_cost = hotel.price_per_night * nights
        print(f"   • {hotel.name}: ${hotel.price_per_night}/night (${total_cost:.0f} total, ⭐{hotel.rating})")

    updated_state = dict(state)
    updated_state.update({
        'hotel_options': suitable_hotels,
        'planning_stage': "weather_check",
        'completed_steps': state['completed_steps'] + ["hotel_search"]
    })
    return TravelPlannerState(**updated_state)







    