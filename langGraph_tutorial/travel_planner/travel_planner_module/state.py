"""LangGraph shared state schema and initial state factory."""

from typing import Dict, List, Optional, TypedDict

from travel_planner_module.models import Attraction, FlightOption, HotelOption, TravelRequest, WeatherInfo, DayPlan      

class TravelPlannerState(TypedDict):
    """
    Complete state for the travel planner agent.
    
    This state flows through all nodes and accumulates information
    throughout the planning process.
    """

    user_request: str
    travel_request: Optional[TravelRequest]
    flight_options: List[FlightOption]
    hotel_options: List[HotelOption]
    weather_info: Optional[WeatherInfo]
    attractions: List[Attraction]

    ### selected options and final plan
    selected_flight: Optional[FlightOption]
    selected_hotel: Optional[HotelOption]


    ## planning results
    itinerary: List[DayPlan]
    total_cost: float
    cost_breakdown: Dict[str, float]

    ### process control
    planning_stage: str
    errors: List[str]
    completed_steps: List[str]
    final_plan: str



# Initlize our state with default values


# State(product_name=product_name, basic_description="", features_benefits="", marketing_message="", final_description="")
def create_initial_state(user_input: str) -> TravelPlannerState:
    return TravelPlannerState(
        user_request=user_input,
        travel_request=None,
        flight_options=[],
        hotel_options=[],
        weather_info=None,
        attractions=[],
        selected_flight=None,
        selected_hotel=None,
        itinerary=[],
        total_cost=0.0,
        cost_breakdown={},
        planning_stage="initial",
        errors=[],
        completed_steps=[],
        final_plan=""
    )


