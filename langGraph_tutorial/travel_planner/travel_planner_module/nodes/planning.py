"""Selection, itinerary, and budget calculation nodes."""

from datetime import datetime, timedelta

from travel_planner_module.models import DayPlan
from travel_planner_module.state import TravelPlannerState

# Node 6: Select Best Options - Choose flights and hotels
def select_best_options(state: TravelPlannerState) -> TravelPlannerState:
    """
    Select the best flight and hotel options based on criteria.
    """
    print("🎯 Selecting best options...")

    best_flight = None
    if state['flight_options']:
        scored_flights = []
        for flight in state['flight_options']:
            score = flight.price + (flight.stops * 50)  # Simple scoring: price + penalty for stops
            scored_flights.append((score, flight))

        # select flight with the best score
        best_flight = min(scored_flights, key=lambda x: x[0])[1]
        print(f"✅ Selected flight: {best_flight.airline} at ${best_flight.price} with {best_flight.stops} stops")

    best_hotel = None
    if state['hotel_options']:
        scored_hotels = []
        for hotel in state['hotel_options']:
            price_score = hotel.price_per_night  # Lower is better
            rating_score = (5.0 - hotel.rating) * 20  # Convert rating to penalty (lower is better)
            location_score = hotel.distance_from_centre * 10  # Distance penalty

            total_score = price_score + rating_score + location_score
            scored_hotels.append((total_score, hotel))
        
        # Select hotel with best score

        best_hotel = min(scored_hotels, key=lambda x: x[0])[1]
        print(f"✅ Selected hotel: {best_hotel.name} at ${best_hotel.price_per_night}/night with rating {best_hotel.rating}")

    updated_state = dict(state)
    updated_state.update({
        'selected_flight': best_flight,
        'selected_hotel': best_hotel,
        'planning_stage': "create_itinerary",
        'completed_steps': state['completed_steps'] + ["select_options"]
    })
    return TravelPlannerState(**updated_state)





# Node 7: Create Itinerary - Build detailed day-by-day plans

def create_itinerary(state: TravelPlannerState) -> TravelPlannerState:
    """
    Create a detailed day-by-day itinerary based on selected options and attractions.
    """

    print("📅 Creating detailed itinerary...")

    if not state['travel_request']:
        err_state = dict(state)
        err_state['errors'] = state['errors'] + ["No travel request found for itinerary creation"]
        return TravelPlannerState(**err_state)
    travel_req = state['travel_request']

    # Calculate trip duration
    from datetime import datetime, timedelta
    departure = datetime.strptime(travel_req.departure_date, "%Y-%m-%d")
    return_date = datetime.strptime(travel_req.return_date, "%Y-%m-%d")
    days = (return_date - departure).days

    # Distribute attractions across days
    attractions = state['attractions']
    itinerary = []


    # Simple distribution logic (in production, use more sophisticated algorithms)
    attractions_per_day = max(1, len(attractions) // max(1, days))
    
    for day in range(days):
        current_date = departure + timedelta(days=day)
        day_attractions = attractions[day * attractions_per_day:(day + 1) * attractions_per_day]

        activities = []
        daily_cost = 0.0
        if day == 0:  # Arrival day
            activities.append({
                "time": "Morning",
                "activity": "Arrival and hotel check-in",
                "description": f"Arrive via {state['selected_flight'].airline if state['selected_flight'] else 'flight'}, check into {state['selected_hotel'].name if state['selected_hotel'] else 'hotel'}",
                "cost": 0.0
            })

        # Add attraction activities

        for i, attraction in enumerate(day_attractions):
            time_slots = ["Morning", "Afternoon", "Evening"]
            time_slot = time_slots[min(i, 2)]
            
            activities.append({
                "time": time_slot,
                "activity": attraction.name,
                "description": f"{attraction.description} ({attraction.estimated_time})",
                "cost": attraction.cost
            })
            daily_cost += attraction.cost

        # Add meals
        if not any("food" in act["activity"].lower() for act in activities):
            activities.append({
                "time": "Evening",
                "activity": "Local Dining",
                "description": "Experience local cuisine at a recommended restaurant",
                "cost": 45.0
            })
            daily_cost += 45.0
        
        if day == days - 1:  # Departure day
            activities.append({
                "time": "Late Morning",
                "activity": "Check-out and departure",
                "description": "Hotel check-out and travel to airport",
                "cost": 0.0
            })

        # Create day plan
        day_plan = DayPlan(
            day=day + 1,
            date=current_date.strftime("%Y-%m-%d"),
            activities=activities,
            estimated_cost=daily_cost,
            notes=f"Weather: {state['weather_info'].conditions if state['weather_info'] else 'Check forecast'}"
        )

        itinerary.append(day_plan)

    print(f"✅ Created {len(itinerary)}-day itinerary:")
    for day_plan in itinerary:
        print(f"   📅 Day {day_plan.day} ({day_plan.date}): {len(day_plan.activities)} activities, ${day_plan.estimated_cost:.0f}")
    
    updated_state = dict(state)
    updated_state.update({
        'itinerary': itinerary,
        'planning_stage': "calculate_budget",
        'completed_steps': state['completed_steps'] + ["create_itinerary"]
    })
    return TravelPlannerState(**updated_state)


# Node 8: Calculate Budget - Provide detailed cost breakdown

def calculate_budget(state: TravelPlannerState) -> TravelPlannerState:
    """
    Calculate total trip cost and provide detailed breakdown.
    """
    print("💰 Calculating budget and costs...")
    
    cost_breakdown = {}
    total_cost = 0.0
    
    # Flight costs
    if state['selected_flight']:
        flight_cost = state['selected_flight'].price * state['travel_request'].travelers * 2  # Round trip
        cost_breakdown['Flights'] = flight_cost
        total_cost += flight_cost
    
    # Hotel costs
    if state['selected_hotel'] and state['travel_request']:
        from datetime import datetime
        departure = datetime.strptime(state['travel_request'].departure_date, "%Y-%m-%d")
        return_date = datetime.strptime(state['travel_request'].return_date, "%Y-%m-%d")
        nights = (return_date - departure).days
        
        hotel_cost = state['selected_hotel'].price_per_night * nights
        cost_breakdown['Accommodation'] = hotel_cost
        total_cost += hotel_cost
    
    # Activities and attractions costs
    activities_cost = sum(day.estimated_cost for day in state['itinerary'])
    if activities_cost > 0:
        cost_breakdown['Activities & Dining'] = activities_cost
        total_cost += activities_cost
    
    # Add miscellaneous costs (transportation, tips, shopping)
    misc_cost = total_cost * 0.15  # 15% for miscellaneous
    cost_breakdown['Transportation & Misc'] = misc_cost
    total_cost += misc_cost
    
    # Budget analysis
    if state['travel_request']:
        budget = state['travel_request'].budget
        budget_status = "✅ Within budget" if total_cost <= budget else "⚠️ Over budget"
        remaining = budget - total_cost
        
        print(f"✅ Budget Analysis:")
        print(f"   💰 Total Cost: ${total_cost:.2f}")
        print(f"   🎯 Budget: ${budget:.2f}")
        print(f"   📊 Status: {budget_status}")
        print(f"   💵 Remaining: ${remaining:.2f}")
        
        print(f"\n📋 Cost Breakdown:")
        for category, cost in cost_breakdown.items():
            percentage = (cost / total_cost) * 100
            print(f"   • {category}: ${cost:.2f} ({percentage:.1f}%)")
    
    updated_state = dict(state)
    updated_state.update({
        'total_cost': total_cost,
        'cost_breakdown': cost_breakdown,
        'planning_stage': "format_plan",
        'completed_steps': state['completed_steps'] + ["calculate_budget"]
    })
    return TravelPlannerState(**updated_state)