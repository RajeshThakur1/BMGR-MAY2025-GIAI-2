"""Format the final travel plan text."""

from travel_planner_module.state import TravelPlannerState


# Node 9: Format Final Plan - Create beautiful, comprehensive travel plan

def format_final_plan(state: TravelPlannerState) -> TravelPlannerState:
    """Format all planning information into a comprehensive travel plan string."""
    print("📝 Formatting final travel plan...")

    plan_sections: list[str] = []

    if state["travel_request"]:
        tr = state["travel_request"]
        trip_type = (tr.trip_type or "leisure").title()
        plan_sections.append(
            f"""
🌍 YOUR PERSONALIZED TRAVEL PLAN 🌍
═══════════════════════════════════════

✈️ Trip Overview:
   📍 Destination: {tr.destination}
   🏠 Origin: {tr.origin}
   📅 Travel Dates: {tr.departure_date} to {tr.return_date}
   👥 Travelers: {tr.travelers}
   💰 Budget: ${(tr.budget or 0):.2f}
   🎭 Trip Type: {trip_type}
"""
        )

    if state["selected_flight"]:
        flight = state["selected_flight"]
        plan_sections.append(
            f"""
✈️ SELECTED FLIGHT:
   🛫 Airline: {flight.airline}
   🕐 Departure: {flight.departure_time}
   🕘 Arrival: {flight.arrival_time}
   ⏱️ Duration: {flight.duration}
   🛑 Stops: {flight.stops}
   💵 Price: ${flight.price} per person
"""
        )

    if state["selected_hotel"]:
        hotel = state["selected_hotel"]
        plan_sections.append(
            f"""
🏨 SELECTED ACCOMMODATION:
   🏩 Hotel: {hotel.name}
   ⭐ Rating: {hotel.rating}/5.0
   📍 Location: {hotel.location}
   🚶 Distance to Center: {hotel.distance_from_centre} km
   💵 Price: ${hotel.price_per_night} per night
   🎯 Amenities: {', '.join(hotel.amenities)}
"""
        )

    if state["weather_info"]:
        weather = state["weather_info"]
        plan_sections.append(
            f"""
🌤️ WEATHER FORECAST:
   🌡️ Temperature: {weather.temperature_range}
   ☁️ Conditions: {weather.conditions}
   🌧️ Rain Chance: {weather.precipitation_chance}%
   🎒 Packing Tips: {', '.join(weather.recommendations)}
"""
        )

    if state["itinerary"]:
        plan_sections.append("\n📅 DETAILED ITINERARY:")
        for day_plan in state["itinerary"]:
            plan_sections.append(
                f"""
   ═══ DAY {day_plan.day} - {day_plan.date} ═══
   💰 Daily Budget: ${day_plan.estimated_cost:.2f}
   📝 Notes: {day_plan.notes}

   Activities:"""
            )
            for activity in day_plan.activities:
                cost_str = f" (${activity['cost']:.2f})" if activity["cost"] > 0 else ""
                plan_sections.append(
                    f"""   • {activity['time']}: {activity['activity']}{cost_str}
     📖 {activity['description']}"""
                )

    if state["cost_breakdown"]:
        tr = state["travel_request"]
        if tr and tr.budget is not None:
            status = (
                "✅ Within Budget"
                if state["total_cost"] <= tr.budget
                else "⚠️ Over Budget"
            )
        else:
            status = "N/A"
        plan_sections.append(
            f"""
💰 BUDGET BREAKDOWN:
   🎯 Total Trip Cost: ${state['total_cost']:.2f}
   💳 Budget Status: {status}

   📊 Cost Details:"""
        )
        for category, cost in state["cost_breakdown"].items():
            pct = (cost / state["total_cost"]) * 100 if state["total_cost"] else 0
            plan_sections.append(f"   • {category}: ${cost:.2f} ({pct:.1f}%)")

    plan_sections.append(
        """
💡 TRAVEL TIPS:
   📱 Download offline maps and translation apps
   💳 Notify your bank of travel dates
   🎫 Book attraction tickets in advance when possible
   📋 Keep copies of important documents
   🌐 Check visa requirements and vaccination needs

🎉 Have an amazing trip! Safe travels! 🎉
═══════════════════════════════════════
"""
    )

    final_plan = "\n".join(plan_sections)

    print("✅ Final travel plan formatted successfully!")
    print(f"📄 Plan length: {len(final_plan)} characters")

    updated_state = dict(state)
    updated_state.update(
        {
            "final_plan": final_plan,
            "planning_stage": "end",
            "completed_steps": state["completed_steps"] + ["format_plan"],
        }
    )
    return TravelPlannerState(**updated_state)