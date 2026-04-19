"""Weather and attractions discovery nodes (mock data)."""


import random

from travel_planner_module.models import Attraction, WeatherInfo
from travel_planner_module.state import TravelPlannerState


def check_weather(state: TravelPlannerState) -> TravelPlannerState:
    """
    Check weather conditions for the destination and dates.
    """
    print("🌤️ Checking weather conditions...")
    
    if not state['travel_request']:
        err_state = dict(state)
        err_state['errors'] = state['errors'] + ["No travel request found for weather check"]
        return TravelPlannerState(**err_state)
    
    # Mock weather data (in production, integrate with weather APIs)
    weather_conditions = [
        ("Sunny", "20-25°C", 10, ["Light clothing", "Sunglasses", "Sunscreen"]),
        ("Partly Cloudy", "18-23°C", 20, ["Layers", "Light jacket", "Comfortable shoes"]),
        ("Rainy", "15-20°C", 70, ["Umbrella", "Waterproof jacket", "Boots"])
    ]
    
    # Randomly select weather for demo
    condition, temp_range, precip, recommendations = random.choice(weather_conditions)
    
    weather_info = WeatherInfo(
        temperature_range=temp_range,
        conditions=condition,
        precipitation_chance=precip,
        recommendations=recommendations
    )
    
    print(f"✅ Weather forecast:")
    print(f"   🌡️ Temperature: {weather_info.temperature_range}")
    print(f"   ☁️ Conditions: {weather_info.conditions}")
    print(f"   🌧️ Rain chance: {weather_info.precipitation_chance}%")
    print(f"   🎒 Pack: {', '.join(weather_info.recommendations)}")
    
    # Create updated state
    updated_state = dict(state)
    updated_state.update({
        'weather_info': weather_info,
        'planning_stage': "attractions_search",
        'completed_steps': state['completed_steps'] + ["weather_check"]
    })
    return TravelPlannerState(**updated_state)




def find_attractions(state: TravelPlannerState) -> TravelPlannerState:
    """
    Find attractions and activities based on user preferences.
    """

    print("🎭 Finding attractions and activities...")
    if not state['travel_request']:
        err_state = dict(state)
        err_state['errors'] = state['errors'] + ["No travel request found for attractions search"]
        return TravelPlannerState(**err_state)
    travel_req = state['travel_request']


    # Mock attractions database (in production, integrate with tourism APIs)
    all_attractions = [
        Attraction("Louvre Museum", "museum", 4.6, "3-4 hours", 17.0, "World's largest art museum"),
        Attraction("Eiffel Tower", "landmark", 4.5, "2-3 hours", 25.0, "Iconic iron tower with city views"),
        Attraction("Notre-Dame Cathedral", "historic", 4.4, "1-2 hours", 0.0, "Gothic masterpiece cathedral"),
        Attraction("Montmartre District", "neighborhood", 4.3, "4-5 hours", 0.0, "Artistic hill with Sacré-Cœur"),
        Attraction("Seine River Cruise", "activity", 4.2, "1-2 hours", 15.0, "Scenic boat tour of Paris"),
        Attraction("Local Cooking Class", "experience", 4.7, "3-4 hours", 85.0, "Learn French cuisine"),
        Attraction("Versailles Palace", "historic", 4.8, "6-8 hours", 20.0, "Opulent royal palace and gardens"),
        Attraction("Latin Quarter Food Tour", "food", 4.5, "3 hours", 65.0, "Taste local specialties")
    ]
    
    # Filter attractions based on preferences
    preferred_attractions = []
    for attraction in all_attractions:
        # Match user preferences
        if any(pref in attraction.category or pref in attraction.description.lower() 
               for pref in travel_req.preferences):
            preferred_attractions.append(attraction)
        # Also include highly rated landmarks
        elif attraction.category in ["landmark", "historic"] and attraction.rating >= 4.4:
            preferred_attractions.append(attraction)

    # Sort by rating and limit to top options
    preferred_attractions.sort(key=lambda x: x.rating, reverse=True)
    top_attractions = preferred_attractions[:8]  # Limit to 8 attractions

    print(f"✅ Found {len(top_attractions)} attractions matching preferences:")
    for attraction in top_attractions:
        print(f"   • {attraction.name} (⭐{attraction.rating}): ${attraction.cost} - {attraction.estimated_time}")
    


    updated_state = dict(state)
    updated_state.update({
        'attractions': top_attractions,
        'planning_stage': "select_options",
        'completed_steps': state['completed_steps'] + ["attractions_search"]
    })
    return TravelPlannerState(**updated_state)


