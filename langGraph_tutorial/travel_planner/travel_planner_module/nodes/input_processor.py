"""Parse natural language into structured TravelRequest using the LLM."""


import json
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage

from travel_planner_module.models import TravelRequest
from travel_planner_module.state import TravelPlannerState
from utils.utils_functions import get_llm


def process_user_input(state: TravelPlannerState) -> TravelPlannerState:
    """
    Parse user input and extract travel requirements using LLM.
    
    This node:
    1. Analyzes the user's travel request
    2. Extracts key information (dates, destination, budget, etc.)
    3. Validates the input
    4. Creates a structured TravelRequest object
    """

    print("Processing user input to extract travel requirements...")
    user_request = state["user_request"]


    try:
        llm = get_llm()
        if llm is None:
            state["errors"].append("LLM initialization failed: No model returned")
            return state
        
        print("LLM initialized successfully. Processing input...")
        system_prompt = """You are a travel planning assistant. Extract travel information from user requests.
            
            Please extract the following information and return it in JSON format:
            {
                "origin": "departure city",
                "destination": "destination city", 
                "departure_date": "YYYY-MM-DD format",
                "return_date": "YYYY-MM-DD format",
                "travelers": number_of_travelers,
                "budget": budget_amount_in_dollars,
                "preferences": ["list", "of", "preferences"],
                "trip_type": "business|leisure|adventure|family"
            }
            
            If any information is missing, make reasonable assumptions based on context.
            For dates, if not specified, assume departure is 30 days from now and return is 7 days later.
            For budget, if not specified, assume $2000 per person.
            For preferences, infer from the context (e.g., museums, local cuisine, history, nature, nightlife, shopping, architecture).
            strictly return only the JSON object without any additional text or explanations or ```json delimiters.
            """
        
        messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User request: {user_request}")
            ]
        response = llm.invoke(messages)
        # Parse JSON response

        import json
        try:
            print("LLM response received. Attempting to parse JSON...")
            print("LLM response content:", response.content)
            extracted_data = json.loads(response.content)

        except json.JSONDecodeError as e:
            state["errors"].append(f"Failed to parse LLM response as JSON: {str(e)}")
            return state
        # Create TravelRequest object
        travel_request = TravelRequest(
            origin=extracted_data.get("origin", "Unknown"),
            destination=extracted_data.get("destination", "Unknown"),
            departure_date=extracted_data.get("departure_date", (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")),
            return_date=extracted_data.get("return_date", (datetime.now() + timedelta(days=37)).strftime("%Y-%m-%d")),
            travelers=extracted_data.get("travelers", 1),
            budget=extracted_data.get("budget", 2000.0),
            preferences=extracted_data.get("preferences", []),
            trip_type=extracted_data.get("trip_type", "leisure")
        )

        print("Extracted travel request:", travel_request)
        
        # Create updated state
        updated_state = dict(state)

        updated_state.update({
            'travel_request': travel_request,
            'planning_stage': "flight_search",
            'completed_steps': state['completed_steps'] + ["input_processing"]
        })

        return TravelPlannerState(**updated_state)
    
    
    except Exception as e:
        state["errors"].append(f"LLM initialization failed: {str(e)}")
        return state