"""Build and run the Travel Planner LangGraph."""

from typing import Any, Optional
from langgraph.graph import END, START, StateGraph

from travel_planner_module.nodes import (
    calculate_budget,
    check_weather,
    create_itinerary,
    find_attractions,
    format_final_plan,
    handle_error,
    process_user_input,
    search_flights,
    search_hotels,
    select_best_options
)

from travel_planner_module.routing import route_planning_stage
from travel_planner_module.state import TravelPlannerState, create_initial_state

def build_graph():
    "Construct the StateGraph with all nodes and conditional edges"
    graph = StateGraph(TravelPlannerState)
    graph.add_node("process_input", process_user_input)
    graph.add_node("search_flights", search_flights)
    graph.add_node("search_hotels", search_hotels)
    graph.add_node("check_weather", check_weather)
    graph.add_node("find_attractions", find_attractions)
    graph.add_node("select_options", select_best_options)
    graph.add_node("create_itinerary", create_itinerary)
    graph.add_node("calculate_budget", calculate_budget)
    graph.add_node("format_plan", format_final_plan)
    graph.add_node("handle_error", handle_error)

    graph.add_edge(START, "process_input")

    graph.add_conditional_edges(
        "process_input",
        route_planning_stage,
        {"flight_search": "search_flights", "error": "handle_error"},
    )
    graph.add_conditional_edges(
        "search_flights",
        route_planning_stage,
        {"hotel_search": "search_hotels", "error": "handle_error"},
    )
    graph.add_conditional_edges(
        "search_hotels",
        route_planning_stage,
        {"weather_check": "check_weather", "error": "handle_error"},
    )
    graph.add_conditional_edges(
        "check_weather",
        route_planning_stage,
        {"attractions_search": "find_attractions", "error": "handle_error"},
    )
    graph.add_conditional_edges(
        "find_attractions",
        route_planning_stage,
        {"select_options": "select_options", "error": "handle_error"},
    )
    graph.add_conditional_edges(
        "select_options",
        route_planning_stage,
        {"create_itinerary": "create_itinerary", "error": "handle_error"},
    )
    graph.add_conditional_edges(
        "create_itinerary",
        route_planning_stage,
        {"calculate_budget": "calculate_budget", "error": "handle_error"},
    )
    graph.add_conditional_edges(
        "calculate_budget",
        route_planning_stage,
        {"format_plan": "format_plan", "error": "handle_error"},
    )
    graph.add_conditional_edges(
        "format_plan",
        route_planning_stage,
        {"end": END, "error": "handle_error"},
    )
    graph.add_edge("handle_error", END)

    return graph.compile()


_compiled_graph = None

def get_compiled_graph():
    """Lazy singleton for the compiled graph (optional reuse)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph



def plan_trip(user_request: str, graph=None) -> Optional[TravelPlannerState]:
    """
    Run the full travel planning workflow from a natural language request.

    Args:
        user_request: User's travel planning request text.
        graph: Optional pre-compiled graph; defaults to get_compiled_graph().

    Returns:
        Final TravelPlannerState, or None if an unexpected exception occurred.
    """
    if graph is None:
        graph = get_compiled_graph()

    print(f"🌍 Planning trip based on: '{user_request}'")
    print("=" * 60)

    initial_state = create_initial_state(user_request)

    try:
        final_state: Any = graph.invoke(initial_state)
        if not isinstance(final_state, dict):
            final_state = dict(final_state)

        print("\n" + "=" * 60)
        print("🎉 TRAVEL PLANNING COMPLETE!")
        print("=" * 60)

        if final_state.get("final_plan"):
            print(final_state["final_plan"])
        else:
            print("❌ No final plan generated")

        return final_state

    except Exception as e:
        print(f"❌ Error during planning: {e}")
        return None
