"""Aggregate errors into a user-visible final plan."""

from travel_planner_module.state import TravelPlannerState

def handle_error(state: TravelPlannerState) -> TravelPlannerState:
    """Handle any errors that occurred during planning."""
    print("❌ Handling planning errors...")

    error_summary = "\n".join(state["errors"])
    final_plan = f"""
🚨 TRAVEL PLANNING ERROR 🚨
═══════════════════════════

Unfortunately, we encountered some issues while planning your trip:

{error_summary}

Please try again with different parameters or contact support for assistance.

Completed steps: {', '.join(state['completed_steps'])}
"""

    updated_state = dict(state)
    updated_state.update({"final_plan": final_plan, "planning_stage": "error_handled"})
    return TravelPlannerState(**updated_state)