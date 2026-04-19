from travel_planner_module.state import TravelPlannerState

def route_planning_stage(state:TravelPlannerState) -> str:
    """
    Return the current planning_stage as the routing key.
    Each node sets planning_stage to the next stage before returning.
    """

    stage = state['planning_stage']
    print(f"Routing to next stage: {stage}")

    if state['errors']:
        return "error"
    
    return stage


