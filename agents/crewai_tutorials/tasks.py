from crewai import Task
from textwrap import dedent
class TravelTasks:
    def plan_itinerary(self, agent, city, travel_dates, interests):
        """
            Creates a task for the agent to build a 7-day travel itinerary.

            The task requires the agent to:
            - Create a detailed 7-day plan with daily schedules for the trip.
            - Include suggestions for places to visit, restaurants, hotels, and activities.
            - Provide packing tips, weather forecasts, and a budget breakdown.

            The task parameters are:
            - City: The destination city.
            - Trip Dates: The dates of travel.
            - Traveler Interests: The traveler's preferences and interests.

            Returns:
                Task: The fully defined task for itinerary creation.
                """
        return Task(
            description = dedent(f"""**Task**: Develop a 7-Day Travel Itinerary
                                    **Description**: Expand the city guide into a full 7-day travel itinerary
                                    with detailed per-day plans, including weather forecasts, places to eat, packing suggestions,
                                    and a budget breakdown. You MUST suggest actual places to visit, actual hotels to stay,
                                    and actual restaurants to go to. This itinerary should cover all aspects of the trip,
                                    from arrival to departure, integrating the city guide information with practical travel logistics.
                                    **Parameters**:
                                    - City: {city}
                                    - Trip Date: {travel_dates}
                                    - Traveler Interests: {interests}"""),
            expected_output="A detailed 7-day travel itinerary with daily schedules, hotel recommendations, restaurant suggestions, packing tips, weather forecast, and a budget breakdown.",
            agent=agent
        )

            


    def identify_city(self, agent, origin, cities, interests, travel_dates):
        """
            Determines the best city for a trip based on key travel factors.

            This task involves analyzing multiple potential destinations by evaluating weather conditions, 
            seasonal events, and travel costs. The selected city should align with the traveler's interests 
            and provide the best overall experience. The output should be a comprehensive report that includes:
            - Current weather conditions
            - Upcoming cultural or seasonal events
            - Estimated travel expenses, including flight costs
            - Popular attractions and points of interest

            Parameters:
            - agent (Agent): The AI agent responsible for executing the task.
            - origin (str): The starting location of the traveler.
            - cities (list): A list of potential destination cities.
            - interests (list): The traveler's preferences and interests.
            - travel_dates (str): The planned dates for the trip.

            Returns:
            - Task: A CrewAI task assigned to the agent for city selection.
        """
        return Task(
            description = dedent(f"""**Task**: Identify the Best Travel Destination
                                    **Description**: Analyze multiple potential destinations by evaluating weather conditions,
                                    seasonal events, and travel costs. The selected city should align with the traveler's interests
                                    and provide the best overall experience. The output should be a comprehensive report that includes:
                                    - Current weather conditions
                                    - Upcoming cultural or seasonal events
                                    - Estimated travel expenses, including flight costs
                                    - Popular attractions and points of interest

                                    *Key Considerations**:
                                        - Compare weather forecasts, seasonal festivals, and cultural events.
                                        - Analyze estimated travel costs, including flights and accommodations.
                                        - Ensure the chosen city aligns with the traveler's preferences and interests.
                                        - Provide a detailed report on the selected destination, including key attractions.

                                    
                                    **Parameters**:
                                    - Origin: {origin}
                                    - Cities: {cities}
                                    - Traveler Interests: {interests}
                                    - Travel Date: {travel_dates}
                                    """),
            expected_output="""A comprehensive report that includes:
                            - Current weather conditions
                            - Upcoming cultural or seasonal events
                            - Estimated travel expenses, including flight costs
                            - Popular attractions and points of interest""",
            agent=agent
        )



    def gather_city_info(self, agent, city,travel_dates, interests):
        """
            Collects detailed travel information about a specific city.

            This task involves compiling a comprehensive city guide that includes key attractions, 
            cultural insights, local events, and travel logistics. The goal is to provide a well-rounded 
            overview of the city's highlights to help travelers make informed decisions.

            Parameters:
            - agent (Agent): The AI agent responsible for gathering and structuring the information.
            - city (str): The selected city for which information is being collected.
            - travel_dates (str): The planned dates of the trip.
            - interests (list): The traveler's preferences and interests.

            Returns:
            - Task: A CrewAI task assigned to the agent for collecting city-specific travel insights.
        """
        return Task(
            description = dedent(f"""**Task**: Gather Detailed City Information
                                    **Description**: Compile a comprehensive city guide that includes key attractions,
                                    cultural insights, local events, and travel logistics. The goal is to provide a well-rounded
                                    overview of the city's highlights to help travelers make informed decisions.
                                    **Parameters**:
                                    - City: {city}
                                    - Travel Date: {travel_dates}
                                    - Traveler Interests: {interests}"""),
            expected_output="A comprehensive city guide with key attractions, cultural insights, local events, and travel logistics.",
            agent=agent
        )