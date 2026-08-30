from textwrap import dedent
from crew import TripCrew


if __name__ == "__main__":
    print("## Welcome to Trip Planner Crew")
    print('-------------------------------')

    origin = input(dedent("From where will you be traveling from?\n"))
    cities = input(dedent("What are the cities options you are interested in visiting?\n"))
    date_range = input(dedent("What is the date range you are interested in traveling?\n"))
    interests = input(dedent("What are some of your high-level interests and hobbies?\n"))

    trip_crew = TripCrew(origin, cities, date_range, interests)
    result = trip_crew.run()

    print("\n\n########################")
    print("## Here is your Trip Plan")
    print("########################\n")
    print(result)