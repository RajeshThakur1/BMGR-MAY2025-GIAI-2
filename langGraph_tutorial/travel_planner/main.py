#!/usr/bin/env python3
"""
CLI entry point for the modular travel planner.

Run from the project root:

    python main.py
    python main.py "Plan a trip from NYC to Paris next June"
"""

import argparse
import sys
from travel_planner_module import plan_trip

DEFAULT_SAMPLE = """
I want to plan a romantic trip to Paris for 3 people from June 15-22, 2024.
Our budget is $3000. We love museums, local cuisine, and historic sites.
We're departing from New York.
"""

def main(argv: list[str] | None= None) -> int:
    parser = argparse.ArgumentParser(description="AI Travel Planner (LangGraph)")
    parser.add_argument(
        "request",
        nargs="*",
        help="Natural Language travel request (optional; uses demo text if omitted)"
    )

    args = parser.parse_args(argv)
    if args.request:
        user_request = " ".join(args.request).strip()
    else:
        user_request = DEFAULT_SAMPLE.strip()

    print("🎬 Travel Planner — modular package")
    print("=" * 50)
    print("📝 Request:\n")
    print(user_request)
    print()

    result = plan_trip(user_request)
    return 0 if result is not None else 1

if __name__ == "__main__":
    sys.exit(main())
