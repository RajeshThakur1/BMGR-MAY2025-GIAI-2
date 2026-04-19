"""
Streamlit UI for the Travel Planner LangGraph app.

Run from the project root:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from travel_planner_module import build_graph, create_initial_state

DEFAULT_PROMPT = """I want to plan a romantic trip to Paris for 3 people from June 15-22, 2024.
Our budget is $3000. We love museums, local cuisine, and historic sites.
We're departing from New York."""


@st.cache_resource
def _compiled_graph():
    return build_graph()


def _run_planner(user_request: str, show_steps: bool) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Execute the graph. Returns (final_state, error_message).
    error_message is set only on unexpected exceptions.
    """
    graph = _compiled_graph()
    initial = create_initial_state(user_request.strip())
    if not user_request.strip():
        return None, "Please enter a travel request."

    try:
        if show_steps:
            final_state: Optional[dict[str, Any]] = None
            status = st.status("Planning your trip…", expanded=True)
            try:
                for state in graph.stream(initial, stream_mode="values"):
                    final_state = dict(state) if not isinstance(state, dict) else state
                    stage = final_state.get("planning_stage", "")
                    errs = final_state.get("errors") or []
                    if errs:
                        status.write(f"**{stage}** — note: {errs[-1]}")
                    else:
                        status.write(f"**{stage}**")
                status.update(label="Done", state="complete", expanded=False)
            except Exception:
                status.update(label="Failed", state="error")
                raise
            return final_state, None

        with st.spinner("Planning your trip…"):
            out = graph.invoke(initial)
            final = dict(out) if not isinstance(out, dict) else out
        return final, None

    except Exception as e:
        return None, str(e)


def main() -> None:
    st.set_page_config(
        page_title="Travel Planner",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🌍 AI Travel Planner")
    st.caption("Powered by LangGraph · set `OPENAI_API_KEY` in `.env` or your environment for LLM parsing.")

    with st.sidebar:
        st.subheader("Options")
        show_steps = st.toggle("Show planning steps", value=True)
        st.divider()
        st.markdown(
            """
**How it works**

1. Describe your trip in natural language.
2. The app runs the full planner graph (flights, hotels, weather, attractions, itinerary, budget).
3. Mock data is used for search results unless you wire real APIs.

**Run locally**

```bash
streamlit run streamlit_app.py
```
            """
        )

    if "trip_prompt" not in st.session_state:
        st.session_state.trip_prompt = DEFAULT_PROMPT

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.subheader("Your request")
        if st.button("Load example", type="secondary", use_container_width=True):
            st.session_state.trip_prompt = DEFAULT_PROMPT
            st.rerun()
        prompt = st.text_area(
            "Describe origin, destination, dates, budget, and interests",
            height=220,
            key="trip_prompt",
            label_visibility="collapsed",
        )

        run = st.button("Plan trip", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Result")

        if not run:
            st.info("Enter a request and click **Plan trip**.")
            return

        final_state, err = _run_planner(prompt, show_steps=show_steps)

        if err:
            st.error(err)
            return

        if not final_state:
            st.warning("No result returned.")
            return

        errs = final_state.get("errors") or []
        if errs:
            st.warning("The planner reported issues (see plan below).")

        plan = final_state.get("final_plan") or ""
        if plan:
            st.text_area(
                "Travel plan",
                value=plan,
                height=520,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            st.error("No final plan was generated.")

        with st.expander("State summary"):
            st.json(
                {
                    "planning_stage": final_state.get("planning_stage"),
                    "completed_steps": final_state.get("completed_steps"),
                    "errors": final_state.get("errors"),
                    "total_cost": final_state.get("total_cost"),
                }
            )


if __name__ == "__main__":
    main()
