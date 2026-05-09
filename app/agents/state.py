# app/agents/state.py
# Purpose: defines the shared state that flows through every node in the graph
# Interview point: LangGraph passes this state object between every agent.
# Each agent reads what it needs and writes its results back.
# This is how A2A (agent-to-agent) communication works in LangGraph —
# not direct function calls, but shared state mutations.

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class BookingState(TypedDict):
    messages:             Annotated[list, add_messages]
    intent:               str
    origin:               str
    destination:          str
    seat_type:            str
    max_price:            float
    departure_date:       str
    time_of_day:          str
    sort_by:              str        # add this
    clarification_needed: bool
    confirmed_origin:     bool
    search_results:       list[dict]
    selected_schedule_id: str
    booking_confirmation: dict
    responding_agent:     str