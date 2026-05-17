# app/agents/clarification_agent.py
from langchain_core.messages import AIMessage
from app.agents.state import BookingState

DEFAULT_CITY = "Hyderabad"

KNOWN_CITIES = [
    "Hyderabad", "Bangalore", "Chennai", "Mumbai", "Pune",
    "Delhi", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
    "Nagpur", "Visakhapatnam", "Kochi", "Coimbatore", "Madurai"
]


def clarification_node(state: BookingState) -> dict:
    origin = state.get("origin")
    destination = state.get("destination")
    time_of_day = state.get("time_of_day")
    intent = state.get("intent", "search")
    messages = state.get("messages", [])

    # refinement requests skip clarification entirely
    # they already have origin+destination from previous search
    if intent == "refine" and origin and destination:
        return {
            "clarification_needed": False,
            "responding_agent":     "clarification"
        }

    # check last user message for "show all" or time keywords
    last_user_msg = ""
    for msg in reversed(messages):
        from langchain_core.messages import HumanMessage
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content.lower()
            break

    # if user said "show all" — set time_of_day to "all" and proceed
    if "show all" in last_user_msg:
        return {
            "time_of_day":          "all",
            "clarification_needed": False,
            "responding_agent":     "clarification"
        }

    # check if user mentioned a time preference in their message
    time_keywords = {
        "morning":   "morning",
        "afternoon": "afternoon",
        "evening":   "evening",
        "night":     "night",
        "am":        "morning",
        "pm":        "evening",
    }
    detected_time = None
    for keyword, value in time_keywords.items():
        if keyword in last_user_msg:
            detected_time = value
            break

    if detected_time and not time_of_day:
        return {
            "time_of_day":          detected_time,
            "clarification_needed": False,
            "responding_agent":     "clarification"
        }

    # case 1: no destination
    if not destination:
        return {
            "clarification_needed": True,
            "responding_agent":     "clarification",
            "messages": [AIMessage(content=(
                "I'd love to help you find a bus! "
                "Which city are you travelling to?"
            ))]
        }

    # case 2: no origin — assume Hyderabad, ask to confirm
    if not origin:
        return {
            "origin":               DEFAULT_CITY,
            "clarification_needed": True,
            "responding_agent":     "clarification",
            "messages": [AIMessage(content=(
                f"Great, travelling to {destination}! "
                f"Are you departing from {DEFAULT_CITY}? "
                f"If not, let me know your departure city."
            ))]
        }

    # case 3: no time preference
    if not time_of_day:
        return {
            "clarification_needed": True,
            "responding_agent":     "clarification",
            "messages": [AIMessage(content=(
                f"Got it — {origin} to {destination}. "
                f"Do you have a time preference?\n"
                f"  - Morning (6am-12pm)\n"
                f"  - Afternoon (12pm-5pm)\n"
                f"  - Evening (5pm-9pm)\n"
                f"  - Night (9pm onwards)\n\n"
                f"Or say 'show all' to see every available bus."
            ))]
        }

    # all info collected
    return {
        "clarification_needed": False,
        "responding_agent":     "clarification"
    }


def needs_clarification(state: BookingState) -> str:
    clarification_needed = state.get("clarification_needed", False)

    print(f"[Clarification Check] origin={state.get('origin')}, "
          f"destination={state.get('destination')}, "
          f"time_of_day={state.get('time_of_day')}, "
          f"clarification_needed={clarification_needed}")

    if clarification_needed:
        print("[Clarification Check] -> END (returning question to user)")
        return "end"

    print("[Clarification Check] -> search_agent")
    return "search_agent"
