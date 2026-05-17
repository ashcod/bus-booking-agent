# app/agents/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import BookingState
from app.agents.orchestrator import orchestrator_node
from app.agents.search_agent import search_agent_node
from app.agents.booking_agent import booking_agent_node
from app.agents.support_agent import support_agent_node
from app.agents.clarification_agent import clarification_node, needs_clarification
from langchain_core.tracers.langchain import wait_for_all_tracers


def route_after_orchestrator(state: BookingState) -> str:
    intent = state.get("intent", "unclear")

    if intent == "blocked":
        return "end"
    if intent in ("search", "refine"):
        return "clarification_agent"

    routing_map = {
        "book":    "booking_agent",
        "cancel":  "support_agent",
        "support": "support_agent",
        "unclear": "support_agent",
    }
    return routing_map.get(intent, "support_agent")


def build_graph():
    graph = StateGraph(BookingState)

    graph.add_node("orchestrator",        orchestrator_node)
    graph.add_node("clarification_agent", clarification_node)
    graph.add_node("search_agent",        search_agent_node)
    graph.add_node("booking_agent",       booking_agent_node)
    graph.add_node("support_agent",       support_agent_node)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "end":                 END,
            "clarification_agent": "clarification_agent",
            "booking_agent":       "booking_agent",
            "support_agent":       "support_agent",
        }
    )

    graph.add_conditional_edges(
        "clarification_agent",
        needs_clarification,
        {
            "end":          END,
            "search_agent": "search_agent",
        }
    )

    graph.add_edge("search_agent",  END)
    graph.add_edge("booking_agent", END)
    graph.add_edge("support_agent", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


agent_graph = build_graph()


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    config = {"configurable": {"thread_id": "test-session-1"}}

    # reusable blank state
    initial_state = {
        "messages":            [],
        "intent":              "",
        "origin":              None,
        "destination":         None,
        "seat_type":           None,
        "max_price":           None,
        "departure_date":      None,
        "time_of_day":         None,
        "clarification_needed":False,
        "confirmed_origin":    False,
        "search_results":      [],
        "selected_schedule_id":None,
        "booking_confirmation":{},
        "responding_agent":    ""
    }

    print("=" * 50)
    print("MULTI-TURN CONVERSATION TEST")
    print("=" * 50)

      # turn 1 — user gives only destination
    print("\nUser: buses to Bangalore")
    result = agent_graph.invoke(
        {**initial_state,
         "messages": [HumanMessage(content="buses to Bangalore")]},
        config=config
    )
    print(f"Bot: {result['messages'][-1].content}")

    # turn 2 — user confirms origin
    # checkpointer restores full state from turn 1 automatically
    print("\nUser: yes from Hyderabad")
    result = agent_graph.invoke(
        {"messages": [HumanMessage(content="yes from Hyderabad")]},
        config=config    # same thread_id = same session
    )
    print(f"Bot: {result['messages'][-1].content}")

    # turn 3 — user gives time preference
    print("\nUser: evening please")
    result = agent_graph.invoke(
        {"messages": [HumanMessage(content="evening please")]},
        config=config
    )
    print(f"Bot: {result['messages'][-1].content}")

    # second session — new thread_id but same user_id
    # long-term memory should pre-fill origin
    print("\n" + "=" * 50)
    print("SECOND SESSION — new conversation, same user")
    print("=" * 50)

    config2 = {"configurable": {"thread_id": "test-session-2"}}

    print("\nUser: buses to Chennai")
    result = agent_graph.invoke(
        {**initial_state,
         "messages": [HumanMessage(content="buses to Chennai")]},
        config=config2
    )
    print(f"Bot: {result['messages'][-1].content}")
