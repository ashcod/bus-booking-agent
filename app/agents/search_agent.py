# app/agents/search_agent.py
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from app.agents.state import BookingState
from app.rag.retriever import retrieve
from app.core.config import LLM_MODEL, GROQ_API_KEY
from app.memory.user_memory import record_search, get_memory_context

llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)


def search_agent_node(state: BookingState) -> dict:
    last_message = state["messages"][-1].content.lower()
    origin      = state.get("origin")
    destination = state.get("destination")

    # if both are missing, ask the user
    if not origin and not destination:
        return {
            "search_results": [],
            "responding_agent": "search",
            "messages": [AIMessage(content=(
                "I'd be happy to help you find a bus! "
                "Could you tell me your departure city and destination? "
                "For example: 'buses from Hyderabad to Bangalore'"
            ))]
        }

    # detect "show all" — clear every filter, show full route
    show_all = any(phrase in last_message for phrase in [
        "show all", "all buses", "any bus", "all options",
        "show everything", "no preference", "any time",
        "couldn't find", "try again"
    ])

    if show_all:
        seat_type  = None
        max_price  = None
        time_of_day = None
        print(f"[Search Agent] 'show all' detected — clearing all filters")
    else:
        seat_type   = state.get("seat_type")
        max_price   = state.get("max_price")
        time_of_day = None if state.get("time_of_day") == "all" \
                      else state.get("time_of_day")

    # record search in long-term memory
    user_id = "default_user"
    if origin and destination:
        record_search(user_id, origin, destination)

    memory_context = get_memory_context(user_id)

    # first attempt with current filters
    results = _retrieve_with_fallback(
        last_message, origin, destination,
        seat_type, max_price, time_of_day
    )

    print(f"[Search Agent] Found {len(results)} results")

    if not results:
        return {
            "search_results": [],
            "responding_agent": "search",
            "messages": [AIMessage(content=(
                f"I couldn't find any available buses from {origin} to {destination} "
                f"matching your criteria.\n\n"
                f"Try:\n"
                f"- Say 'show all buses' to see every available option\n"
                f"- Adjust your time preference\n"
                f"- Check a different route"
            ))]
        }

    context = "\n\n".join([
        f"Option {i+1}:\n"
        f"  ID: {r['schedule_id']}\n"
        f"  Operator: {r['operator']}\n"
        f"  Route: {r['origin']} -> {r['destination']}\n"
        f"  Seat type: {r['seat_type']}\n"
        f"  Departure: {r['departure']} | Arrival: {r['arrival']}\n"
        f"  Price: Rs {r['price']}\n"
        f"  Available seats: {r['available']}\n"
        f"  Operates: {r['days']}"
        for i, r in enumerate(results)
    ])

    prompt = f"""You are a bus booking assistant. Present ONLY these bus options.

User profile:
{memory_context}

User asked: "{last_message}"

Available buses:
{context}

List each option clearly with operator, departure time, seat type, price and seats.
End by telling the user to say 'book SC00000' with the actual schedule ID.
Do NOT answer any other topic."""

    response = llm.invoke([SystemMessage(content=prompt)])

    return {
        "search_results":   results,
        "responding_agent": "search",
        "messages":         [AIMessage(content=response.content)]
    }


def _retrieve_with_fallback(
    query, origin, destination,
    seat_type, max_price, time_of_day
) -> list:
    """
    Try retrieval with filters first.
    If no results, automatically fall back to no filters.
    Interview point: graceful degradation — always try to show
    something useful rather than an empty result.
    """
    results = retrieve(
        query=query,
        origin=origin,
        destination=destination,
        seat_type=seat_type,
        max_price=max_price,
        time_of_day=time_of_day,
        top_k=5
    )

    # if no results with filters, try without time filter
    if not results and time_of_day:
        print(f"[Search Agent] No results with time={time_of_day}, retrying without time filter")
        results = retrieve(
            query=query,
            origin=origin,
            destination=destination,
            seat_type=seat_type,
            max_price=max_price,
            time_of_day=None,
            top_k=5
        )

    # if still no results, try with only origin+destination
    if not results and (seat_type or max_price):
        print(f"[Search Agent] No results with filters, retrying with origin+destination only")
        results = retrieve(
            query=query,
            origin=origin,
            destination=destination,
            seat_type=None,
            max_price=None,
            time_of_day=None,
            top_k=5
        )

    return results
