# app/agents/reflection_agent.py
# Purpose: agent reviews its own answer before sending to user
#
# Interview point: this is the self-improvement loop.
# Before the search agent's response reaches the user, a reflection
# node scores it on three criteria:
# 1. Accuracy — does it use real data from search results?
# 2. Completeness — does it answer what was asked?
# 3. Safety — does it avoid revealing internal details?
#
# If score < threshold, the agent retries with a corrected prompt.
# This catches bad responses before the user sees them.
# In production you'd log all retries to build a correction dataset
# for future fine-tuning.

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.agents.state import BookingState
from app.config import LLM_MODEL, GROQ_API_KEY

llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)

REFLECTION_PROMPT = """You are a quality reviewer for a bus booking chatbot.

Review the assistant's response and score it 1-10 on:
1. Accuracy: does it only use information from the search results?
2. Completeness: does it fully answer what the user asked?
3. Tone: is it helpful and professional?

Return ONLY a JSON dict like this:
{"score": 8, "accuracy": 9, "completeness": 7, "tone": 8, "issue": "missing arrival time"}

If score >= 7, the response is acceptable.
If score < 7, describe the main issue in the "issue" field."""

RETRY_PROMPT = """You are a bus booking assistant. Your previous response had an issue: {issue}

User asked: "{question}"

Available buses:
{context}

Write an improved response that fixes the issue.
Only use information from the bus data above.
Be clear, accurate and helpful."""


def reflection_node(state: BookingState) -> dict:
    """
    Reviews the last agent response and retries if quality is low.

    Interview point: this implements the ReAct pattern —
    Reason (score the response) → Act (retry if needed).
    It's also an example of an agent improving its own output
    without human intervention — key for production reliability.

    We only apply reflection to search responses because:
    - Booking confirmations are deterministic (from DB)
    - Support responses are policy-based
    - Only search responses involve LLM synthesis over retrieved data
      where quality can vary
    """
    responding_agent = state.get("responding_agent", "")

    # only reflect on search agent responses
    if responding_agent != "search":
        return {}

    messages = state.get("messages", [])
    if not messages:
        return {}

    last_response = messages[-1].content
    search_results = state.get("search_results", [])

    # skip reflection if no search results (nothing to verify against)
    if not search_results:
        return {}

    # build context string from search results for verification
    context = "\n".join([
        f"- {r['operator']} | {r['seat_type']} | "
        f"Rs {r['price']} | dep {r['departure']} | "
        f"{r['available']} seats"
        for r in search_results
    ])

    # get the original user question
    user_question = ""
    for msg in reversed(messages[:-1]):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    # score the response
    try:
        score_response = llm.invoke([
            SystemMessage(content=REFLECTION_PROMPT),
            HumanMessage(content=(
                f"User question: {user_question}\n\n"
                f"Search results available:\n{context}\n\n"
                f"Assistant response:\n{last_response}"
            ))
        ])

        import json, re
        cleaned = re.sub(r"```[\w]*\n?", "", score_response.content).strip()
        scores = json.loads(cleaned)
        overall_score = scores.get("score", 7)
        issue = scores.get("issue", "")

        print(f"[Reflection] Score: {overall_score}/10 | Issue: '{issue}'")

        # if score is acceptable, return unchanged
        if overall_score >= 7:
            print(f"[Reflection] Response accepted")
            return {}

        # score too low — retry with corrected prompt
        print(f"[Reflection] Score too low — retrying")
        retry_response = llm.invoke([
            HumanMessage(content=RETRY_PROMPT.format(
                issue=issue,
                question=user_question,
                context=context
            ))
        ])

        print(f"[Reflection] Retry complete")
        return {
            "messages": [AIMessage(content=retry_response.content)],
            "responding_agent": "search_reflected"
        }

    except Exception as e:
        print(f"[Reflection] Error during reflection: {e} — keeping original")
        return {}