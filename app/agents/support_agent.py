# app/agents/support_agent.py
# Purpose: handle cancellations, refund queries, general support
# Kept simple for portfolio — shows agent specialisation pattern

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.agents.state import BookingState
from langchain_groq import ChatGroq
from app.config import LLM_MODEL, GROQ_API_KEY

llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)

SUPPORT_PROMPT = """You are a helpful customer support agent for a bus booking service.
You handle: cancellations, refund queries, complaints, and general questions.

Refund policy:
- Cancelled 24hrs before departure: 90% refund
- Cancelled 12-24hrs before: 50% refund  
- Cancelled under 12hrs: no refund

For cancellation requests, ask for their booking ID (format: BK followed by 8 characters).
Be empathetic, concise, and helpful. Do not make up booking details."""


def support_agent_node(state: BookingState) -> dict:
    last_message = state["messages"][-1].content
    
    # second line of defence — support agent should never reveal internals
    # even if guardrail missed it
    sensitive_keywords = [
        "system prompt", "your prompt", "instructions", "what model",
        "what llm", "source code", "api key", "architecture",
        "who made you", "who created you", "what ai"
    ]
    if any(kw in last_message.lower() for kw in sensitive_keywords):
        from langchain_core.messages import AIMessage
        return {
            "responding_agent": "support",
            "messages": [AIMessage(content=(
                "I can only help with bus bookings, cancellations, "
                "and travel queries. What can I help you with today?"
            ))]
        }

    # pass full conversation history for context
    history = state["messages"][:-1]

    messages = [SystemMessage(content=SUPPORT_PROMPT)]
    messages.extend(history)
    messages.append(HumanMessage(content=last_message))

    response = llm.invoke(messages)

    return {
        "responding_agent": "support",
        "messages": [AIMessage(content=response.content)]
    }