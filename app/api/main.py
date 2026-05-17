# app/api/main.py
# Purpose: HTTP interface between the chat UI and the agent graph
#
# Interview point: the API layer is intentionally thin.
# No business logic lives here — it only receives requests,
# passes them to the agent graph, and streams responses back.
# This separation means you can swap the UI without touching agents,
# and swap agents without touching the API.

import json
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from langchain_core.messages import HumanMessage
from app.tools.mcp_client import call_tool
from app.agents.graph import agent_graph

app = FastAPI(title="Bus Booking Agent API")

# CORS allows the chat UI (served from a file or different port)
# to call this API without browser security errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory session store — maps session_id to conversation config
# Interview point: in production this would be Redis with TTL.
# MemorySaver already holds the state — we just need the thread_id
# to be consistent per user session.
sessions: dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    user_id: str = "default_user"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    responding_agent: str


def get_or_create_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "configurable": {"thread_id": session_id},
            "is_new": True
        }
    return sessions[session_id]


def get_initial_state() -> dict:
    """
    Clean state for first message in a session.
    Interview point: we only pass this on the first turn.
    Subsequent turns only pass the new message — checkpointer
    restores everything else automatically.
    """
    return {
        "messages":             [],
        "intent":               "",
        "origin":               None,
        "destination":          None,
        "seat_type":            None,
        "max_price":            None,
        "departure_date":       None,
        "time_of_day":          None,
        "clarification_needed": False,
        "confirmed_origin":     False,
        "search_results":       [],
        "selected_schedule_id": None,
        "booking_confirmation": {},
        "responding_agent":     ""
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — synchronous response.
    Receives a message, runs the agent graph, returns the response.

    Interview point: we use asyncio.to_thread because agent_graph.invoke
    is synchronous (blocking). Wrapping it in to_thread prevents it from
    blocking the FastAPI event loop — other requests can be served
    while the agent is thinking.
    """
    config = get_or_create_session(request.session_id)

    # first message in session gets full initial state
    # subsequent messages only need the new human message
    is_new_session = len(sessions[request.session_id]) == 1

    if is_new_session:
        input_state = {
            **get_initial_state(),
            "messages": [HumanMessage(content=request.message)]
        }
    else:
        input_state = {
            "messages": [HumanMessage(content=request.message)]
        }

    try:
        result = await asyncio.to_thread(
            agent_graph.invoke,
            input_state,
            {
                **config,
                "run_name": f"bus-chat-{request.session_id[:8]}",
                "tags": ["production", "bus-booking"],
                "metadata": {
                    "session_id": request.session_id,
                    "user_id":    request.user_id,
                    }
                }
            )

        response_text = result["messages"][-1].content
        responding_agent = result.get("responding_agent", "unknown")

        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            responding_agent=responding_agent
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    config = get_or_create_session(request.session_id)
    is_new = config.pop("is_new", False)

    if is_new:
        input_state = {
            **get_initial_state(),
            "messages": [HumanMessage(content=request.message)]
        }
    else:
        input_state = {
            "messages": [HumanMessage(content=request.message)]
        }

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'thinking', 'content': '...'})}\n\n"

            result = await asyncio.to_thread(
                agent_graph.invoke,
                input_state,
                {
                    **config,
                    "run_name": f"bus-chat-{request.session_id[:8]}",
                    "tags": ["production", "bus-booking"],
                    "metadata": {
                        "session_id": request.session_id,
                        "user_id": request.user_id,
                    }
                }
            )

            messages = result.get("messages", [])
            if not messages:
                yield f"data: {json.dumps({'type': 'token', 'content': 'Sorry, something went wrong. Please try again.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'agent': 'error', 'search_results': []})}\n\n"
                return

            response_text = messages[-1].content
            responding_agent = result.get("responding_agent", "unknown")
            search_results = result.get("search_results", [])

            print(f"[Stream] agent={responding_agent} results={len(search_results)}")

            words = response_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                await asyncio.sleep(0.03)

            yield f"data: {json.dumps({'type': 'done', 'agent': responding_agent, 'search_results': search_results})}\n\n"

        except Exception as e:
            print(f"[API] Stream error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'token', 'content': 'Something went wrong. Please try again.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'agent': 'error', 'search_results': []})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    """
    Return conversation history for a session.
    Useful for UI to restore conversation on page refresh.
    """
    config = get_or_create_session(session_id)
    try:
        state = agent_graph.get_state(config)
        messages = []
        for msg in state.values.get("messages", []):
            messages.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            })
        return {"session_id": session_id, "messages": messages}
    except Exception:
        return {"session_id": session_id, "messages": []}


@app.get("/health")
def health():
    return {"status": "ok", "service": "bus-booking-agent"}


@app.get("/")
def serve_ui():
    """Serve the chat UI from the static file."""
    ui_path = Path(__file__).parent / "static" / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text())
    return HTMLResponse("<h1>UI not found. Looking for: " + str(ui_path) + "</h1>")


@app.post("/tools/seat_map")
async def get_seat_map(request: Request):
    body = await request.json()
    result = call_tool("get_seat_map", body)
    return result

@app.post("/tools/book_ticket_ui")
async def book_ticket_ui(request: Request):
    body = await request.json()
    result = call_tool("book_ticket", body)
    return result
