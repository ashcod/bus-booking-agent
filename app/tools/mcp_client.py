# app/tools/mcp_client.py
# Purpose: thin HTTP client that agents use to call the tool server
# Interview point: agents never import tool_server.py directly.
# They always go through this client — that's the MCP pattern.
# Swap the URL and all agents automatically use a different tool server.

import httpx
from app.core.config import TOOL_SERVER_URL


def call_tool(tool_name: str, params: dict) -> dict:
    """
    Call any tool on the MCP server by name.
    Interview point: one function to call any tool —
    agents don't need to know HTTP details.
    """
    url = f"{TOOL_SERVER_URL}/tools/{tool_name}"
    try:
        response = httpx.post(url, json=params, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "detail": e.response.json()}
    except httpx.ConnectError:
        return {"error": "Tool server not reachable. Is it running?"}


def discover_tools() -> list[dict]:
    """List all available tools from the server."""
    try:
        response = httpx.get(f"{TOOL_SERVER_URL}/tools", timeout=5.0)
        return response.json()["tools"]
    except Exception as e:
        return []
