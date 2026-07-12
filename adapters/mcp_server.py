"""Mimir MCP adapter: exposes the memory gateway as MCP tools so any MCP
client (Claude Code, Claude Desktop, Cursor, ...) gets persistent memory.

Thin by design (the adapter rule: no business logic, just lifecycle-to-HTTP
translation). Run the gateway first, then register this server:

    uvicorn app.main:app --port 8080

    # e.g. Claude Code:
    claude mcp add mimir -- python adapters/mcp_server.py

Environment:
    MIMIR_URL      gateway base URL   (default http://127.0.0.1:8080)
    MIMIR_API_KEY  bearer key if the gateway has one configured (else keyless local mode)
    MIMIR_USER_ID  identity to remember/recall as (default "local-user")
"""

import os
import uuid

import httpx
from mcp.server.fastmcp import FastMCP

MIMIR_URL = os.environ.get("MIMIR_URL", "http://127.0.0.1:8080")
MIMIR_USER_ID = os.environ.get("MIMIR_USER_ID", "local-user")
_SESSION_ID = f"mcp_{uuid.uuid4().hex[:8]}"  # one session per adapter process

mcp = FastMCP("mimir")


def _post(path: str, payload: dict) -> dict:
    headers = {}
    if os.environ.get("MIMIR_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['MIMIR_API_KEY']}"
    response = httpx.post(f"{MIMIR_URL}/v1{path}", json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


@mcp.tool()
def mimir_recall(query: str) -> str:
    """Recall what is known about the user relevant to a query. Returns a
    memory-context block ready to ground your answer in."""
    result = _post("/recall", {"user_id": MIMIR_USER_ID, "query": query, "session_id": _SESSION_ID})
    return result["context_string"]


@mcp.tool()
def mimir_remember(content: str, role: str = "user") -> str:
    """Store something worth remembering about the user (a fact, preference,
    event, or instruction they stated)."""
    _post("/capture", {
        "user_id": MIMIR_USER_ID, "session_id": _SESSION_ID,
        "messages": [{"role": role, "content": content}],
    })
    return "remembered"


@mcp.tool()
def mimir_flush() -> str:
    """Flush this session into long-term memory: writes the vault scene note,
    extracts facts, refreshes the persona. Call at natural conversation ends."""
    result = _post("/session/end", {"user_id": MIMIR_USER_ID, "session_id": _SESSION_ID})
    return (
        f"flushed: scene={result['note']} facts={result['facts_extracted']} "
        f"(extraction={result['extraction']}, persona={result['persona']})"
    )


if __name__ == "__main__":
    mcp.run()
