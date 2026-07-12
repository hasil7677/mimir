"""Mimir embedded MCP adapter: the whole engine in-process, no gateway.

The Obsidian model — your memory is just files (~/.mimir: DuckDB file,
Qdrant directory, markdown vault), and a process only exists while an MCP
client session is open. Claude Code spawns this script when a session starts
and kills it when the session ends; there is nothing to keep running.

    claude mcp add mimir --scope user -e MIMIR_USER_ID=you -- \
        python C:/path/to/mimir/engine/adapters/mcp_embedded.py

Environment:
    MIMIR_USER_ID  identity to remember/recall as (default "local-user")
    MIMIR_CONFIG   optional path to a mimir.yaml (embeddings/LLM config etc.)

Trade-off vs the gateway adapter (mcp_server.py): DuckDB and embedded Qdrant
are single-writer, so run ONE session at a time against the same ~/.mimir.
For concurrent sessions or other machines, run the gateway and use
mcp_server.py instead. Same tools, same files, swap any time.
"""

import os
import sys
import uuid
from pathlib import Path

# Runnable directly by path (how MCP clients invoke it) — make `app` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import functools

from mcp.server.fastmcp import FastMCP

from app.core import pipeline, recall as recall_pipeline

TENANT = "local"  # embedded mode is single-tenant by definition
USER_ID = os.environ.get("MIMIR_USER_ID", "local-user")
_SESSION_ID = f"mcp_{uuid.uuid4().hex[:8]}"  # one session per adapter process

mcp = FastMCP("mimir")

_LOCKED_MESSAGE = (
    "Mimir's local store is held by another running session (DuckDB/Qdrant are "
    "single-writer). Close the other agent session, or run the gateway "
    "(uvicorn app.main:app) and switch to adapters/mcp_server.py for "
    "concurrent access. This session's conversation is unaffected."
)


def _graceful_on_lock(tool):
    """DuckDB is opened lazily on first tool call — NOT at adapter startup —
    so a second agent session can still connect and list tools while another
    holds the store. Only actual memory use collides, and it answers with an
    explanation instead of a stack trace."""

    @functools.wraps(tool)
    def wrapper(*args, **kwargs):
        try:
            return tool(*args, **kwargs)
        except Exception as exc:
            if "lock" in str(exc).lower() or "Conflicting" in str(exc):
                return _LOCKED_MESSAGE
            raise

    return wrapper


@mcp.tool()
@_graceful_on_lock
def mimir_recall(query: str) -> str:
    """Recall what is known about the user relevant to a query. Returns a
    memory-context block ready to ground your answer in."""
    result = recall_pipeline.recall(TENANT, USER_ID, query, _SESSION_ID)
    return result["context_string"]


@mcp.tool()
@_graceful_on_lock
def mimir_remember(content: str, role: str = "user") -> str:
    """Store something worth remembering about the user (a fact, preference,
    event, or instruction they stated)."""
    pipeline.capture(TENANT, USER_ID, _SESSION_ID, [{"role": role, "content": content}])
    return "remembered"


@mcp.tool()
@_graceful_on_lock
def mimir_flush() -> str:
    """Flush this session into long-term memory: writes the vault scene note,
    extracts + dedupes facts, refreshes the persona. Call at natural
    conversation ends.

    Also recovers any OTHER session that captured turns but never got
    flushed — e.g. if this adapter process was restarted mid-conversation
    (a Claude Code /model switch, a crash) and lost track of the session_id
    it was using. Nothing captured via mimir_remember is ever silently
    stranded; this always finds and distills it eventually.
    """
    summary = pipeline.flush_all_pending(TENANT, USER_ID, _SESSION_ID)
    if summary["sessions_flushed"] == 0:
        return "nothing captured this session"
    lines = [f"flushed {summary['sessions_flushed']} session(s):"]
    for session_id, result in summary["results"].items():
        marker = " (this session)" if session_id == _SESSION_ID else " (recovered orphan)"
        lines.append(
            f"- {session_id}{marker}: scene={result['note']} facts={result['facts_extracted']} "
            f"(extraction={result['extraction']}, persona={result['persona']})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # Deliberately no ensure_schema() here: opening DuckDB at startup would
    # make a second session fail to CONNECT while any other session is alive.
    # The schema is ensured lazily by the first real tool call instead.
    mcp.run()
