"""L1: atomic fact extraction from session turns.

LLM path returns typed, prioritized facts. Offline fallback keeps recall
functional with zero models: each substantive user turn becomes a verbatim
episodic fact (priority 50, `extraction: verbatim` in metadata) — keyword
search over verbatim facts is still real recall, and the facts can be
re-extracted properly once an LLM is reachable.
"""

import json
import re

from app.config import settings
from app.core.llm import LlmUnavailable, chat

_VALID_TYPES = {"persona", "episodic", "instruction"}

_L1_PROMPT = """Extract atomic memory facts about the user from this conversation.

Conversation:
{transcript}

Return ONLY a JSON array. Each item:
{{"content": "one complete standalone sentence", "type": "persona|episodic|instruction",
  "priority": 0-100, "scene_name": "short topic label"}}

Rules:
- persona = stable traits/preferences/skills; episodic = events; instruction = rules the user set for the AI
- priority reflects long-term usefulness; trivial chit-chat scores low
- no facts about the assistant, only the user
"""

_MIN_VERBATIM_LENGTH = 15  # skip "yes", "ok thanks" etc. in the offline path


def _verbatim_facts(turns: list[dict]) -> list[dict]:
    facts = []
    for turn in turns:
        content = turn["content"].strip()
        if turn.get("role") != "user" or len(content) < _MIN_VERBATIM_LENGTH:
            continue
        facts.append(
            {
                "content": content,
                "type": "episodic",
                "priority": 50,
                "scene_name": content[:40],
                "extraction": "verbatim",
            }
        )
    return facts


def extract_facts(turns: list[dict]) -> tuple[list[dict], str]:
    """-> (facts, mode) where mode is 'llm' or 'verbatim'. Facts below
    extraction.min_priority are discarded (spec: immediately, at the source)."""
    transcript = "\n".join(f"{t.get('role', 'user')}: {t['content']}" for t in turns)
    try:
        raw = chat(_L1_PROMPT.format(transcript=transcript), max_tokens=800)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        parsed = json.loads(match.group(0) if match else raw)
        facts = []
        for item in parsed:
            content = str(item.get("content", "")).strip()
            fact_type = str(item.get("type", "episodic")).strip()
            if not content or fact_type not in _VALID_TYPES:
                continue
            priority = int(item.get("priority", 50))
            if priority < settings.extraction.min_priority:
                continue
            facts.append(
                {
                    "content": content,
                    "type": fact_type,
                    "priority": priority,
                    "scene_name": str(item.get("scene_name", "")).strip()[:60],
                    "extraction": "llm",
                }
            )
        return facts[: settings.extraction.max_memories_per_session], "llm"
    except (LlmUnavailable, json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return _verbatim_facts(turns)[: settings.extraction.max_memories_per_session], "verbatim"
