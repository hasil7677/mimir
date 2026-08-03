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

_L1_PROMPT = """Extract atomic memory facts from this conversation that the user would want remembered later.

Conversation:
{transcript}

Return ONLY a JSON array. Each item:
{{"content": "one complete standalone sentence", "type": "persona|episodic|instruction",
  "priority": 0-100, "scene_name": "short topic label"}}

Rules:
- persona = stable traits/preferences/skills; episodic = events; instruction = rules the user set for the AI
- priority reflects long-term usefulness; trivial chit-chat scores low
- Extract facts about the user, AND any specific information, answers, or recommendations the
  assistant gave that the user would want to recall later (e.g. "the assistant recommended
  learning Ruby, Python, or PHP"). Do not extract facts about the assistant's own nature,
  capabilities, or opinions that aren't useful to recall later.
- A short personal detail mentioned only in passing — one aside buried inside an otherwise
  unrelated, long response (a number, a date, a name, a one-line disclosure like "by the way,
  I used to be...") is often exactly the fact that matters most. Do not skip it just because
  the rest of the conversation is about a different topic.
- When the assistant gives specific named examples (technologies, books, numbers, dates,
  people, places), preserve those specific names in the fact instead of generalizing them
  away — "recommended learning Ruby, Python, or PHP" is a useful fact; "recommended learning
  a back-end language" is not, because it drops the actual answer.
- When the user states a REASON or CAUSE for a choice, feeling, or change of mind, preserve
  the exact reason given, not a paraphrase or a similar-sounding cause — "switched hobbies
  because of awkward social encounters" and "switched hobbies because it felt overwhelming"
  are different facts even though they sound alike and lead to the same outcome. Getting the
  stated reason wrong is as bad as missing the fact entirely.
"""

_MIN_VERBATIM_LENGTH = 15  # skip "yes", "ok thanks" etc. in the offline path

_FACT_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_facts_resilient(raw: str) -> list[dict]:
    """Parses each {...} fact object independently instead of json.loads-ing
    the whole array in one shot. A response cut off mid-array by max_tokens
    (long sessions produce many facts) used to fail the single json.loads on
    the last, incomplete object and silently discard every fact that *did*
    parse fine before it — including, in one observed case, the exact fact a
    query needed. Fact objects are flat (no nested braces), so this is safe."""
    items = []
    for match in _FACT_OBJECT_RE.finditer(raw):
        try:
            items.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return items


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
        # 2000 not 800: a detailed session can generate enough facts that 800
        # truncates the response mid-array — see _parse_facts_resilient for
        # why that used to lose everything, not just the tail.
        raw = chat(_L1_PROMPT.format(transcript=transcript), max_tokens=2000)
        parsed = _parse_facts_resilient(raw)
        if not parsed:
            raise ValueError("no parseable fact objects in LLM output")
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
