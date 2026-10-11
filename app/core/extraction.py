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

# ~4 chars/token, so 16000 chars is ~4000 tokens — a single-prompt extraction
# past this size measurably skims: a 60k-char (32k-token) session run through
# one prompt loses subtle one-line asides that the same content extracts fine
# from when split into smaller windows. Below this, chunking is pure overhead
# for no benefit, so short sessions (the common case) take the exact one-call
# path this always used to, unchanged.
_CHUNK_CHAR_THRESHOLD = 16000
_CHUNK_OVERLAP_TURNS = 1


def _chunk_turns(turns: list[dict]) -> list[list[dict]]:
    """Splits into sequential windows under _CHUNK_CHAR_THRESHOLD chars each,
    with a 1-turn overlap so a fact split across a boundary (a user statement
    immediately followed by the assistant's acknowledgment of it) still has
    both halves visible to at least one window. Whole-turn granularity only —
    never splits a single turn's content — so every chunk transcript stays
    well-formed dialogue the extraction prompt already knows how to read."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_len = 0
    for turn in turns:
        turn_len = len(turn.get("content", ""))
        if current and current_len + turn_len > _CHUNK_CHAR_THRESHOLD:
            chunks.append(current)
            current = current[-_CHUNK_OVERLAP_TURNS:] if _CHUNK_OVERLAP_TURNS else []
            current_len = sum(len(t.get("content", "")) for t in current)
        current.append(turn)
        current_len += turn_len
    if current:
        chunks.append(current)
    return chunks


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


def _extract_chunk(turns: list[dict]) -> list[dict]:
    """One LLM round-trip over one chunk's turns -> parsed, filtered fact
    dicts. Raises the same exceptions extract_facts already treats as
    'take the offline path' — chunking must not introduce a new failure
    mode, only more calls of the existing one."""
    transcript = "\n".join(f"{t.get('role', 'user')}: {t['content']}" for t in turns)
    # 8000 not 2000: 2000 was set when the only thing competing for the
    # budget was the answer itself. It still truncated — measured on a
    # 60k-char session, thinking-disabled extraction hit the 2000 cap at
    # 32 facts and ran to a natural stop at 44 when given room. Facts are
    # ~60 tokens each, so a dense session legitimately needs several
    # thousand tokens of output and truncation costs whole facts, not
    # formatting.
    raw = chat(_L1_PROMPT.format(transcript=transcript), max_tokens=8000)
    parsed = _parse_facts_resilient(raw)
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
    return facts


def extract_facts(turns: list[dict]) -> tuple[list[dict], str]:
    """-> (facts, mode) where mode is 'llm' or 'verbatim'. Facts below
    extraction.min_priority are discarded (spec: immediately, at the source).

    Sessions under _CHUNK_CHAR_THRESHOLD take a single LLM call, exactly as
    before. Longer sessions (long personas, 32k-context benchmarks) are split
    into overlapping windows first — a single prompt over a long transcript
    measurably skims past one-line asides (a date, a name, a passing
    disclosure) that the same content extracts cleanly when the model only
    has to read a few thousand tokens at a time. Any chunk's failure falls
    through to the same whole-session verbatim fallback as the single-call
    path — an LLM outage doesn't care how the input was split."""
    full_transcript = "\n".join(f"{t.get('role', 'user')}: {t['content']}" for t in turns)
    chunks = _chunk_turns(turns) if len(full_transcript) > _CHUNK_CHAR_THRESHOLD else [turns]
    try:
        facts: list[dict] = []
        for chunk in chunks:
            facts.extend(_extract_chunk(chunk))
        if not facts:
            raise ValueError("no parseable fact objects in LLM output")
        return facts[: settings.extraction.max_memories_per_session], "llm"
    except (LlmUnavailable, json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return _verbatim_facts(turns)[: settings.extraction.max_memories_per_session], "verbatim"
