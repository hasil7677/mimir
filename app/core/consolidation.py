"""L1.5: conflict detection and deduplication, run on new facts before they
are inserted.

One batched LLM call sees all new facts plus a unified candidate pool of
existing memories (the spec's batch optimization — cross-fact duplicates get
caught in the same pass). Decisions per fact:

  store  — genuinely new, insert as-is
  skip   — an existing memory already says this, drop it
  update — supersedes an existing memory: insert new, mark old inactive

Contradictions are flagged into l1_contradictions and the new fact is STILL
stored — flag-and-log, never auto-resolve. Offline fallback: exact
normalized-text duplicates are skipped, everything else stores; no LLM means
less consolidation, never lost facts.
"""

import json
import re

from app.core.llm import LlmUnavailable, chat
from app.db import l1_store

_CONSOLIDATION_PROMPT = """You are deduplicating an AI memory store.

NEW facts (index: content):
{new_facts}

EXISTING memories (id: content):
{candidates}

For each NEW fact decide:
- "store" if it adds information no existing memory has
- "skip" if an existing memory already fully captures it
- "update" if it replaces/newer-versions a specific existing memory (set target_id)
Also set contradicts_id when a new fact directly conflicts with an existing memory.

Return ONLY a JSON array, one item per NEW fact index:
[{{"index": 0, "decision": "store|skip|update", "target_id": "...", "contradicts_id": "..."}}]
(target_id only for update; contradicts_id only when a conflict exists)
"""


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", " ".join(text.lower().split()))


def consolidate(tenant_id: str, user_id: str, new_facts: list[dict]) -> list[dict]:
    """-> new_facts annotated with '_decision', '_supersedes', '_contradicts'.
    Facts with decision 'skip' are removed. Callers insert survivors, then
    apply supersede/contradiction records once the new ids exist."""
    if not new_facts:
        return []

    # Candidate pool: keyword-nearest existing memories per new fact, unified.
    candidates: dict[str, dict] = {}
    for fact in new_facts:
        for hit in l1_store.search_keyword(tenant_id, user_id, fact["content"], top_k=5):
            candidates[hit["id"]] = hit

    # Offline-safe layer first: exact normalized duplicates never survive.
    surviving: list[dict] = []
    candidate_norms = {_normalize(c["content"]): cid for cid, c in candidates.items()}
    seen_new_norms: set[str] = set()
    for fact in new_facts:
        norm = _normalize(fact["content"])
        if norm in candidate_norms or norm in seen_new_norms:
            continue
        seen_new_norms.add(norm)
        fact["_decision"], fact["_supersedes"], fact["_contradicts"] = "store", None, None
        surviving.append(fact)

    if not surviving or not candidates:
        return surviving

    try:
        new_lines = "\n".join(f"{i}: {f['content']}" for i, f in enumerate(surviving))
        cand_lines = "\n".join(f"{cid}: {c['content']}" for cid, c in candidates.items())
        raw = chat(_CONSOLIDATION_PROMPT.format(new_facts=new_lines, candidates=cand_lines), max_tokens=800)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        decisions = json.loads(match.group(0) if match else raw)
    except (LlmUnavailable, json.JSONDecodeError, AttributeError, TypeError):
        return surviving  # offline: store everything that isn't an exact dup

    valid_ids = set(candidates)
    by_index = {d.get("index"): d for d in decisions if isinstance(d, dict)}
    consolidated: list[dict] = []
    for i, fact in enumerate(surviving):
        decision = by_index.get(i, {})
        verdict = decision.get("decision", "store")
        target = decision.get("target_id")
        contradicts = decision.get("contradicts_id")

        if verdict == "skip":
            continue
        fact["_decision"] = verdict if verdict in ("store", "update") else "store"
        fact["_supersedes"] = target if verdict == "update" and target in valid_ids else None
        fact["_contradicts"] = contradicts if contradicts in valid_ids else None
        consolidated.append(fact)
    return consolidated
