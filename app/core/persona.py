"""L3: the persona document — the user's stable identity, synthesized from
accumulated facts into vault persona.md.

Trigger is count-based: re-synthesize once l3_every_n_memories new active
facts have landed since the last synthesis (tracked in the persona note's own
frontmatter, so the vault carries its own bookkeeping — no separate state
store to drift out of sync).

Offline fallback is a prioritized digest of persona/instruction facts:
structurally the same document, honestly labeled `synthesis: digest`.
"""

from app.config import settings
from app.core import vault
from app.core.llm import LlmUnavailable, chat
from app.db import l1_store

_PERSONA_PROMPT = """You maintain a persona document about a user for an AI assistant.

Current persona document:
{existing}

Highest-priority facts on record:
{facts}

Rewrite the persona document as clean markdown: identity, preferences, habits,
skills, and standing instructions, as concise bullet points. Preserve still-true
older information; drop anything clearly superseded. Return ONLY the markdown body.
"""


def _digest(facts: list[dict]) -> str:
    lines = ["# Persona (digest)", ""]
    persona_facts = [f for f in facts if f["type"] == "persona"]
    instructions = [f for f in facts if f["type"] == "instruction"]
    others = [f for f in facts if f["type"] not in ("persona", "instruction")]
    if persona_facts:
        lines += ["## Traits & preferences"] + [f"- {f['content']}" for f in persona_facts] + [""]
    if instructions:
        lines += ["## Standing instructions"] + [f"- {f['content']}" for f in instructions] + [""]
    if others and not persona_facts:
        lines += ["## Recent context"] + [f"- {f['content']}" for f in others[:8]]
    return "\n".join(lines).strip()


def maybe_synthesize(tenant_id: str, user_id: str) -> str | None:
    """-> 'llm' | 'digest' if a synthesis ran, None if not due yet."""
    fact_count = l1_store.count_active_facts(tenant_id, user_id)
    if fact_count == 0:
        return None

    existing = vault.read_note(tenant_id, user_id, "persona")
    last_count = (existing[0].get("fact_count", 0) if existing else 0) or 0
    if existing is not None and fact_count - last_count < settings.pipeline.l3_every_n_memories:
        return None

    facts = l1_store.get_top_facts(tenant_id, user_id, limit=30)
    try:
        body = chat(
            _PERSONA_PROMPT.format(
                existing=existing[1] if existing else "(none yet)",
                facts="\n".join(f"- [{f['type']}, p{f['priority']}] {f['content']}" for f in facts),
            ),
            max_tokens=700,
        ).strip()
        mode = "llm"
    except LlmUnavailable:
        body, mode = _digest(facts), "digest"

    vault.upsert_persona(
        tenant_id, user_id, body,
        extra_frontmatter={"fact_count": fact_count, "synthesis": mode},
    )
    return mode


def profile_lines(tenant_id: str, user_id: str, limit: int = 5) -> list[str]:
    """Top persona lines for the recall context's USER PROFILE section —
    read fresh from the vault, so hand edits show up immediately."""
    note = vault.read_note(tenant_id, user_id, "persona")
    if note is None:
        return []
    lines = [ln.strip() for ln in note[1].splitlines() if ln.strip().startswith("- ")]
    return lines[:limit]
