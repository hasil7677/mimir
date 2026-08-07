"""The recall pipeline: hot inject -> hybrid search -> RRF -> 4-signal
scoring -> vault enrichment -> context string.

Every leg degrades independently: no Redis = no hot turns, no embedding
endpoint = keyword-only candidates, empty vault = no graph signal. The
pipeline's contract is "best context available right now", never an error
because a dependency is down.
"""

import logging
from datetime import datetime, timezone

from app.config import settings
from app.core import persona, scoring, semantic_cache, synthesis, vault
from app.core.embeddings import EmbeddingsUnavailable, embed, embed_hybrid, supports_hybrid
from app.db import l1_store, redis_client, vector_store

logger = logging.getLogger(__name__)


def _graph_hops(tenant_id: str, user_id: str, query: str) -> dict[str, int]:
    """entity name (lowercased) -> hop distance from the query. Hop 0 =
    named in the query itself; hops 1-2 = reached via vault wikilinks
    (typically entity -> scene it was mentioned in -> other entities
    mentioned in that same scene)."""
    hops: dict[str, int] = {}
    query_entities = synthesis.extract_entities_naive(query)
    for name in query_entities:
        hops[name.lower()] = 0

    # Seed the frontier with each query entity's OWN note body, not a
    # re-resolution of the entity itself — resolving "[[Sarah]]" just
    # fetches Sarah's note again, which is already known at hop 0 and gets
    # dropped by the `not in hops` check below, wasting the first iteration.
    # Starting from the note's contents means hop 1 lands on what Sarah's
    # note actually links to (the scenes she's mentioned in).
    seeded = vault.expand_links(tenant_id, user_id, [f"[[{e}]]" for e in query_entities], hops=1)
    frontier = list(seeded.values())

    for hop in (1, 2):
        expanded = vault.expand_links(tenant_id, user_id, frontier, hops=1)
        frontier = []
        for name, body in expanded.items():
            frontier.append(body)  # keep expanding even if already seen at an earlier hop
            # Use the note's own heading, not the wikilink target, as the
            # key: multi-word entities are wikilinked as their slug
            # ("iron-temple"), which never substring-matches natural fact
            # prose ("...at Iron Temple..."). The note's "# Iron Temple"
            # heading gives back the space-separated form hop 0 already
            # uses, so both hop levels match fact content the same way.
            display = _note_heading(body, fallback=name).lower()
            if display not in hops:
                hops[display] = hop
    return hops


def _note_heading(body: str, fallback: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def _fact_graph_hops(content: str, entity_hops: dict[str, int]) -> int | None:
    content_lower = content.lower()
    found = [hop for name, hop in entity_hops.items() if name in content_lower]
    return min(found) if found else None


def recall(tenant_id: str, user_id: str, query: str, session_id: str | None = None) -> dict:
    # 0. Semantic cache intercept: embed once (reused by the vector leg),
    # exact-hash hit needs no embeddings at all.
    query_vector: list[float] | None = None
    hybrid_query: dict | None = None
    try:
        if supports_hybrid():
            hybrid_query = embed_hybrid([query])[0]
            query_vector = hybrid_query["dense"]
        else:
            query_vector = embed([query])[0]
    except EmbeddingsUnavailable:
        logger.info("embeddings unavailable — keyword-only recall, exact-only cache")

    cached = semantic_cache.get(tenant_id, user_id, query, query_vector)
    if cached is not None:
        cached_ids = [m["id"] for m in cached.get("memories", [])]
        if cached_ids:
            l1_store.bump_access(cached_ids)
        return {**cached, "cache_hit": True}

    # 1. Hot memory: recent turns, verbatim, unscored (degrades to empty).
    hot_turns: list[dict] = []
    if session_id:
        try:
            hot_turns = redis_client.get_recent_turns(tenant_id, user_id, session_id)
        except Exception:
            logger.info("hot memory unavailable — recall continues without recent turns")

    # 2. Search. Two paths:
    #   - hybrid (fastembed configured): Qdrant's native dense+sparse prefetch
    #     fused with RRF server-side — no DuckDB-fts, no our own rrf_merge.
    #   - legacy (openai/none): DuckDB-fts keyword search + optional Qdrant
    #     dense-only, merged with our own rrf_merge below. Also the automatic
    #     fallback if the hybrid index is simply empty yet (e.g. just switched
    #     providers and nothing's been re-embedded there) — DuckDB is always
    #     available, so recall never regresses to zero results either way.
    keyword_hits: list[dict] = []
    facts_by_id: dict[str, dict] = {}
    semantic_by_id: dict[str, float] = {}
    vector_used = False
    hybrid_used = False

    if hybrid_query is not None:
        hits = vector_store.search_hybrid(tenant_id, user_id, hybrid_query["dense"], hybrid_query["sparse"], top_k=20)
        if hits:
            raw_scores = {h["id"]: h["semantic_score"] for h in hits}
            # Qdrant's fused RRF score isn't a 0-1 cosine — normalize against
            # its own max, same treatment the legacy RRF fallback gets below,
            # so it's comparable to the other three signals in final_score.
            max_hybrid = max(raw_scores.values())
            semantic_by_id = {k: v / max_hybrid for k, v in raw_scores.items()}
            vector_used = True
            hybrid_used = True
            fetched = l1_store.get_facts_by_ids(tenant_id, user_id, list(semantic_by_id))
            facts_by_id = {f["id"]: f for f in fetched}

    if not facts_by_id:
        keyword_hits = l1_store.search_keyword(tenant_id, user_id, query, top_k=20)
        facts_by_id = {f["id"]: f for f in keyword_hits}
        if hybrid_query is None and query_vector is not None:
            for hit in vector_store.search(tenant_id, user_id, query_vector, top_k=20):
                semantic_by_id[hit["id"]] = hit["semantic_score"]
            vector_used = True
            for fact in l1_store.get_facts_by_ids(
                tenant_id, user_id, [i for i in semantic_by_id if i not in facts_by_id]
            ):
                facts_by_id[fact["id"]] = fact

    profile = persona.profile_lines(tenant_id, user_id)

    if not facts_by_id:
        return {
            "context_string": _assemble(hot_turns, [], {}, profile),
            "memories": [], "hot_turns": len(hot_turns),
            "vector_used": vector_used, "cache_hit": False,
        }

    # 3. RRF merge — only for the legacy path. Qdrant already returned one
    # fused ranking in semantic_by_id when hybrid_used, so every fact_id in
    # facts_by_id already has a normalized score and this is never consulted.
    if hybrid_used:
        rrf: dict[str, float] = {}
        max_rrf = 1.0
    else:
        keyword_ranking = [f["id"] for f in keyword_hits]
        vector_ranking = sorted(semantic_by_id, key=semantic_by_id.get, reverse=True)
        rrf = scoring.rrf_merge([keyword_ranking, vector_ranking])
        max_rrf = max(rrf.values()) if rrf else 1.0

    # 4. Four-signal scoring.
    entity_hops = _graph_hops(tenant_id, user_id, query)
    max_access = max((f["access_count"] for f in facts_by_id.values()), default=0)
    now = datetime.now(timezone.utc)

    scored = []
    for fact_id, fact in facts_by_id.items():
        # semantic = real cosine when the vector leg found this fact;
        # otherwise the normalized RRF rank stands in, so an exact BM25
        # keyword hit that the vector leg missed still keeps its keyword
        # relevance instead of being collapsed to 0.0 (rrf is legitimately
        # {} only in the hybrid_used path, where every fact in facts_by_id
        # is already in semantic_by_id and this fallback never fires).
        semantic = semantic_by_id.get(fact_id, rrf.get(fact_id, 0.0) / max_rrf)
        recency = scoring.recency_score(fact["created_at"], now)
        frequency = scoring.frequency_score(fact["access_count"], max_access)
        graph = scoring.graph_score(_fact_graph_hops(fact["content"], entity_hops))
        final = scoring.final_score(semantic, frequency, recency, graph)
        scored.append({**fact, "semantic_score": round(semantic, 4), "recency_score": round(recency, 4),
                       "frequency_score": round(frequency, 4), "graph_score": graph,
                       "score": round(final, 4)})

    scored.sort(key=lambda f: f["score"], reverse=True)
    kept = [f for f in scored if f["score"] >= settings.recall.recall_threshold]
    kept = kept[: settings.recall.max_results]

    # 5. Access frequency: being recalled reinforces future ranking.
    l1_store.bump_access([f["id"] for f in kept])

    # 6. Vault enrichment: 1-hop linked notes for entities the returned
    # memories mention. Fact contents are plain text (wikilinks live in the
    # vault, not the fact store), so entities are extracted here and handed
    # to the walker as synthetic [[links]].
    mentioned = {e for f in kept for e in synthesis.extract_entities_naive(f["content"])}
    pseudo_body = " ".join(f"[[{name}]]" for name in mentioned)
    linked = vault.expand_links(tenant_id, user_id, [pseudo_body], hops=1) if mentioned else {}

    # 7. Temporal history: attached whenever a kept fact was actually
    # superseded — no query-intent gating. A keyword check on the query text
    # ("used to", "changed", ...) sounds like the right gate but doesn't fire
    # on the queries that actually need this: "how did your view on X
    # evolve" is a normal-sounding statement, not a question phrased with
    # any of those words — the AI is expected to *notice* the change
    # unprompted, not answer only when asked. get_predecessor_chains only
    # ever returns entries for facts that were genuinely superseded, so this
    # is naturally zero-cost/zero-noise for the common case where nothing
    # ever changed — no heuristic needed to keep it cheap.
    history = l1_store.get_predecessor_chains(tenant_id, user_id, [f["id"] for f in kept])
    for fact in kept:
        if fact["id"] in history:
            fact["history"] = history[fact["id"]]

    context_string = _assemble(hot_turns, kept, linked, profile)
    for fact in kept:  # cache serializes with json.dumps — datetimes must go
        fact["created_at"] = fact["created_at"].isoformat()

    result = {
        "context_string": context_string,
        "memories": kept, "hot_turns": len(hot_turns),
        "vector_used": vector_used, "cache_hit": False,
    }
    semantic_cache.put(tenant_id, user_id, query, query_vector, result)
    return result


def _age_label(created_at: datetime, now: datetime) -> str:
    days = max(int((now - created_at).total_seconds() // 86400), 0)
    return "today" if days == 0 else ("1 day ago" if days == 1 else f"{days} days ago")


def _assemble(
    hot_turns: list[dict], memories: list[dict], linked: dict[str, str], profile: list[str] | None = None
) -> str:
    """The product: one clean block ready for prompt injection, bounded by
    recall.max_context_chars (lowest-score memories dropped first — the list
    arrives score-descending, so popping the tail sheds the weakest)."""
    now = datetime.now(timezone.utc)

    def build(kept_memories: list[dict]) -> str:
        sections = ["[MEMORY CONTEXT]"]
        if profile:
            sections.append("\nUSER PROFILE:")
            sections.extend(profile)
        if hot_turns:
            sections.append("\nRECENT CONVERSATION:")
            sections.extend(f"- {t['role']}: {t['content']}" for t in hot_turns)
        if kept_memories:
            sections.append("\nRELEVANT MEMORIES:")
            for i, m in enumerate(kept_memories, 1):
                created = m["created_at"]
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                sections.append(
                    f"{i}. [{m['type']}, priority: {m['priority']}, {_age_label(created, now)}] {m['content']}"
                )
                if m.get("history"):
                    chain = " -> ".join([h["content"] for h in m["history"]] + [m["content"]])
                    sections.append(f"   (history: {chain})")
        if linked:
            lines = []
            for name, body in linked.items():
                first = next((ln for ln in body.splitlines() if ln.strip() and not ln.startswith("#")), "")
                if first:
                    lines.append(f"- {name}: {first.strip()}")
            if lines:
                sections.append("\nLINKED NOTES:")
                sections.extend(lines)
        sections.append("\n[END MEMORY CONTEXT]")
        return "\n".join(sections)

    remaining = list(memories)
    text = build(remaining)
    while len(text) > settings.recall.max_context_chars and remaining:
        remaining.pop()
        text = build(remaining)
    return text
