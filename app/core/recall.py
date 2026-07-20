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
    named in the query itself; hops 1-2 = reached via vault wikilinks."""
    hops: dict[str, int] = {}
    query_entities = synthesis.extract_entities_naive(query)
    for name in query_entities:
        hops[name.lower()] = 0

    frontier = [f"[[{e}]]" for e in query_entities]
    for hop in (1, 2):
        expanded = vault.expand_links(tenant_id, user_id, frontier, hops=1)
        frontier = []
        for name, body in expanded.items():
            if name.lower() not in hops:
                hops[name.lower()] = hop
                frontier.append(body)
    return hops


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
        # semantic = real cosine when the vector leg ran; otherwise the
        # normalized RRF stands in so keyword relevance still drives ranking
        semantic = semantic_by_id.get(fact_id, rrf.get(fact_id, 0.0) / max_rrf if not vector_used else 0.0)
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
