"""RRF merge + the 4-signal recall scoring formula (spec section 8, step 5).

score(m) = alpha*semantic + beta*frequency + gamma*recency + delta*graph
Weights come from config (recall.weights) and are tunable per deployment.
"""

import math
from datetime import datetime, timezone

from app.config import settings

RRF_K = 60


def rrf_merge(ranked_lists: list[list[str]]) -> dict[str, float]:
    """Reciprocal Rank Fusion: id -> summed 1/(k + rank) across every list it
    appears in. Scale-free, so BM25 scores and cosine similarities never need
    to be made comparable — only their rankings matter."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores


def recency_score(created_at: datetime, now: datetime | None = None) -> float:
    """e^(-decay_rate * age_days); today ~1.0, ~14 days ago ~0.5 at the 0.05 default."""
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    age_days = max((now - created_at).total_seconds() / 86400, 0.0)
    return math.exp(-settings.recall.decay_rate * age_days)


def frequency_score(access_count: int, max_access_count: int) -> float:
    """log-normalized so one obsessively-recalled memory doesn't flatten the
    signal for everything else."""
    if max_access_count <= 0 or access_count <= 0:
        return 0.0
    return math.log(1 + access_count) / math.log(1 + max_access_count)


def graph_score(hops: int | None) -> float:
    """Vault-link proximity: direct entity match 1.0, 1-hop 0.5, 2-hop 0.25."""
    if hops is None:
        return 0.0
    return {0: 1.0, 1: 0.5, 2: 0.25}.get(hops, 0.0)


def final_score(semantic: float, frequency: float, recency: float, graph: float) -> float:
    w = settings.recall.weights
    return semantic * w.semantic + frequency * w.frequency + recency * w.recency + graph * w.graph
