from datetime import datetime, timedelta, timezone

from app.config import settings
from app.core import scoring


def test_rrf_doc_in_both_lists_beats_doc_in_one():
    merged = scoring.rrf_merge([["a", "b"], ["a", "c"]])
    assert merged["a"] > merged["b"]
    assert merged["a"] > merged["c"]


def test_rrf_rank_order_preserved_within_a_list():
    merged = scoring.rrf_merge([["first", "second", "third"]])
    assert merged["first"] > merged["second"] > merged["third"]


def test_recency_halves_around_fourteen_days_at_default_rate():
    now = datetime.now(timezone.utc)
    score_14d = scoring.recency_score(now - timedelta(days=14), now)
    assert 0.45 < score_14d < 0.55
    assert scoring.recency_score(now, now) > 0.99


def test_frequency_is_log_normalized_to_unit_range():
    assert scoring.frequency_score(0, 10) == 0.0
    assert scoring.frequency_score(10, 10) == 1.0
    assert 0.0 < scoring.frequency_score(3, 10) < 1.0


def test_graph_score_hop_mapping():
    assert scoring.graph_score(0) == 1.0
    assert scoring.graph_score(1) == 0.5
    assert scoring.graph_score(2) == 0.25
    assert scoring.graph_score(None) == 0.0


def test_final_score_respects_config_weights(monkeypatch):
    monkeypatch.setattr(settings.recall.weights, "semantic", 1.0)
    monkeypatch.setattr(settings.recall.weights, "frequency", 0.0)
    monkeypatch.setattr(settings.recall.weights, "recency", 0.0)
    monkeypatch.setattr(settings.recall.weights, "graph", 0.0)
    assert scoring.final_score(0.8, 0.5, 0.5, 0.5) == 0.8
