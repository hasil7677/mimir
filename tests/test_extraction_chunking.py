"""Long sessions are split into overlapping windows before extraction; short
ones keep the exact single-call path. Failure of any window falls back to the
whole-session verbatim path, same as an outage on the single-call path."""

from unittest.mock import patch

from app.config import settings
from app.core import extraction
from app.core.llm import LlmUnavailable


def _turns(n: int, size: int) -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn{i:03d} " + "x" * size}
        for i in range(n)
    ]


def _fact_reply(label: str) -> str:
    return (
        '[{"content": "User fact %s", "type": "persona", "priority": 90, "scene_name": "s"}]' % label
    )


def test_short_session_takes_a_single_call():
    turns = _turns(6, 100)
    with patch("app.core.extraction.chat", return_value=_fact_reply("a")) as chat:
        facts, mode = extraction.extract_facts(turns)

    assert chat.call_count == 1
    assert mode == "llm"
    assert len(facts) == 1


def test_chunk_turns_respects_threshold_and_overlaps_one_turn():
    turns = _turns(40, 1000)  # ~40k chars, well past the 16k threshold
    chunks = extraction._chunk_turns(turns)

    assert len(chunks) > 1
    # every turn appears, in order, with no turn split or altered
    seen = []
    for chunk in chunks:
        for t in chunk:
            if not seen or seen[-1] is not t:
                seen.append(t)
    assert [t["content"] for t in seen] == [t["content"] for t in turns]
    # consecutive windows share exactly the last turn of the previous one
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt[0] is prev[-1]
    # a window may exceed the threshold by at most the overlap turn it started with
    for chunk in chunks:
        assert sum(len(t["content"]) for t in chunk) <= extraction._CHUNK_CHAR_THRESHOLD + 1100


def test_single_oversize_turn_is_never_split():
    big = {"role": "user", "content": "y" * (extraction._CHUNK_CHAR_THRESHOLD * 2)}
    chunks = extraction._chunk_turns([big, {"role": "assistant", "content": "ok then"}])

    assert any(big in chunk for chunk in chunks)
    assert all(t["content"] == big["content"] for chunk in chunks for t in chunk if t is big)


def test_long_session_makes_one_call_per_window_and_merges_facts():
    turns = _turns(40, 1000)
    expected = len(extraction._chunk_turns(turns))
    replies = iter(_fact_reply(str(i)) for i in range(expected))
    with patch("app.core.extraction.chat", side_effect=lambda *a, **k: next(replies)) as chat:
        facts, mode = extraction.extract_facts(turns)

    assert chat.call_count == expected
    assert mode == "llm"
    assert [f["content"] for f in facts] == [f"User fact {i}" for i in range(expected)]


def test_failure_in_any_window_falls_back_to_whole_session_verbatim():
    turns = _turns(40, 1000)
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise LlmUnavailable("down mid-session")
        return _fact_reply("ok")

    with patch("app.core.extraction.chat", side_effect=flaky):
        facts, mode = extraction.extract_facts(turns)

    assert mode == "verbatim"
    assert facts  # verbatim path still returns the user turns
    assert all(f["extraction"] == "verbatim" for f in facts)


def test_merged_facts_are_capped_per_session(monkeypatch):
    monkeypatch.setattr(settings.extraction, "max_memories_per_session", 2)
    turns = _turns(40, 1000)
    with patch("app.core.extraction.chat", return_value=_fact_reply("z")):
        facts, mode = extraction.extract_facts(turns)

    assert mode == "llm"
    assert len(facts) == 2
