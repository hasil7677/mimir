from unittest.mock import patch

from app.core import synthesis
from app.core.llm import LlmUnavailable


def test_entities_uses_spacy_when_available():
    text = "The user said Sarah works at Acme Corp."
    entities = synthesis.extract_entities_naive(text)
    assert "Sarah" in entities
    assert "Acme Corp" in entities
    assert "The" not in entities


def test_entities_drops_numeric_and_temporal_labels():
    # A quantity like "110kg" is real spaCy NER output but isn't a "thing
    # worth linking" in the vault graph sense.
    text = "Next goal is 110kg, coach Vikram approves."
    entities = synthesis.extract_entities_naive(text)
    assert "110kg" not in entities
    assert "Vikram" in entities


def test_entities_keeps_multiword_spans():
    text = "Iron Temple is where I train now."
    assert "Iron Temple" in synthesis.extract_entities_naive(text)


def test_entities_falls_back_to_regex_when_spacy_unavailable():
    with patch("app.core.synthesis._spacy_nlp", side_effect=ImportError("no spacy")):
        entities = synthesis.extract_entities_naive(
            "Huge! How did it feel? Smooth. Next goal is 110kg, coach Vikram approves."
        )
    assert "Huge" not in entities
    assert "Smooth" not in entities
    assert "Next" not in entities
    assert "Vikram" in entities


def test_regex_fallback_finds_proper_nouns_and_skips_stopwords():
    text = "The user said Sarah works at Acme Corp on Project X."
    entities = synthesis._extract_entities_regex(text)
    assert "Sarah" in entities
    assert "Acme Corp" in entities
    assert "Project X" in entities
    assert "The" not in entities


def test_regex_fallback_drops_lone_sentence_starters():
    # "Huge!" / "Smooth." are English, not entities — the exact junk the
    # first live demo wrote into the vault as [[Huge]] and [[Smooth]]
    text = "Huge! How did it feel? Smooth. Next goal is 110kg, coach Vikram approves."
    entities = synthesis._extract_entities_regex(text)
    assert "Huge" not in entities
    assert "Smooth" not in entities
    assert "Next" not in entities
    assert "Vikram" in entities


def test_regex_fallback_keeps_sentence_starter_seen_mid_sentence_too():
    text = "Vikram said the plan works. I trust Vikram on this."
    entities = synthesis._extract_entities_regex(text)
    assert "Vikram" in entities


def test_regex_fallback_keeps_multiword_runs_at_sentence_start():
    text = "Iron Temple is where I train now."
    assert "Iron Temple" in synthesis._extract_entities_regex(text)


def test_digest_fallback_when_llm_unreachable():
    turns = [
        {"role": "user", "content": "I fixed the deadlock in Project X today"},
        {"role": "assistant", "content": "Nice, was it the asyncio issue?"},
    ]
    with patch("app.core.synthesis.chat", side_effect=LlmUnavailable("down")):
        title, body, entities, mode = synthesis.synthesize_scene(turns)

    assert mode == "digest"
    assert title == "I fixed the deadlock in Project X today"
    assert "**user**" in body and "**assistant**" in body
    assert "Project X" in entities


def test_llm_path_parses_json_response():
    llm_reply = '{"title": "Deadlock fix", "summary": "User fixed a deadlock.", "entities": ["Project X"]}'
    with patch("app.core.synthesis.chat", return_value=llm_reply):
        title, body, entities, mode = synthesis.synthesize_scene([{"role": "user", "content": "x"}])

    assert mode == "llm"
    assert title == "Deadlock fix"
    assert entities == ["Project X"]


def test_llm_garbage_output_falls_back_to_digest():
    with patch("app.core.synthesis.chat", return_value="Sure! Here is a summary of..."):
        _, _, _, mode = synthesis.synthesize_scene([{"role": "user", "content": "hello there"}])
    assert mode == "digest"
