"""Turning raw session turns into an OKF scene note.

Two paths, one contract:
  - LLM path: proper prose synthesis (any OpenAI-compatible endpoint, incl. Ollama).
  - Offline digest path: deterministic, no network, no models — because the
    local-first principle says capture must never be blocked on an LLM being
    reachable. A digest scene is honest about what it is (frontmatter-visible
    `synthesis: digest`) and can be re-synthesized later when an LLM exists.
"""

import json
import re
from functools import lru_cache

from app.core.llm import LlmUnavailable, chat

# Words that start sentences and look like names but aren't entities.
_STOPWORDS = {
    "The", "This", "That", "These", "Those", "There", "Then", "They",
    "A", "An", "And", "But", "Or", "So", "If", "When", "While", "After",
    "I", "It", "He", "She", "We", "You", "My", "Our", "His", "Her",
    "What", "Which", "Who", "How", "Why", "Where", "Yes", "No", "Not",
    "User", "Assistant", "Also", "Just", "Now", "Today", "Yesterday",
    # Conversational filler and discourse markers — capitalized only
    # because they open a sentence, never because they name anything.
    "Sure", "Okay", "Ok", "Great", "Awesome", "Perfect", "Thanks", "Thank",
    "Please", "Right", "Well", "Actually", "Basically", "Honestly",
    "Certainly", "Absolutely", "Definitely", "Exactly", "Alright", "Cool",
    "Nice", "Wow", "Oh", "Hmm", "Hey", "Hi", "Hello", "Sorry", "Maybe",
    "Perhaps", "Probably", "Anyway", "Anyways", "Meanwhile", "However",
    "Therefore", "Overall", "Generally", "Currently", "Recently",
    "Previously", "Additionally", "Finally", "First", "Second", "Third",
    "Next", "Last", "Here", "Everything", "Something", "Nothing",
    "Someone", "Everyone", "Anybody", "Nobody", "Let", "Look", "Listen",
    "See", "Note", "Remember", "Alrighty",
}

_CAPITALIZED_RUN = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b")

_SCENE_PROMPT = """You are turning a conversation into a memory note.

Conversation:
{transcript}

Return ONLY a JSON object:
{{"title": "short scene title", "summary": "2-5 sentence factual summary in markdown",
  "entities": ["proper nouns / named things worth linking"]}}
"""


_SENTENCE_START = re.compile(r"(?:\A|[.!?]\s+|\n\s*)$")

# Entity types spaCy tags that are numeric/temporal, not "things worth
# linking" in the vault graph sense (a QUANTITY like "110kg" shouldn't get
# its own wikilinked note).
_NUMERIC_ENTITY_LABELS = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "DATE", "TIME"}


@lru_cache(maxsize=1)
def _spacy_nlp():
    # Imported lazily so a missing spaCy/model install never breaks the
    # offline path, same degrade contract as embeddings.py's fastembed.
    # Only tok2vec+ner loaded (tagger/parser/lemmatizer unused here):
    # ~20% faster per call, same entity output, and this runs on every
    # recall query, not just at ingestion.
    import spacy

    return spacy.load("en_core_web_sm", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])


def extract_entities_naive(text: str) -> list[str]:
    """Entity extractor: spaCy NER when the model is installed (optional
    extra, `pip install "mimir-engine[ner]"`), falling back to a regex
    capitalized-run heuristic when it isn't. Deliberately conservative
    either way: a missed entity is a missing wikilink (harmless), a false
    one is a junk note in the user's vault."""
    try:
        nlp = _spacy_nlp()
    except (ImportError, OSError):
        return _extract_entities_regex(text)

    entities: list[str] = []
    for ent in nlp(text).ents:
        if ent.label_ in _NUMERIC_ENTITY_LABELS:
            continue
        # A sentence-final entity ("...at Acme Corp.") absorbs the period
        # into the span since there's no trailing space to split on.
        cleaned = ent.text.strip().rstrip(".,;:!?")
        # The small model occasionally tags a lowercase word as an entity
        # ("quantum computing" -> ORG "quantum"); these are never proper
        # nouns, so require at least one capital letter.
        if cleaned and cleaned != cleaned.lower() and cleaned not in entities:
            entities.append(cleaned)
    return entities


def _extract_entities_regex(text: str) -> list[str]:
    """Runs of Capitalized Words, minus stopwords, minus the sentence-start
    trap: a lone capitalized word opening a sentence ("Huge! ...",
    "Smooth. ...") is almost always just English, so it only counts if the
    same word also shows up capitalized mid-sentence somewhere."""
    mid_sentence_words: set[str] = set()
    candidates: list[tuple[str, bool]] = []  # (run, at_sentence_start)

    for match in _CAPITALIZED_RUN.finditer(text):
        at_start = bool(_SENTENCE_START.search(text[: match.start()]))
        candidates.append((match.group(1), at_start))
        if not at_start:
            mid_sentence_words.update(match.group(1).split())

    entities: list[str] = []
    for run, at_start in candidates:
        words = [w for w in run.split() if w not in _STOPWORDS]
        if not words:
            continue
        if at_start and len(words) == 1 and words[0] not in mid_sentence_words:
            continue
        cleaned = " ".join(words)
        if len(cleaned) >= 3 and any(len(w) > 2 for w in words) and cleaned not in entities:
            entities.append(cleaned)
    return entities


def _digest_scene(turns: list[dict]) -> tuple[str, str, list[str]]:
    first_user = next((t for t in turns if t.get("role") == "user"), turns[0])
    title = first_user["content"].strip().splitlines()[0][:60]

    lines = [f"- **{t.get('role', 'user')}**: {t['content'].strip()}" for t in turns]
    body = "## Transcript digest\n\n" + "\n".join(lines)

    entities: list[str] = []
    for t in turns:
        for e in extract_entities_naive(t["content"]):
            if e not in entities:
                entities.append(e)
    return title, body, entities


def synthesize_scene(turns: list[dict]) -> tuple[str, str, list[str], str]:
    """-> (title, body, entities, mode) where mode is 'llm' or 'digest'."""
    transcript = "\n".join(f"{t.get('role', 'user')}: {t['content']}" for t in turns)
    try:
        raw = chat(_SCENE_PROMPT.format(transcript=transcript), max_tokens=600)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        title = str(data["title"]).strip()[:80]
        body = str(data["summary"]).strip()
        entities = [str(e).strip() for e in data.get("entities", []) if str(e).strip()]
        return title, body, entities, "llm"
    except (LlmUnavailable, json.JSONDecodeError, KeyError, AttributeError):
        title, body, entities = _digest_scene(turns)
        return title, body, entities, "digest"
