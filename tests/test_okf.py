from app.core import okf


def test_render_and_parse_roundtrip():
    fm = {"id": "scene_1", "type": "scene", "tags": ["python", "debugging"]}
    body = "# Scene: debugging\n\nThe user fixed a [[deadlock]] in [[Project X]]."

    raw = okf.render_note(fm, body)
    parsed_fm, parsed_body = okf.parse_note(raw)

    assert parsed_fm == fm
    assert parsed_body == body


def test_parse_note_without_frontmatter_is_not_an_error():
    # a hand-written Obsidian note with no frontmatter must parse cleanly
    fm, body = okf.parse_note("just some plain markdown the user wrote")
    assert fm == {}
    assert body == "just some plain markdown the user wrote"


def test_parse_note_with_broken_yaml_falls_back_to_whole_body():
    raw = "---\n: : not yaml : :\n---\nbody text"
    fm, body = okf.parse_note(raw)
    assert fm == {}
    assert "body text" in body


def test_extract_wikilinks_dedupes_and_resolves_aliases():
    body = "Saw [[Sarah]] at [[Acme Corp|work]]. [[Sarah]] mentioned [[Project X]]."
    assert okf.extract_wikilinks(body) == ["Sarah", "Acme Corp", "Project X"]


def test_wikify_single_word_entity_needs_no_alias():
    # slug == entity text already, so a bare [[Deadlock]] resolves fine as-is
    text = "Fixed a Deadlock today."
    result, linked = okf.wikify(text, ["Deadlock"])
    assert result == "Fixed a [[Deadlock]] today."
    assert linked == ["Deadlock"]


def test_wikify_links_first_occurrence_only():
    text = "Sarah joined Acme. Sarah likes Acme."
    result, linked = okf.wikify(text, ["Sarah", "Acme"])
    assert result == "[[Sarah]] joined [[Acme]]. Sarah likes Acme."
    assert set(linked) == {"Sarah", "Acme"}


def test_wikify_multiword_entity_uses_slug_alias():
    # Obsidian resolves links by filename, is case-insensitive but does NOT
    # turn spaces into hyphens — a bare [[Iron Temple]] would never resolve
    # to entities/iron-temple.md and would render as a phantom unresolved
    # node. The alias form points at the real file, displays the real name.
    text = "Trains at Iron Temple daily."
    result, linked = okf.wikify(text, ["Iron Temple"])
    assert result == "Trains at [[iron-temple|Iron Temple]] daily."
    assert linked == ["Iron Temple"]


def test_wikify_does_not_double_wrap_existing_links():
    text = "Talked to [[Sarah]] about the launch."
    result, linked = okf.wikify(text, ["Sarah"])
    assert result == text
    assert linked == ["Sarah"]


def test_wikify_prefers_longest_entity_name():
    text = "Working on Project X Alpha this week."
    result, linked = okf.wikify(text, ["Project X", "Project X Alpha"])
    assert "[[project-x-alpha|Project X Alpha]]" in result
    assert "[[project-x|Project X]] Alpha" not in result
    assert linked == ["Project X Alpha"]


def test_wikify_does_not_link_entity_absent_from_text():
    # An LLM-named entity that never literally appears in its own summary
    # (paraphrase, case drift) must not be reported as linked — the caller
    # uses this to avoid stubbing an orphaned vault note for it.
    text = "Sarah joined the team."
    result, linked = okf.wikify(text, ["Sarah", "Acme Corp"])
    assert result == "[[Sarah]] joined the team."
    assert linked == ["Sarah"]


def test_slugify():
    assert okf.slugify("Project X: The Reckoning!") == "project-x-the-reckoning"
    assert okf.slugify("Café däy") == "cafe-day"
    assert okf.slugify("!!!") == "untitled"
