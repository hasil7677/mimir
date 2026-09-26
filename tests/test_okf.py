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


def test_wikify_survives_an_entity_containing_a_backslash():
    # Regression, 2026-09-25. re.sub with a *string* replacement parses it as
    # a template, so a Windows path in the entity list raised
    # "bad escape \\U" and took the whole flush down with it. Every other
    # fixture in this file is a clean human name, which is why it survived
    # until a real coding session got flushed.
    path = "C:\\Users\\Sahil\\projects"
    text = f"Sahil worked in {path} today."
    result, linked = okf.wikify(text, [path])
    assert linked == [path]
    assert path in result, "the backslashes must survive into the note"
    assert result == f"Sahil worked in [[c-users-sahil-projects|{path}]] today."


def test_wikify_does_not_expand_a_backreference_in_an_entity_name():
    # Same defect, different symptom: the template parser read "\\1" as a
    # group reference. This one raised rather than corrupting, only because
    # the pattern happens to have no capturing groups.
    entity = "Item \\1"
    text = f"The {entity} shipped."
    result, linked = okf.wikify(text, [entity])
    assert linked == [entity]
    assert result == f"The [[item-1|{entity}]] shipped."


def test_wikify_does_not_silently_mangle_a_path_with_an_escape_lookalike():
    # The dangerous half of the defect, and the most likely one in practice.
    # "\\t" and "\\n" ARE valid escapes, so these never raised — the template
    # parser quietly turned them into a real tab and a real newline and wrote
    # the mangled text into the vault note. A crash at least tells you.
    entity = "C:\\temp\\new"
    text = f"Sahil used {entity} today."
    result, linked = okf.wikify(text, [entity])
    assert linked == [entity]
    assert "\t" not in result, "a literal backslash-t must not become a tab"
    assert "\n" not in result, "a literal backslash-n must not become a newline"
    assert result == f"Sahil used [[c-temp-new|{entity}]] today."


def test_slugify():
    assert okf.slugify("Project X: The Reckoning!") == "project-x-the-reckoning"
    assert okf.slugify("Café däy") == "cafe-day"
    assert okf.slugify("!!!") == "untitled"
