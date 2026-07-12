import pytest

from app.config import settings
from app.core import vault


@pytest.fixture(autouse=True)
def _isolated_vault(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.storage.vault, "path", str(tmp_path / "vault"))


def test_write_scene_creates_obsidian_compatible_note_with_wikilinks():
    path = vault.write_scene(
        "t1", "u1",
        scene_id="scene_1", session_id="s1",
        title="Debugging the async scraper",
        body="User fixed a deadlock in Project X using asyncio.Lock.",
        entities=["deadlock", "Project X"],
        source_ids=["m1", "m2"],
    )

    raw = path.read_text(encoding="utf-8")
    assert raw.startswith("---\n")
    assert "[[deadlock]]" in raw  # single word, case-insensitive resolution covers it
    assert "[[project-x|Project X]]" in raw  # multi-word needs the slug alias

    note = vault.read_note("t1", "u1", "Debugging the async scraper")
    assert note is not None
    fm, body = note
    assert fm["id"] == "scene_1"
    assert fm["source_ids"] == ["m1", "m2"]


def test_multiword_entity_wikilink_target_matches_its_own_stub_filename():
    """Regression for a bug caught live via an actual Obsidian screenshot:
    a scene mentioning "Iron Temple" wikified as bare [[Iron Temple]] never
    resolved to entities/iron-temple.md in Obsidian (it normalizes case but
    not spaces-to-hyphens), rendering as a phantom unresolved node duplicate
    of the real, resolved "iron-temple" node. The wikilink target actually
    written into the note must equal the entity stub's own slug.
    """
    from app.core import okf

    path = vault.write_scene(
        "t1", "u1", "scene_1", "s1", "Deadlift day",
        body="Coached by Vikram at Iron Temple.",
        entities=["Vikram", "Iron Temple"], source_ids=[],
    )

    raw = path.read_text(encoding="utf-8")
    targets = okf.extract_wikilinks(raw)

    stub_path = vault.ensure_entity_stub("t1", "u1", "Iron Temple")
    assert okf.slugify(stub_path.stem) in targets, "wikilink target must match the real file's slug"
    assert "Iron Temple" in raw, "the human-readable name must still be visible as display text"


def test_write_scene_creates_entity_stubs_so_no_dangling_links():
    vault.write_scene(
        "t1", "u1", "scene_1", "s1", "A scene",
        body="Met Sarah at Acme.", entities=["Sarah", "Acme"], source_ids=[],
    )
    assert vault.read_note("t1", "u1", "Sarah") is not None
    assert vault.read_note("t1", "u1", "Acme") is not None


def test_entity_stub_never_overwrites_user_edits():
    stub = vault.ensure_entity_stub("t1", "u1", "Sarah")
    stub.write_text("# Sarah\n\nMy sister. Lives in Berlin.", encoding="utf-8")

    vault.ensure_entity_stub("t1", "u1", "Sarah")  # second call must be a no-op

    _, body = vault.read_note("t1", "u1", "Sarah")
    assert "Lives in Berlin" in body


def test_persona_upsert_keeps_timestamped_history():
    vault.upsert_persona("t1", "u1", "# Persona v1")
    vault.upsert_persona("t1", "u1", "# Persona v2")

    _, current = vault.read_note("t1", "u1", "persona")
    assert "v2" in current

    notes = vault.list_notes("t1", "u1")
    assert any(n.startswith("persona_history") for n in notes), "old persona must be backed up, not lost"


def test_vaults_isolated_per_tenant_and_user():
    vault.write_scene("t1", "u1", "sc1", "s1", "T1 scene", "tenant one fact", [], [])
    vault.write_scene("t2", "u1", "sc2", "s1", "T2 scene", "tenant two fact", [], [])

    assert vault.read_note("t1", "u1", "T2 scene") is None
    assert vault.read_note("t2", "u1", "T1 scene") is None


def test_expand_links_walks_two_hops():
    # scene -> [[Project X]] -> [[Acme]]  (hop 1 gets Project X, hop 2 gets Acme)
    project = vault.ensure_entity_stub("t1", "u1", "Project X")
    project.write_text("# Project X\n\nThe scraper rebuild for [[Acme]].", encoding="utf-8")
    acme = vault.ensure_entity_stub("t1", "u1", "Acme")
    acme.write_text("# Acme\n\nOur biggest client, since 2024.", encoding="utf-8")

    scene_body = "User shipped a fix for [[Project X]] today."
    expanded = vault.expand_links("t1", "u1", [scene_body], hops=2)

    assert "Project X" in expanded
    assert "Acme" in expanded, "2-hop walk should follow Project X's own link to Acme"
    assert "since 2024" in expanded["Acme"]


def test_expand_links_one_hop_does_not_reach_second_level():
    project = vault.ensure_entity_stub("t1", "u1", "Project X")
    project.write_text("# Project X\n\nBuilt for [[Acme]].", encoding="utf-8")
    vault.ensure_entity_stub("t1", "u1", "Acme")

    expanded = vault.expand_links("t1", "u1", ["About [[Project X]]."], hops=1)

    assert "Project X" in expanded
    assert "Acme" not in expanded


def test_expand_links_handles_cycles_without_hanging():
    a = vault.ensure_entity_stub("t1", "u1", "Alpha")
    a.write_text("# Alpha\n\nSee [[Beta]].", encoding="utf-8")
    b = vault.ensure_entity_stub("t1", "u1", "Beta")
    b.write_text("# Beta\n\nSee [[Alpha]].", encoding="utf-8")

    expanded = vault.expand_links("t1", "u1", ["Start at [[Alpha]]."], hops=5)

    assert set(expanded) == {"Alpha", "Beta"}


def test_expand_links_ignores_dangling_targets():
    expanded = vault.expand_links("t1", "u1", ["Mentions [[Never Written]]."], hops=2)
    assert expanded == {}
