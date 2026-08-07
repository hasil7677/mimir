"""The vault: the agent's memory as a directory of OKF markdown files.

Layout (one vault root, per-tenant/per-user subtrees so a shared deployment
still keeps every user's brain in its own openable folder):

    {vault_root}/{tenant_id}/{user_id}/
        scenes/{scene_slug}.md
        entities/{entity_slug}.md
        persona.md

The vault is the ground truth. Databases index it; they never own it. That's
why reads always come from disk — if the user edits a note in Obsidian, the
next read sees their edit with no sync step.
"""

from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.core import okf


def _user_root(tenant_id: str, user_id: str) -> Path:
    root = Path(settings.storage.vault.path).expanduser() / tenant_id / user_id
    return root


def _note_index(tenant_id: str, user_id: str) -> dict[str, Path]:
    """slug -> path for every note in the user's vault. Wikilink targets
    resolve through this, so a [[link]] finds its note wherever it lives."""
    root = _user_root(tenant_id, user_id)
    if not root.exists():
        return {}
    return {p.stem: p for p in root.rglob("*.md")}


def write_scene(
    tenant_id: str,
    user_id: str,
    scene_id: str,
    session_id: str,
    title: str,
    body: str,
    entities: list[str],
    source_ids: list[str],
) -> Path:
    """Writes an L2 scene note. Entity names in the body become [[wikilinks]],
    and each linked entity gets a stub note if none exists yet — so the
    Obsidian graph never shows dangling links.

    Only entities that actually got wikified in the body are stubbed. The
    caller's `entities` list can include names an LLM synthesis step named
    but never literally wrote into `body` (paraphrase, case drift); stubbing
    those anyway would create a vault note with no backlink to it at all —
    an orphaned single node, not just a sparsely-connected one.
    """
    root = _user_root(tenant_id, user_id)
    scenes_dir = root / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    linked_body, linked_entities = okf.wikify(body, entities)
    frontmatter = {
        "id": scene_id,
        "type": "scene",
        "created": datetime.now(timezone.utc).isoformat(),
        "session": session_id,
        "source_ids": source_ids,
    }
    note_path = scenes_dir / f"{okf.slugify(title)}.md"
    note_path.write_text(okf.render_note(frontmatter, f"# {title}\n\n{linked_body}"), encoding="utf-8")

    scene_slug = note_path.stem
    for entity in linked_entities:
        entity_path = ensure_entity_stub(tenant_id, user_id, entity)
        _add_scene_backlink(entity_path, scene_slug, title)
    return note_path


def ensure_entity_stub(tenant_id: str, user_id: str, entity_name: str) -> Path:
    """Creates entities/{slug}.md if missing. Never overwrites an existing
    note — the user may have enriched it by hand, and their version wins."""
    entities_dir = _user_root(tenant_id, user_id) / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    path = entities_dir / f"{okf.slugify(entity_name)}.md"
    if not path.exists():
        frontmatter = {
            "type": "entity",
            "created": datetime.now(timezone.utc).isoformat(),
            "aliases": [],
        }
        path.write_text(okf.render_note(frontmatter, f"# {entity_name}\n"), encoding="utf-8")
    return path


def _add_scene_backlink(path: Path, scene_slug: str, scene_title: str) -> None:
    """Appends a backlink to the given scene under a `Mentioned in scenes:`
    line — additive-only, idempotent (safe to call every time the same
    entity gets linked again), never touches anything else in the note.

    Without this, entity notes are permanent dead ends: no outgoing link
    means graph_hops has nothing to walk past hop 0, and recall's LINKED
    NOTES section (which looks for the first non-heading line in a linked
    note) never has anything to show. This is what actually makes the vault
    a graph instead of a pile of disconnected single-line stubs.
    """
    link = f"[[{scene_slug}|{scene_title}]]"
    frontmatter, body = okf.parse_note(path.read_text(encoding="utf-8"))
    if link in body:
        return
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Mentioned in scenes:"):
            lines[i] = f"{line.rstrip()}, {link}"
            path.write_text(okf.render_note(frontmatter, "\n".join(lines)), encoding="utf-8")
            return
    body = body.rstrip() + f"\n\nMentioned in scenes: {link}\n"
    path.write_text(okf.render_note(frontmatter, body), encoding="utf-8")


def upsert_persona(tenant_id: str, user_id: str, body: str, extra_frontmatter: dict | None = None) -> Path:
    """Writes persona.md, keeping a timestamped backup of the previous version
    — persona synthesis is lossy by nature, so the old document is history
    worth keeping, same supersede-not-overwrite stance as everywhere else."""
    root = _user_root(tenant_id, user_id)
    root.mkdir(parents=True, exist_ok=True)
    persona_path = root / "persona.md"

    if persona_path.exists():
        backups = root / "persona_history"
        backups.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        (backups / f"persona-{stamp}.md").write_text(persona_path.read_text(encoding="utf-8"), encoding="utf-8")

    frontmatter = {"type": "persona", "updated": datetime.now(timezone.utc).isoformat(), **(extra_frontmatter or {})}
    persona_path.write_text(okf.render_note(frontmatter, body), encoding="utf-8")
    return persona_path


def read_note(tenant_id: str, user_id: str, target: str) -> tuple[dict, str] | None:
    """Reads a note by wikilink target (title or slug), fresh from disk."""
    path = _note_index(tenant_id, user_id).get(okf.slugify(target))
    if path is None:
        return None
    return okf.parse_note(path.read_text(encoding="utf-8"))


def list_notes(tenant_id: str, user_id: str) -> list[str]:
    """Relative note paths, always forward-slash (as_posix) so the API shape
    is identical on Windows and Unix and matches Obsidian's own link style."""
    root = _user_root(tenant_id, user_id)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.md")) if root.exists() else []


def erase_user(tenant_id: str, user_id: str) -> int:
    """Deletes the user's entire vault subtree. Returns notes removed."""
    import shutil

    root = _user_root(tenant_id, user_id)
    if not root.exists():
        return 0
    count = sum(1 for _ in root.rglob("*.md"))
    shutil.rmtree(root)
    return count


def export_user(tenant_id: str, user_id: str) -> dict[str, str]:
    """Every note in the user's vault: relative posix path -> raw content."""
    root = _user_root(tenant_id, user_id)
    if not root.exists():
        return {}
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in root.rglob("*.md")}


def flushed_session_ids(tenant_id: str, user_id: str) -> set[str]:
    """Every session_id that already has a scene note — the vault's own
    record of what's been distilled, since write_scene always stamps
    `session:` in frontmatter. Used to recover sessions an MCP adapter
    captured but never got to flush (e.g. the process was recycled
    mid-conversation) without needing a separate tracking table."""
    root_by_scenes = _user_root(tenant_id, user_id) / "scenes"
    if not root_by_scenes.exists():
        return set()
    sessions = set()
    for path in root_by_scenes.glob("*.md"):
        frontmatter, _ = okf.parse_note(path.read_text(encoding="utf-8"))
        session = frontmatter.get("session")
        if session:
            sessions.add(session)
    return sessions


def expand_links(tenant_id: str, user_id: str, bodies: list[str], hops: int = 2) -> dict[str, str]:
    """The enrichment walk: collect [[wikilinks]] from the given note bodies,
    read each linked note, then follow *their* links, up to `hops` levels.
    Returns {link target -> note body} for every reachable note — the same
    associative pull a human gets clicking through their Obsidian graph.
    """
    index = _note_index(tenant_id, user_id)
    collected: dict[str, str] = {}
    frontier = [t for body in bodies for t in okf.extract_wikilinks(body)]

    for _ in range(hops):
        next_frontier: list[str] = []
        for target in frontier:
            slug = okf.slugify(target)
            if target in collected or slug not in index:
                continue
            _, note_body = okf.parse_note(index[slug].read_text(encoding="utf-8"))
            collected[target] = note_body
            next_frontier.extend(okf.extract_wikilinks(note_body))
        frontier = next_frontier

    return collected
