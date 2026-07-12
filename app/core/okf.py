"""Open Knowledge Format (OKF): the on-disk memory format.

An OKF note is a plain markdown file that Obsidian/Logseq open natively:

    ---
    id: scene_8f9a2b
    type: scene
    created: 2025-07-08T14:30:00Z
    ...
    ---
    # Scene: ...
    body with [[wikilinks]]

This module is pure format logic — rendering, parsing, wikilinks, slugs.
It never touches the filesystem; that's app.core.vault's job.
"""

import re
import unicodedata

import yaml

_FRONTMATTER_PATTERN = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
# [[Target]] or [[Target|display text]] — capture only the target
_WIKILINK_PATTERN = re.compile(r"\[\[([^\[\]|]+)(?:\|[^\[\]]*)?\]\]")


def render_note(frontmatter: dict, body: str) -> str:
    """Renders a complete OKF note. Frontmatter keys keep insertion order so
    files stay diff-stable when rewritten with unchanged metadata."""
    fm = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{fm}\n---\n\n{body.strip()}\n"


def parse_note(raw: str) -> tuple[dict, str]:
    """Splits a note into (frontmatter dict, body). A file without valid
    frontmatter — e.g. a note the user created by hand in Obsidian — parses
    as ({}, whole file) rather than erroring; hand-edited files are a
    first-class input in this system, never a corruption case."""
    match = _FRONTMATTER_PATTERN.match(raw)
    if not match:
        return {}, raw.strip()
    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
        if not isinstance(frontmatter, dict):
            return {}, raw.strip()
    except yaml.YAMLError:
        return {}, raw.strip()
    return frontmatter, raw[match.end() :].strip()


def extract_wikilinks(body: str) -> list[str]:
    """All [[wikilink]] targets in order of first appearance, deduplicated,
    display-text aliases ([[Target|shown]]) resolved to the target."""
    seen: list[str] = []
    for target in _WIKILINK_PATTERN.findall(body):
        cleaned = target.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def wikify(text: str, entities: list[str]) -> str:
    """Wraps the first occurrence of each entity name in [[slug|Display Name]],
    longest names first so "Project X Alpha" links whole before "Project X"
    can split it. Occurrences already inside a wikilink are left alone.

    Aliased to the slug rather than a bare [[Display Name]] because Obsidian
    resolves wikilinks against filenames — it's case-insensitive but does NOT
    turn spaces into hyphens, so a bare [[Iron Temple]] never resolves to
    entities/iron-temple.md and renders as a phantom unresolved node instead
    of the real one, even though nothing about the underlying data is wrong
    (vault.expand_links/read_note already slugify before comparing). The
    alias form points Obsidian at the real file while still displaying the
    natural name.
    """
    for entity in sorted(entities, key=len, reverse=True):
        pattern = re.compile(
            # not already inside brackets: no [[ immediately before, no ]] after
            r"(?<!\[\[)" + re.escape(entity) + r"(?!\]\])(?![^\[]*\]\])"
        )
        # Obsidian's own link resolution is case-insensitive, so a plain
        # case difference ("Sarah" -> sarah.md) still resolves fine bare.
        # Only alias when slugify changed something case-insensitivity
        # wouldn't cover — spaces, punctuation, accents.
        slug = slugify(entity)
        replacement = f"[[{entity}]]" if slug == entity.lower() else f"[[{slug}|{entity}]]"
        text = pattern.sub(replacement, text, count=1)
    return text


def slugify(title: str) -> str:
    """Filesystem-safe filename stem for a note title. Obsidian matches
    wikilinks to filenames, so this must be deterministic and collision-poor:
    'Project X: The Reckoning!' -> 'project-x-the-reckoning'."""
    normalized = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    lowered = normalized.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "untitled"
