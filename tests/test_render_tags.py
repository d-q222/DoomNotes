"""Rendering, slugs, tag normalisation and consolidation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from doomnotes.consolidate import consolidate, plan_merges, plan_splits
from doomnotes.models import Note
from doomnotes.render import note_slug, render_note, render_transcript, slugify
from doomnotes.tags import TagRegistry, normalise_tag
from doomnotes.vault import VaultWriter


def make_note(**kw) -> Note:
    base = dict(
        title="Postgres partial indexes for soft-deleted rows",
        source_url="https://www.instagram.com/reel/AAA/",
        platform="instagram",
        summary="A short summary of the video.",
        key_points=["point one", "point two"],
        links=["https://example.com/tool"],
        topic="coding",
        tags=["coding", "postgres"],
        author="@someone",
        has_transcript=True,
        source_order=42,
        processed_at=datetime(2026, 8, 9),
    )
    base.update(kw)
    return Note(**base)


# ── slugs ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Postgres partial indexes", "postgres-partial-indexes"),
        ("  Spaces   everywhere  ", "spaces-everywhere"),
        ("Émojis 🌱 and accents", "emojis-and-accents"),
        ("!!!", "untitled"),
    ],
)
def test_slugify(title: str, expected: str) -> None:
    assert slugify(title) == expected


def test_slug_collision_appends_url_hash() -> None:
    n1 = make_note()
    n2 = make_note(source_url="https://www.instagram.com/reel/BBB/")
    taken: set[str] = set()
    s1 = note_slug(n1, taken)
    taken.add(s1)
    s2 = note_slug(n2, taken)
    assert s1 != s2
    assert s2.startswith(s1)


def test_slug_is_stable_across_reruns() -> None:
    """Hash comes from the URL, not a counter, so re-running is not -2, -3, -4."""
    n = make_note()
    taken = {slugify(n.title)}
    assert note_slug(n, taken) == note_slug(n, taken)


# ── markdown ─────────────────────────────────────────────────────────────


def test_render_includes_wikilink_when_transcript_exists() -> None:
    md = render_note(make_note(), transcript_slug="my-slug")
    assert "[[_transcripts/my-slug|Raw transcript]]" in md
    assert "has_transcript: true" in md


def test_caption_only_note_has_no_wikilink_and_flags_false() -> None:
    md = render_note(make_note(has_transcript=False, caption="the caption"), None)
    assert "Raw transcript" not in md
    assert "has_transcript: false" in md
    assert "## Caption" in md


def test_posted_at_absent_on_caption_only_notes() -> None:
    """The known interaction between Tier-1 #2's two halves."""
    md = render_note(make_note(has_transcript=False, posted_at=None), None)
    assert "posted_at:" not in md


def test_frontmatter_quotes_titles_containing_colons() -> None:
    md = render_note(make_note(title='Why "X: Y" breaks YAML'), None)
    fm = md.split("---")[1]
    assert 'title: "Why \\"X: Y\\" breaks YAML"' in fm


def test_tiktok_note_has_saved_at_and_no_author() -> None:
    md = render_note(
        make_note(platform="tiktok", author=None, caption=None,
                  saved_at=datetime(2026, 8, 5)),
        None,
    )
    assert "saved_at: 2026-08-05" in md
    assert "author:" not in md
    assert "## Caption" not in md


def test_transcript_file_links_back() -> None:
    md = render_transcript(make_note(), "the raw text")
    assert "kind: transcript" in md
    assert "the raw text" in md


# ── tags ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("AI Tools", "ai-tools"),
        ("ai-tools / prompts", "ai-tools/prompts"),
        ("Coding/Databases", "coding/databases"),
        ("  spaced  out  ", "spaced-out"),
        ("//weird//", "weird"),
    ],
)
def test_normalise_tag(raw: str, expected: str) -> None:
    assert normalise_tag(raw) == expected


def test_registry_counts_and_roundtrips(tmp_path: Path) -> None:
    reg = TagRegistry()
    reg.observe(["coding", "postgres"])
    reg.observe(["coding"])
    assert reg.counts()["coding"] == 2
    p = tmp_path / "tags.json"
    p.write_text(reg.to_json())
    assert TagRegistry.load(p).counts() == reg.counts()


def test_registry_survives_corrupt_file(tmp_path: Path) -> None:
    p = tmp_path / "tags.json"
    p.write_text("{ not json")
    assert TagRegistry.load(p).counts() == {}


# ── consolidation (pass 2) ───────────────────────────────────────────────


def test_merge_collapses_plurals_and_near_duplicates() -> None:
    merges = plan_merges({"plant": 2, "plants": 9, "gardening": 6, "garden": 1})
    assert merges["plant"] == "plants"
    assert merges.get("garden") == "gardening"


def test_merge_is_deterministic() -> None:
    counts = {"plant": 2, "plants": 9}
    assert plan_merges(counts) == plan_merges(counts)


def test_split_creates_nested_tags_above_threshold() -> None:
    note_tags = [["coding", "databases"]] * 4 + [["coding", "frontend"]] * 3 + [["coding"]] * 9
    counts = {"coding": 16, "databases": 4, "frontend": 3}
    splits = plan_splits(note_tags, counts, threshold=15, min_cluster=3)
    assert splits["coding"]["databases"] == "coding/databases"


def test_split_ignores_tags_below_threshold() -> None:
    note_tags = [["coding", "databases"]] * 4
    assert plan_splits(note_tags, {"coding": 4, "databases": 4}, threshold=15) == {}


def test_consolidate_rewrites_frontmatter_and_is_idempotent(tmp_path: Path) -> None:
    vault = tmp_path / "ai-notes-vault"
    (vault / "_meta").mkdir(parents=True)
    writer = VaultWriter(vault)

    for i in range(3):
        writer.write_text(
            f"n{i}.md",
            render_note(make_note(title=f"Note {i}", tags=["plants", "care"]), None),
        )
    writer.write_text("n3.md", render_note(make_note(title="Note 3", tags=["plant"]), None))

    plan = consolidate(writer, "_meta/tags.json")
    assert plan.notes_touched >= 1
    assert "plant" in plan.merges

    again = consolidate(writer, "_meta/tags.json")
    assert again.notes_touched == 0, "consolidation is not idempotent"


def test_consolidate_writes_through_the_guard(tmp_path: Path) -> None:
    """Registry and rewrites land in the vault, so the guard covers them."""
    vault = tmp_path / "ai-notes-vault"
    (vault / "_meta").mkdir(parents=True)
    writer = VaultWriter(vault)
    writer.write_text("n.md", render_note(make_note(), None))
    consolidate(writer, "_meta/tags.json")
    assert (vault / "_meta" / "tags.json").is_file()
