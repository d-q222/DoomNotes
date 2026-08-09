"""Rendering, slugs, tag normalisation and consolidation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from doomnotes.consolidate import consolidate, plan_merges, plan_splits
from doomnotes.models import Note
from doomnotes.render import SlugIndex, note_slug, url_hash, render_note, render_transcript, slugify
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


# ── slug ownership across runs ───────────────────────────────────────────
#
# `note_slug` alone cannot decide a collision: it sees a name, not who holds
# it. SlugIndex adds the owner, which is the difference between "another video
# wants this name" (suffix it, keep both) and "this video already owns it"
# (reuse it, overwrite in place). Runs are separate processes days apart, so
# the index is rebuilt from the vault rather than accumulated in memory.


def _vault_with(tmp_path: Path, notes: dict[str, str]) -> Path:
    root = tmp_path / "ai-notes-vault"
    root.mkdir(exist_ok=True)
    for slug, url in notes.items():
        (root / f"{slug}.md").write_text(
            f'---\ntitle: "t"\nsource_url: {url}\nplatform: instagram\n---\n\nbody\n',
            encoding="utf-8",
        )
    return root


def test_index_reads_ownership_from_existing_notes(tmp_path: Path) -> None:
    root = _vault_with(tmp_path, {"a-note": "https://www.instagram.com/reel/AAA/"})
    index = SlugIndex.from_vault(root)
    assert index.owner_of("a-note") == "https://www.instagram.com/reel/AAA/"
    assert index.owner_of("never-written") is None


def test_a_different_video_wanting_a_taken_name_gets_a_suffix(tmp_path: Path) -> None:
    root = _vault_with(tmp_path, {"shared-title": "https://www.instagram.com/reel/AAA/"})
    index = SlugIndex.from_vault(root)
    slug = index.claim(make_note(title="Shared title", source_url="https://www.instagram.com/reel/BBB/"))
    assert slug != "shared-title"
    assert slug.startswith("shared-title-")


def test_the_same_video_reclaims_its_own_name(tmp_path: Path) -> None:
    """Otherwise a re-run accumulates a hash-suffixed duplicate every time."""
    url = "https://www.instagram.com/reel/AAA/"
    root = _vault_with(tmp_path, {"shared-title": url})
    index = SlugIndex.from_vault(root)
    assert index.claim(make_note(title="Shared title", source_url=url)) == "shared-title"


def test_claiming_is_recorded_within_a_run_too(tmp_path: Path) -> None:
    """The in-memory half still has to work; the vault seed only adds to it."""
    index = SlugIndex.from_vault(_vault_with(tmp_path, {}))
    first = index.claim(make_note(title="Same", source_url="https://www.instagram.com/reel/AAA/"))
    second = index.claim(make_note(title="Same", source_url="https://www.instagram.com/reel/BBB/"))
    assert first != second


def test_a_note_with_no_readable_source_url_still_reserves_its_name(tmp_path: Path) -> None:
    """It is someone's note. Overwriting it would lose something.

    Hand-written notes in the vault have no `source_url`, and the conservative
    reading of an unknown owner is "not mine".
    """
    root = tmp_path / "ai-notes-vault"
    root.mkdir()
    (root / "hand-written.md").write_text("# just a note I wrote\n", encoding="utf-8")
    index = SlugIndex.from_vault(root)
    slug = index.claim(make_note(title="Hand written", source_url="https://www.instagram.com/reel/AAA/"))
    assert slug != "hand-written"


def test_a_source_url_in_the_body_is_not_read_as_ownership(tmp_path: Path) -> None:
    """A note quoting `source_url:` in its body must not claim that URL."""
    root = tmp_path / "ai-notes-vault"
    root.mkdir()
    (root / "quoting.md").write_text(
        '---\ntitle: "t"\nplatform: instagram\n---\n\n'
        "    source_url: https://www.instagram.com/reel/AAA/\n",
        encoding="utf-8",
    )
    index = SlugIndex.from_vault(root)
    assert index.owner_of("quoting") == ""


def test_transcripts_reserve_their_slug_too(tmp_path: Path) -> None:
    """Note and transcript share a slug, so a free note name is not enough."""
    root = tmp_path / "ai-notes-vault"
    (root / "_transcripts").mkdir(parents=True)
    (root / "_transcripts" / "orphaned.md").write_text("raw asr\n", encoding="utf-8")
    index = SlugIndex.from_vault(root, subdirs=("_transcripts",))
    assert "orphaned" in index


def test_a_symlinked_note_cannot_claim_a_url_from_outside_the_vault(tmp_path: Path) -> None:
    """Resolve-then-verify on read, same order the write guard uses.

    Reading through a symlink is far less serious than writing through one —
    the result only feeds an ownership comparison and never reaches output or
    a filesystem path. But an `evil.md` symlinked at some file outside the
    vault could otherwise claim a source_url it does not own, and the
    conservative answer costs nothing.

    "" rather than None is the safe direction: the slug stays reserved, so the
    worst outcome is a hash suffix that was not strictly needed.
    """
    root = tmp_path / "ai-notes-vault"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text(
        '---\ntitle: "t"\nsource_url: https://elsewhere.invalid/x/\n---\n\nbody\n',
        encoding="utf-8",
    )
    (root / "evil.md").symlink_to(outside)

    index = SlugIndex.from_vault(root)
    assert index.owner_of("evil") == ""
    assert "evil" in index, "the slug is still reserved — refusing to read is not permission to clobber"


def test_titles_that_differ_only_past_the_slug_truncation_still_get_separate_files() -> None:
    """Slugs are cut at 80 characters, so two titles can collide after cutting.

    Not hypothetical: a descriptive title capped at 90 characters can easily
    share its first 80 with another, and "part one"/"part two" is exactly the
    shape short-form creators use.
    """
    prefix = "Ten ways to use partial indexes in postgres for soft deleted rows in production"
    a, b = prefix + " part one", prefix + " part two"
    assert slugify(a) == slugify(b), "the premise: these collide after truncation"

    index = SlugIndex()
    first = index.claim(make_note(title=a, source_url="https://www.instagram.com/reel/AAA/"))
    second = index.claim(make_note(title=b, source_url="https://www.instagram.com/reel/BBB/"))
    assert first != second


def test_a_third_video_with_the_same_title_does_not_reuse_the_second_ones_name() -> None:
    """Each suffix is that video's own URL hash, so N videos yield N filenames."""
    index = SlugIndex()
    slugs = [
        index.claim(make_note(title="Same title", source_url=f"https://www.instagram.com/reel/{c}/"))
        for c in ("AAA", "BBB", "CCC", "DDD")
    ]
    assert len(set(slugs)) == 4, slugs


def test_a_hash_suffixed_name_is_checked_too_not_assumed_free() -> None:
    """A six-hex suffix is short enough to be someone else's actual title.

    `<base>-<6 hex>` is a perfectly ordinary slug for a note titled e.g.
    "Foo 0467e6", so appending a hash and writing without looking would
    overwrite a real note — the exact failure this class exists to prevent.
    Widening is deterministic, so the URL still resolves to one stable name.
    """
    url_b = "https://www.instagram.com/reel/BBB/"
    index = SlugIndex({
        "foo": "https://www.instagram.com/reel/AAA/",
        f"foo-{url_hash(url_b)}": "https://www.instagram.com/reel/CCC/",
    })
    slug = index.claim(make_note(title="Foo", source_url=url_b))
    assert slug not in ("foo", f"foo-{url_hash(url_b)}")
    assert slug.startswith("foo-")


def test_the_same_video_keeps_one_file_when_its_title_drifts() -> None:
    """Titles are model output and are not stable across runs.

    Reprocessing one URL must not leave two notes carrying that `source_url`.
    Nothing in the system reconciles that afterwards — `consolidate` merges
    tags, never notes — so the duplicate would simply stay.

    The cost is a filename reflecting the older title. That is cosmetic; the
    frontmatter and body are rewritten, and it is the behaviour the URL-derived
    hash was chosen for in the first place.
    """
    url = "https://www.instagram.com/reel/AAA/"
    index = SlugIndex()
    first = index.claim(make_note(title="Foo", source_url=url))
    second = index.claim(make_note(title="A completely different title", source_url=url))
    assert first == second == "foo"


def test_a_video_reclaims_its_slug_from_its_own_orphaned_transcript(tmp_path: Path) -> None:
    """`write_pair` lands the transcript first, so a kill in between orphans one.

    Reading only the transcript's *name* would make it an unknown owner, and
    the retry of the very video that owns the slug would then be suffixed away
    from its own filename — stranding the orphan for good. The transcript
    carries a source_url precisely so the owner can be read.
    """
    root = tmp_path / "ai-notes-vault"
    (root / "_transcripts").mkdir(parents=True)
    url = "https://www.instagram.com/reel/AAA/"
    (root / "_transcripts" / "foo.md").write_text(
        f'---\ntitle: "Foo (transcript)"\nsource_url: {url}\nkind: transcript\n---\n\nasr\n',
        encoding="utf-8",
    )
    index = SlugIndex.from_vault(root, subdirs=("_transcripts",))
    assert index.claim(make_note(title="Foo", source_url=url)) == "foo"


def test_an_orphaned_transcript_from_a_different_video_still_blocks(tmp_path: Path) -> None:
    """The other direction: reading owners must not make orphans free-for-all."""
    root = tmp_path / "ai-notes-vault"
    (root / "_transcripts").mkdir(parents=True)
    (root / "_transcripts" / "foo.md").write_text(
        '---\ntitle: "Foo (transcript)"\nsource_url: https://www.instagram.com/reel/AAA/\n'
        "kind: transcript\n---\n\nasr\n",
        encoding="utf-8",
    )
    index = SlugIndex.from_vault(root, subdirs=("_transcripts",))
    slug = index.claim(make_note(title="Foo", source_url="https://www.instagram.com/reel/BBB/"))
    assert slug != "foo"


# ── the filesystem has opinions the index has to share ───────────────────


def test_a_lowercase_slug_does_not_clobber_a_mixed_case_note(tmp_path: Path) -> None:
    """macOS ships APFS case-insensitive, so these are ONE file.

    `slugify` only emits lowercase, so the collision can only be with a file
    the pipeline did not write — a hand-made `Weekly-Review-Notes.md`, or one
    of our own notes renamed in Obsidian. Precisely the notes with no second
    copy anywhere.

    Asserted through VaultWriter rather than against the index alone, because
    the index believing a name is free is only harmful once something writes.
    """
    root = tmp_path / "ai-notes-vault"
    root.mkdir()
    (root / "Weekly-Review-Notes.md").write_text(
        '---\ntitle: "hand written"\nsource_url: https://mine.invalid/x/\n---\n\nirreplaceable\n',
        encoding="utf-8",
    )
    index = SlugIndex.from_vault(root)
    slug = index.claim(make_note(title="Weekly review notes",
                                 source_url="https://www.instagram.com/reel/AAA/"))
    VaultWriter(root).write_text(f"{slug}.md", "NEW CONTENT")

    assert "irreplaceable" in (root / "Weekly-Review-Notes.md").read_text(encoding="utf-8")
    assert slug != "weekly-review-notes"


def test_ownership_lookup_is_case_insensitive_both_ways(tmp_path: Path) -> None:
    index = SlugIndex({"Foo-Bar": "https://www.instagram.com/reel/AAA/"})
    assert index.owner_of("foo-bar") == "https://www.instagram.com/reel/AAA/"
    assert "FOO-BAR" in index


# ── frontmatter is parsed as a block, not scanned for ────────────────────


def test_a_quoted_source_url_is_still_recognised(tmp_path: Path) -> None:
    """Obsidian's Properties editor re-saves frontmatter with the URL quoted.

    We write it bare, so an unstripped read would stop matching the moment a
    note is opened in Obsidian — and the video would then fail to recognise
    its own file on a reprocess and write a duplicate.
    """
    root = tmp_path / "ai-notes-vault"
    root.mkdir()
    url = "https://www.instagram.com/reel/AAA/"
    (root / "foo.md").write_text(
        f'---\ntitle: "Foo"\nsource_url: "{url}"\n---\n\nbody\n', encoding="utf-8"
    )
    index = SlugIndex.from_vault(root)
    assert index.owner_of("foo") == url
    assert index.claim(make_note(title="Foo reprocessed", source_url=url)) == "foo"


def test_an_unindented_source_url_in_the_body_claims_nothing(tmp_path: Path) -> None:
    """The dangerous half of the body case, and the reason for two-step parsing.

    An indented body line never matched. An UNINDENTED one did, because a lazy
    "any lines" prefix walks straight past the closing `---`. A note quoting
    frontmatter inside a fenced code block would then be read as owning that
    URL — and the real video, on the "I already own a file" path, would write
    itself over the top of it.
    """
    root = tmp_path / "ai-notes-vault"
    root.mkdir()
    victim = "https://www.instagram.com/reel/VICTIM/"
    (root / "gardening-journal.md").write_text(
        f'---\ntitle: "My journal"\n---\n\nCopied from another note:\n\nsource_url: {victim}\n',
        encoding="utf-8",
    )
    index = SlugIndex.from_vault(root)
    assert index.owner_of("gardening-journal") == ""
    assert index.claim(make_note(title="Real video", source_url=victim)) != "gardening-journal"


def test_a_bom_or_crlf_note_is_still_parsed(tmp_path: Path) -> None:
    """Failing to parse reads as "unknown owner", which costs a note its name."""
    root = tmp_path / "ai-notes-vault"
    root.mkdir()
    url = "https://www.instagram.com/reel/AAA/"
    (root / "crlf.md").write_bytes(
        f'﻿---\r\ntitle: "t"\r\nsource_url: {url}\r\n---\r\n\r\nbody\r\n'.encode("utf-8")
    )
    assert SlugIndex.from_vault(root).owner_of("crlf") == url


def test_two_notes_with_no_source_url_do_not_collapse_onto_one_file() -> None:
    """A blank owner is "unknown", not "matches me".

    Not reachable through today's sources — all three drop a ref with no URL
    before building one — but the class should not rely on its callers for
    that, and a browser-scraper source is the obvious way it would stop being
    true.
    """
    index = SlugIndex()
    first = index.claim(make_note(title="Video number one has this title", source_url=""))
    second = index.claim(make_note(title="Video number one has this title", source_url=""))
    assert first != second
