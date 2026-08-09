"""Tag consolidation (pass 2) — the cases test_render_tags.py does not cover.

`test_render_tags.py` already asserts the happy path: plurals collapse, splits
appear above threshold, the rewrite is idempotent, and every write goes through
the guard. This file covers what pass 2 must *not* do — the ways a rewrite pass
over notes you cannot regenerate can quietly lose information.

That framing is deliberate. Consolidation is the only stage that edits notes
already on disk, so its failure mode is not "a bad note" but "a good note,
damaged". Everything here is an invariant; none of it encodes a HANDS-ON
baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doomnotes.consolidate import consolidate, plan_merges, plan_splits
from doomnotes.tags import TagRegistry
from doomnotes.vault import VaultWriter

NOTE = """---
title: "A note about {topic}"
source_url: https://www.instagram.com/reel/{code}/
platform: instagram
author: "@someone"
posted_at: 2026-07-14
topic: {topic}
tags: [{tags}]
has_transcript: true
source_order: {order}
processed_at: 2026-08-09
---

## Summary

Body text that must survive the rewrite untouched.

## Key points

- a point with [brackets] and a `tags: [not-frontmatter]` decoy

---

[[_transcripts/a-note|Raw transcript]] · [Original](https://example.invalid/)
"""


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "ai-notes-vault"
    (root / "_meta").mkdir(parents=True)
    (root / "_transcripts").mkdir(parents=True)
    return root


@pytest.fixture()
def writer(vault: Path) -> VaultWriter:
    return VaultWriter(vault)


def write_notes(vault: Path, specs: list[list[str]]) -> None:
    for i, tags in enumerate(specs):
        (vault / f"note-{i:03d}.md").write_text(
            NOTE.format(
                topic=tags[0],
                code=f"AAAAAAAAA{i:02d}",
                tags=", ".join(tags),
                order=i,
            ),
            encoding="utf-8",
        )


# ── what must not be damaged ─────────────────────────────────────────────


def test_rewrite_preserves_body_and_non_tag_frontmatter(writer: VaultWriter, vault: Path) -> None:
    write_notes(vault, [["gardens"], ["garden"], ["garden"]])
    before = (vault / "note-000.md").read_text(encoding="utf-8")

    consolidate(writer, "_meta/tags.json")

    after = (vault / "note-000.md").read_text(encoding="utf-8")
    assert after != before, "this note's tags should have been merged"
    assert "source_url: https://www.instagram.com/reel/AAAAAAAAA00/" in after
    assert 'author: "@someone"' in after
    assert "has_transcript: true" in after
    assert "Body text that must survive the rewrite untouched." in after
    assert "[[_transcripts/a-note|Raw transcript]]" in after


def test_a_tags_line_in_the_body_is_not_mistaken_for_frontmatter(
    writer: VaultWriter, vault: Path
) -> None:
    """The frontmatter regex is anchored to the leading `---` block.

    A note whose body happens to contain `tags: [...]` — a code snippet, a
    quoted YAML example — must not have its body rewritten.
    """
    write_notes(vault, [["gardens"], ["garden"], ["garden"]])
    consolidate(writer, "_meta/tags.json")
    after = (vault / "note-000.md").read_text(encoding="utf-8")
    assert "`tags: [not-frontmatter]`" in after


def test_transcripts_are_never_rewritten(writer: VaultWriter, vault: Path) -> None:
    """Transcripts are receipts. Nothing derived may edit them."""
    write_notes(vault, [["gardens"], ["garden"], ["garden"]])
    transcript = vault / "_transcripts" / "a-note.md"
    transcript.write_text("---\nkind: transcript\ntags: [garden]\n---\n\nraw asr\n", "utf-8")
    before = transcript.read_text(encoding="utf-8")

    consolidate(writer, "_meta/tags.json")

    assert transcript.read_text(encoding="utf-8") == before


def test_dry_run_writes_nothing_at_all(writer: VaultWriter, vault: Path) -> None:
    write_notes(vault, [["gardens"], ["garden"], ["garden"]])
    snapshot = {p: p.read_bytes() for p in sorted(vault.rglob("*")) if p.is_file()}

    plan = consolidate(writer, "_meta/tags.json", dry_run=True)

    assert plan.merges, "there was something to do, so this is a real dry run"
    assert plan.notes_touched == 0
    after = {p: p.read_bytes() for p in sorted(vault.rglob("*")) if p.is_file()}
    assert after == snapshot


def test_an_empty_vault_is_a_no_op(writer: VaultWriter) -> None:
    plan = consolidate(writer, "_meta/tags.json")
    assert plan.notes_scanned == 0
    assert plan.merges == {} and plan.splits == {}


# ── merge judgement ──────────────────────────────────────────────────────


def test_shared_prefix_is_not_a_merge() -> None:
    """`post` must not be swallowed by `postgres`.

    A plain prefix rule merges them; only real derivational suffixes count,
    which is what keeps the rule morphological rather than lexical.
    """
    merges = plan_merges({"post": 9, "postgres": 4})
    assert merges == {}


@pytest.mark.parametrize(
    "pair",
    [
        # Tags this vault will plausibly hold at the same time. Not invented
        # adversarial pairs — these are collisions a coding/gardening/AI-tools
        # corpus actually produces.
        #
        # Note what is NOT here: `plant`/`planter`. That pair DOES merge, and
        # correctly — `-er` is a real derivation, and the rule's own comment
        # names `build`/`builder` as a case it is meant to catch. Adding it
        # here would have invented a defect out of the rule working.
        ("go", "golang"),
        ("ai", "aim"),
    ],
)
def test_unrelated_words_sharing_a_prefix_stay_separate(pair: tuple[str, str]) -> None:
    merged = plan_merges({pair[0]: 5, pair[1]: 5})
    assert merged == {}, f"{pair} should not merge, got {merged}"


def test_merge_target_is_the_most_used_variant() -> None:
    """Canonical = most used, so consolidation follows your actual vocabulary."""
    merges = plan_merges({"plant": 2, "plants": 11})
    assert merges == {"plant": "plants"}


def test_merge_chains_collapse_to_one_hop() -> None:
    """a -> b -> c must resolve to a -> c, or a note keeps a dead tag."""
    merges = plan_merges({"garden": 5, "gardens": 2, "gardening": 9})
    assert set(merges.values()) == {"gardening"}
    assert merges["garden"] == "gardening"


def test_a_child_tag_is_never_merged_into_its_parent() -> None:
    """`coding/databases` must not collapse back into `coding`.

    That would undo the split pass 2 just performed, and the two passes would
    fight each other on every run.

    Note what is deliberately NOT asserted: that nested tags are exempt from
    merging altogether. `coding/test` and `coding/tests` DO merge, and should —
    they are singular and plural of one tag, and collapsing them is the whole
    job. An earlier version of this test used a namespaced pair whose stems
    happen to differ, so it passed without exercising anything.
    """
    assert plan_merges({"coding": 10, "coding/databases": 6}) == {}
    assert plan_merges({"coding/databases": 6, "coding": 10}) == {}


def test_sibling_nested_tags_stay_distinct() -> None:
    """`coding/databases` and `coding/frontend` are the specificity, not noise."""
    assert plan_merges({"coding/databases": 6, "coding/frontend": 5}) == {}


# ── split judgement ──────────────────────────────────────────────────────


def test_a_tag_on_every_note_is_not_a_subcluster() -> None:
    """If all 20 `coding` notes are also `programming`, that is a synonym.

    Splitting on it would produce `coding/programming` for every note, which
    is noise, not specificity.
    """
    note_tags = [["coding", "programming"] for _ in range(20)]
    splits = plan_splits(note_tags, {"coding": 20, "programming": 20}, threshold=15)
    assert splits == {}


def test_split_drops_the_parent_it_nests_under(writer: VaultWriter, vault: Path) -> None:
    """`[coding, coding/databases]` is redundant — Obsidian rolls the child up."""
    specs = [["coding", "databases"] for _ in range(6)]
    specs += [["coding", "frontend"] for _ in range(6)]
    specs += [["coding", "testing"] for _ in range(6)]
    write_notes(vault, specs)

    consolidate(writer, "_meta/tags.json", split_threshold=15)

    tags_line = next(
        line
        for line in (vault / "note-000.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("tags:")
    )
    assert "coding/databases" in tags_line
    assert "coding," not in tags_line and not tags_line.endswith("coding]")


def test_registry_counts_match_the_notes_after_consolidation(
    writer: VaultWriter, vault: Path
) -> None:
    """The registry is what pass 1 injects next run. If it drifts, so do tags."""
    write_notes(vault, [["gardens"], ["garden"], ["garden"], ["cooking"]])

    consolidate(writer, "_meta/tags.json")

    registry = TagRegistry.load(vault / "_meta" / "tags.json")
    from_notes: dict[str, int] = {}
    for note in sorted(vault.glob("*.md")):
        line = next(
            l for l in note.read_text(encoding="utf-8").splitlines() if l.startswith("tags:")
        )
        for tag in line[len("tags: ["):-1].split(","):
            tag = tag.strip()
            if tag:
                from_notes[tag] = from_notes.get(tag, 0) + 1
    assert registry.counts() == from_notes


# ── the registry survives the pass ───────────────────────────────────────
#
# Counts are rebuilt from the notes every run, because the notes are the truth.
# Descriptions cannot be rebuilt that way — frontmatter does not carry them —
# so they have to be carried across explicitly, which is what these pin.


@pytest.mark.parametrize(
    "counts,expected_canonical",
    [
        # Three morphologically identical vocabularies. Before the two rules
        # were changed to build equivalence groups, only the first consolidated
        # — the other two stranded their third tag, because whichever variant
        # the tally favoured was the only one the second rule could see.
        pytest.param({"garden": 5, "gardens": 2, "gardening": 9}, "gardening", id="singular-wins"),
        pytest.param({"garden": 1, "gardens": 2, "gardening": 9}, "gardening", id="plural-wins"),
        pytest.param({"build": 3, "builder": 5, "building": 4}, "builder", id="suffix-race"),
    ],
)
def test_merging_does_not_depend_on_which_variant_is_most_used(
    counts: dict[str, int], expected_canonical: str
) -> None:
    """Whether a tag merges must depend on the words, not on the tally.

    All three vocabularies below are morphologically identical. Two of them
    consolidate; one silently does not, because stage 1's choice of canonical
    changes what stage 2 is able to see.
    """
    merges = plan_merges(counts)
    merged = set(merges.values()) | {t for t in counts if t not in merges}
    assert merged == {expected_canonical}, (
        f"{counts} left {sorted(merged)} instead of collapsing to one tag"
    )


def test_consolidation_preserves_tag_descriptions(writer: VaultWriter, vault: Path) -> None:
    write_notes(vault, [["cooking"], ["cooking"]])
    (vault / "_meta" / "tags.json").write_text(
        json.dumps({"tags": {"cooking": {"count": 2, "description": "recipes and technique"}}}),
        encoding="utf-8",
    )

    consolidate(writer, "_meta/tags.json")

    registry = TagRegistry.load(vault / "_meta" / "tags.json")
    assert registry.tags["cooking"]["description"] == "recipes and technique"


def test_a_merge_inherits_the_description_of_what_it_absorbed(
    writer: VaultWriter, vault: Path
) -> None:
    write_notes(vault, [["garden"], ["gardens"], ["gardens"]])
    (vault / "_meta" / "tags.json").write_text(
        json.dumps(
            {
                "tags": {
                    "garden": {"count": 1, "description": "growing plants outdoors"},
                    "gardens": {"count": 2, "description": ""},
                }
            }
        ),
        encoding="utf-8",
    )

    consolidate(writer, "_meta/tags.json")

    registry = TagRegistry.load(vault / "_meta" / "tags.json")
    assert registry.tags["gardens"]["description"] == "growing plants outdoors"
