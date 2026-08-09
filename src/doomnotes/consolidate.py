"""Pass 2 — tag consolidation: merge near-duplicates, split over-large tags.

Pass 1 (tags.py) is order-dependent by construction: video #1 sees an empty
registry, video #342 sees a mature one. This pass is what repairs that, so it is
meant to be re-run as the vault grows. It is idempotent — running it twice
changes nothing the second time.

Two jobs, per Tier-1 #3:
  merge  `garden` / `gardening` / `plants`  -> one canonical tag
  split  `coding` (>15 notes)               -> `coding/databases`, `coding/frontend`

Splitting uses tag CO-OCCURRENCE rather than re-reading content: if 6 of the 20
`coding` notes are also tagged `databases`, that is the sub-cluster, and it is
visible without another model call. Obsidian nests natively, so `coding/databases`
gives both the rollup and the specificity.

Every rewrite goes through the vault write guard.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from doomnotes.tags import TagRegistry, normalise_tag
from doomnotes.vault import VaultWriter

log = logging.getLogger(__name__)

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
TAGS_LINE = re.compile(r"^tags:\s*\[(.*?)\]\s*$", re.M)
TOPIC_LINE = re.compile(r"^topic:\s*(.+?)\s*$", re.M)


@dataclass
class ConsolidationPlan:
    merges: dict[str, str] = field(default_factory=dict)      # from -> to
    splits: dict[str, dict[str, str]] = field(default_factory=dict)  # parent -> {child_key: nested}
    notes_touched: int = 0
    notes_scanned: int = 0

    def render(self) -> str:
        lines = ["Tag consolidation", "-" * 50]
        lines.append(f"  notes scanned : {self.notes_scanned}")
        lines.append(f"  notes rewritten: {self.notes_touched}")
        lines.append(f"  merges ({len(self.merges)}):")
        for a, b in sorted(self.merges.items()):
            lines.append(f"      {a}  ->  {b}")
        lines.append(f"  splits ({len(self.splits)}):")
        for parent, mapping in sorted(self.splits.items()):
            for child, nested in sorted(mapping.items()):
                lines.append(f"      {parent} + {child}  ->  {nested}")
        if not self.merges and not self.splits:
            lines.append("      (nothing to do — registry is already consolidated)")
        return "\n".join(lines)


DERIVATIONAL_SUFFIXES = ("ing", "ed", "er", "s", "es")


def _is_derivation(short: str, long: str) -> bool:
    """True if `long` is `short` plus a common English suffix.

    Handles consonant doubling (run -> running) and a dropped trailing `e`
    (bake -> baking). Restricting to real suffixes is what keeps `post` and
    `postgres` apart, which a plain prefix check would merge.
    """
    if short == long or len(short) < 3:
        return False
    stems = {short, short + short[-1], short.rstrip("e")}
    return any(long == stem + suf for stem in stems for suf in DERIVATIONAL_SUFFIXES)


def _singular(tag: str) -> str:
    for suffix in ("ies", "es", "s"):
        if tag.endswith(suffix) and len(tag) > len(suffix) + 2:
            return tag[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return tag


def plan_merges(counts: dict[str, int], threshold: float = 0.85) -> dict[str, str]:
    """Map near-duplicate tags onto a canonical one.

    Canonical = the most-used variant, tie-broken by shortest then alphabetical,
    so the result is deterministic and re-running produces the same answer.
    """
    merges: dict[str, str] = {}
    # Group by singular stem first — catches garden/gardens, plant/plants.
    by_stem: dict[str, list[str]] = defaultdict(list)
    for tag in counts:
        by_stem[_singular(tag)].append(tag)

    def canonical(group: list[str]) -> str:
        return sorted(group, key=lambda t: (-counts[t], len(t), t))[0]

    for group in by_stem.values():
        if len(group) > 1:
            keep = canonical(group)
            for tag in group:
                if tag != keep:
                    merges[tag] = keep

    # Then derivational suffixes: garden/gardening, cook/cooking, build/builder.
    # Deliberately NOT a general prefix rule — that merges `post` into
    # `postgres`. Only these suffixes count, so the match stays morphological.
    remaining = sorted(t for t in counts if t not in merges)
    for i, a in enumerate(remaining):
        for b in remaining[i + 1 :]:
            if a in merges or b in merges or "/" in a or "/" in b:
                continue
            short, long = (a, b) if len(a) <= len(b) else (b, a)
            if _is_derivation(short, long) or (
                difflib.SequenceMatcher(None, a, b).ratio() >= threshold
            ):
                keep = canonical([a, b])
                merges[b if keep == a else a] = keep

    # Collapse chains so a -> b -> c becomes a -> c.
    for tag in list(merges):
        seen = {tag}
        target = merges[tag]
        while target in merges and target not in seen:
            seen.add(target)
            target = merges[target]
        merges[tag] = target
    return {a: b for a, b in merges.items() if a != b}


def plan_splits(
    note_tags: list[list[str]],
    counts: dict[str, int],
    threshold: int = 15,
    min_cluster: int = 3,
) -> dict[str, dict[str, str]]:
    """For each oversized tag, find co-occurring tags that form a sub-cluster."""
    splits: dict[str, dict[str, str]] = {}
    for parent, n in counts.items():
        if n < threshold or "/" in parent:
            continue
        co = Counter()
        for tags in note_tags:
            if parent in tags:
                co.update(t for t in tags if t != parent and "/" not in t)
        mapping = {
            child: f"{parent}/{child}"
            for child, c in co.items()
            if c >= min_cluster and c < n  # a tag on ALL of them isn't a sub-cluster
        }
        if mapping:
            splits[parent] = mapping
    return splits


def _read_tags(text: str) -> list[str]:
    m = FRONTMATTER.search(text)
    if not m:
        return []
    tm = TAGS_LINE.search(m.group(1))
    if not tm:
        return []
    return [normalise_tag(t) for t in tm.group(1).split(",") if t.strip()]


def _rewrite(text: str, new_tags: list[str], new_topic: str | None) -> str:
    m = FRONTMATTER.search(text)
    if not m:
        return text
    fm = m.group(1)
    fm2 = TAGS_LINE.sub("tags: [" + ", ".join(new_tags) + "]", fm, count=1)
    if new_topic:
        fm2 = TOPIC_LINE.sub(f"topic: {new_topic}", fm2, count=1)
    return text[: m.start(1)] + fm2 + text[m.end(1) :]


def consolidate(
    writer: VaultWriter,
    registry_rel: str,
    *,
    merge_threshold: float = 0.85,
    split_threshold: int = 15,
    dry_run: bool = False,
) -> ConsolidationPlan:
    """Run pass 2 over the vault. Idempotent."""
    root = writer.root
    note_files = [
        p for p in sorted(root.glob("*.md")) if not p.name.startswith(".")
    ]

    per_note: dict[Path, list[str]] = {}
    for p in note_files:
        per_note[p] = _read_tags(p.read_text(encoding="utf-8"))

    plan = ConsolidationPlan(notes_scanned=len(note_files))
    counts = Counter(t for tags in per_note.values() for t in tags)
    if not counts:
        return plan

    plan.merges = plan_merges(dict(counts), merge_threshold)

    merged_note_tags = [
        [plan.merges.get(t, t) for t in tags] for tags in per_note.values()
    ]
    merged_counts = Counter(t for tags in merged_note_tags for t in tags)
    plan.splits = plan_splits(merged_note_tags, dict(merged_counts), split_threshold)

    if dry_run:
        return plan

    new_registry = TagRegistry()
    for path, tags in per_note.items():
        text = path.read_text(encoding="utf-8")
        updated = [plan.merges.get(t, t) for t in tags]

        for parent, mapping in plan.splits.items():
            if parent in updated:
                for child, nested in mapping.items():
                    if child in updated:
                        updated = [nested if t == parent else t for t in updated]
                        break

        # Dedupe, preserve order, drop a parent made redundant by its own child.
        seen: set[str] = set()
        final: list[str] = []
        for t in updated:
            if t in seen:
                continue
            if any(o != t and o.startswith(t + "/") for o in updated):
                continue
            seen.add(t)
            final.append(t)

        new_registry.observe(final)

        if final != tags:
            rewritten = _rewrite(text, final, final[0] if final else None)
            writer.write_text(path.name, rewritten)
            plan.notes_touched += 1

    writer.write_text(registry_rel, new_registry.to_json())
    return plan
