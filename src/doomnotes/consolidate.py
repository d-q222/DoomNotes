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


def _related(a: str, b: str, threshold: float) -> bool:
    """Whether two tags are variants of one another.

    Two rules. Derivational suffixes catch garden/gardening, cook/cooking,
    build/builder. Sequence similarity catches typos and spelling variants.
    Deliberately NOT a general prefix rule — that merges `post` into `postgres`.
    """
    if "/" in a or "/" in b:
        # Nested tags are pass-2 *output*. Merging one into its parent would
        # undo the split this same pass just performed, and the two halves
        # would fight each other on every run.
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return _is_derivation(short, long) or (
        difflib.SequenceMatcher(None, a, b).ratio() >= threshold
    )


def plan_merges(counts: dict[str, int], threshold: float = 0.85) -> dict[str, str]:
    """Map near-duplicate tags onto a canonical one.

    Canonical = the most-used variant, tie-broken by shortest then alphabetical,
    so the result is deterministic and re-running produces the same answer.

    Both rules — shared singular stem, and derivation/similarity — build
    EQUIVALENCE GROUPS, and a canonical is chosen once per group at the end.

    Why not apply them in sequence, tag onto tag: the first rule consumes the
    token the second one needs. With {garden: 1, gardens: 2, gardening: 9},
    stem-grouping picks `gardens` as canonical because it outnumbers `garden`,
    and stage 2 then has only `gardens` to compare against — `gardens` is not a
    derivation of `gardening`, so `gardening` never merges. Give `garden` the
    larger count and the identical vocabulary consolidates fully. Whether a tag
    merges must depend on the words, not on the tally.

    That failure was also permanent rather than merely wrong: once notes have
    been rewritten from `garden` to `gardens`, the bridging form no longer
    exists in the vault, so no later run can recover it.
    """
    parent: dict[str, str] = {t: t for t in counts}

    def find(t: str) -> str:
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Root choice here is arbitrary; the canonical is picked per group
            # afterwards. Ordering by name keeps unions deterministic.
            parent[max(ra, rb)] = min(ra, rb)

    tags = sorted(counts)

    # Shared singular stem — garden/gardens, plant/plants, coding/test(s).
    by_stem: dict[str, list[str]] = defaultdict(list)
    for tag in tags:
        by_stem[_singular(tag)].append(tag)
    for group in by_stem.values():
        for other in group[1:]:
            union(group[0], other)

    # Derivation and similarity, over every pair. Comparing raw tags rather
    # than group representatives is what makes the result order-independent.
    for i, a in enumerate(tags):
        for b in tags[i + 1 :]:
            if find(a) != find(b) and _related(a, b, threshold):
                union(a, b)

    grouped: dict[str, list[str]] = defaultdict(list)
    for tag in tags:
        grouped[find(tag)].append(tag)

    merges: dict[str, str] = {}
    for group in grouped.values():
        if len(group) < 2:
            continue
        keep = sorted(group, key=lambda t: (-counts[t], len(t), t))[0]
        for tag in group:
            if tag != keep:
                merges[tag] = keep
    return merges


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


def _carry_descriptions(
    old: TagRegistry,
    merges: dict[str, str],
    counts: dict[str, int],
) -> dict[str, str]:
    """Descriptions that should survive this pass, keyed by their new tag name.

    Two rules:

    - A tag that is not merged keeps its own description. Pass 2 rebuilds the
      registry from note frontmatter, and frontmatter carries no descriptions,
      so without this every annotation is silently emptied on every run — the
      pass that runs after every batch quietly deleting the field #5.2 depends
      on.

    - A canonical tag with no description of its own inherits from the most-used
      tag it absorbed. `garden` documented and `gardens` bare should not lose the
      documentation just because the plural was more common. A canonical that
      already has one keeps it: what the surviving name says about itself beats
      what an absorbed variant said.

    Splits deliberately inherit nothing. `coding/databases` is a narrower tag
    than `coding`, so handing it the parent's description would assert something
    about it that was never written.
    """
    out: dict[str, str] = {}
    for tag, entry in old.tags.items():
        description = (entry.get("description") or "").strip()
        if description and tag not in merges:
            out[tag] = description

    absorbed: dict[str, list[str]] = defaultdict(list)
    for source, target in merges.items():
        absorbed[target].append(source)

    for target, sources in absorbed.items():
        if out.get(target):
            continue
        ranked = sorted(sources, key=lambda t: (-counts.get(t, 0), t))
        for source in ranked:
            inherited = (old.tags.get(source, {}).get("description") or "").strip()
            if inherited:
                out[target] = inherited
                break
    return out


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

    # Counts are always recomputed from the notes — the notes are the truth and
    # a stale count would be worse than none. Descriptions are not derivable
    # from a note, so they are carried across instead of being rebuilt as "".
    old_registry = TagRegistry.load(writer.root / registry_rel)
    descriptions = _carry_descriptions(old_registry, plan.merges, dict(counts))

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

        for tag in final:
            new_registry.observe([tag], descriptions.get(tag, ""))

        if final != tags:
            rewritten = _rewrite(text, final, final[0] if final else None)
            writer.write_text(path.name, rewritten)
            plan.notes_touched += 1

    writer.write_text(registry_rel, new_registry.to_json())
    return plan
