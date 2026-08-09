"""Tag registry — the live vocabulary injected into every summarization prompt.

Pass 1 (here): reuse an existing tag if one fits, mint a new one only if nothing
does. Pass 2 (consolidate.py) repairs what pass 1 gets wrong.

The registry lives in the VAULT (`_meta/tags.json`), not in the repo, because it
is derived from the notes and grows with them. It is written through the vault
write guard like everything else.

    # ── DELIBERATELY NAIVE: registry injection prompt ───────────────────────
    # CURRENT: dumps every tag, alphabetically, as a flat comma list, with a
    #   one-line instruction to prefer them.
    #
    # WHY THAT IS INSUFFICIENT: three failures, and they compound.
    #   1. ALPHABETICAL, so at 200 tags the ordering carries no signal about
    #      which tags matter. Ranking by count would surface the real
    #      vocabulary first.
    #   2. NO DESCRIPTIONS, so `growth` is unresolvable — plants or startups?
    #      The registry has a `description` field this prompt never uses, and
    #      ambiguous tags are exactly the ones that fragment.
    #   3. UNBOUNDED, so past a few hundred tags the list crowds out the
    #      transcript inside num_ctx and quality degrades silently. There is
    #      no truncation and no signal that truncation occurred.
    #
    #   Accepted by construction: pass 1 is order-dependent. The first video
    #   sees an empty registry, the last sees a mature one. That is inherent
    #   to a growing vocabulary; the consolidation pass repairs it.
    #
    # INTENDED: rank by count, include descriptions, cap the injected list,
    #   and decide what the model is told when the cap truncates.
    # ────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SLUG_OK = re.compile(r"^[a-z0-9]+(?:[-/][a-z0-9]+)*$")


def normalise_tag(tag: str) -> str:
    """Lowercase, hyphenate, keep `/` so Obsidian nested tags survive."""
    t = tag.strip().lower().replace("_", "-")
    # Collapse whitespace around a separator BEFORE hyphenating, or a model
    # emitting "ai-tools / prompts" yields the nonsense tag "ai-tools-/-prompts".
    t = re.sub(r"\s*/\s*", "/", t)
    t = re.sub(r"\s+", "-", t)
    t = re.sub(r"[^a-z0-9/-]", "", t)
    t = re.sub(r"-{2,}", "-", t)
    t = re.sub(r"/{2,}", "/", t)
    # Trim separators at each path segment boundary, not just the ends.
    parts = [p.strip("-") for p in t.split("/") if p.strip("-")]
    return "/".join(parts)


@dataclass
class TagRegistry:
    """tag -> {count, description}."""

    tags: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "TagRegistry":
        p = Path(path)
        if not p.is_file():
            return cls()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls()
        return cls(tags=raw.get("tags", {}))

    def to_json(self) -> str:
        ordered = dict(
            sorted(self.tags.items(), key=lambda kv: (-kv[1].get("count", 0), kv[0]))
        )
        return json.dumps({"tags": ordered}, indent=2, ensure_ascii=False) + "\n"

    def observe(self, tags: list[str], description: str = "") -> list[str]:
        """Record tags from one note. Returns the normalised list."""
        out = []
        for raw in tags:
            t = normalise_tag(raw)
            if not t or not SLUG_OK.match(t):
                continue
            entry = self.tags.setdefault(t, {"count": 0, "description": description})
            entry["count"] = entry.get("count", 0) + 1
            if not entry.get("description") and description:
                entry["description"] = description
            out.append(t)
        return out

    def counts(self) -> dict[str, int]:
        return {t: e.get("count", 0) for t, e in self.tags.items()}

    # -- prompt injection -------------------------------------------------

    def prompt_block(self) -> str:
        """Naive registry rendering. See the module docstring."""
        if not self.tags:
            return "(the registry is empty — this is one of the first notes)"
        return ", ".join(sorted(self.tags))
