"""Note -> markdown, and the slug/filename rules.

Task 6.2. Frontmatter shape is Tier-1 #2, ruled: descriptive LLM title, posted
date from yt-dlp metadata, transcript kept out of the note body but wikilinked.

The interaction worth remembering: `posted_at` comes from yt-dlp, so it is null
on caption-only Instagram notes — the field added specifically to judge recency
is missing on exactly the notes whose content cannot be judged either. That
is a consequence of two rulings meeting, not an oversight.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from doomnotes.models import Note

MAX_SLUG_LEN = 80

# Cheap enough to read for every note in the vault, and anchored so a
# `source_url:` line in a note *body* cannot be mistaken for the real one.
FRONTMATTER_SOURCE_URL = re.compile(r"\A---\n(?:.*?\n)??source_url:[ \t]*(\S+)\s*$", re.M | re.S)


def slugify(title: str) -> str:
    """Filename-safe, readable, stable."""
    text = unicodedata.normalize("NFKD", title)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text[:MAX_SLUG_LEN].strip("-") or "untitled"


def url_hash(url: str, length: int = 6) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:length]


def note_slug(note: Note, taken: "set[str] | SlugIndex | None" = None) -> str:
    """Slug for this note, with a short URL hash appended on collision.

    The hash is derived from the URL rather than a counter so that re-running a
    note lands on the same filename instead of accumulating `-2`, `-3` copies.

    `taken` may be a bare set of slugs — in which case any match is treated as a
    collision — or a `SlugIndex`, which knows *whose* slug each one is and so
    can tell "another video wants this name" from "this video already owns it".
    """
    base = slugify(note.title)
    if taken is None:
        return base
    if isinstance(taken, SlugIndex):
        return taken.allocate(base, note.source_url)
    if base not in taken:
        return base
    return f"{base}-{url_hash(note.source_url)}"


class SlugIndex:
    """Which slugs are taken, and by which video.

    Rebuilt from the vault at the start of each run, because runs are separate
    processes days apart — an in-memory set only ever knew about the notes the
    *current* run had written, so two videos that generated the same title in
    two different runs quietly resolved to one file and the earlier note was
    replaced. The store still recorded the lost video as done, so nothing
    regenerated it and nothing reported it.

    Keyed on `source_url` rather than on the slug alone, because the two cases
    look identical from the filename and must not be treated alike:

        another video wants this name  -> append the URL hash, keep both notes
        this video already owns it     -> reuse the name, overwrite in place

    Collapsing them either loses a note (always reuse) or accumulates a
    duplicate on every re-run (always suffix).

    It is indexed both ways. slug -> owner answers "may I have this name"; the
    reverse, owner -> slug, answers "do I already have one", which matters
    because a video's title is model output and is not stable across runs. The
    same URL summarised twice can produce two different titles, and without the
    reverse lookup the second run has no way to discover it already owns a file
    — so the vault ends up with two notes carrying the same `source_url` and
    nothing to reconcile them.
    """

    def __init__(self, owners: dict[str, str] | None = None) -> None:
        self._owners: dict[str, str] = dict(owners or {})
        self._by_url: dict[str, str] = {}
        for slug, url in self._owners.items():
            if url:
                self._by_url.setdefault(url, slug)

    @classmethod
    def from_vault(cls, root: str | Path, *, subdirs: tuple[str, ...] = ()) -> "SlugIndex":
        """Read every note already in the vault and record who owns its slug.

        Notes are scanned at the top level; `_meta/` holds no notes. A note
        without a readable `source_url` still reserves its slug — it is
        someone's note, and the conservative reading is that overwriting it
        would lose something.

        Transcripts are scanned too, and their frontmatter is read rather than
        just their names. `write_pair` deliberately lands the transcript first
        so a wikilink can never dangle, which means a run killed in between
        leaves a transcript with no note. Recording only the name would make
        that orphan an *unknown* owner, so the retry of the very video that owns
        the slug would read it as a collision and get suffixed away from its own
        filename — leaving the orphan stranded permanently. `render_transcript`
        writes a `source_url`, so the owner is right there to be read.
        """
        root = Path(root)
        resolved_root = root.resolve(strict=False)
        owners: dict[str, str] = {}
        for path in sorted(root.glob("*.md")):
            owners[path.stem] = cls._owner_in(path, resolved_root)
        for sub in subdirs:
            for path in sorted((root / sub).glob("*.md")):
                if path.stem not in owners:
                    owners[path.stem] = cls._owner_in(path, resolved_root)
        return cls(owners)

    @staticmethod
    def _owner_in(path: Path, resolved_root: Path) -> str:
        """The source_url in `path`, or "" if it cannot be trusted.

        Resolve-then-verify, the same order vault.py's write guard uses and for
        the same reason: a symlink inside the vault resolves out of it. Reading
        through one is far less serious than writing through one — the result
        only ever feeds an ownership comparison and never reaches output or a
        filesystem path — but a symlinked `evil.md` could otherwise claim a URL
        it does not own, and the conservative answer costs nothing.

        Returning "" is the safe direction: the slug stays reserved, so the
        worst case is a note getting a hash suffix it did not strictly need.
        """
        try:
            if not path.resolve(strict=False).is_relative_to(resolved_root):
                return ""
            text = path.read_text(encoding="utf-8")
        except OSError:
            return ""
        match = FRONTMATTER_SOURCE_URL.search(text)
        return match.group(1) if match else ""

    def owner_of(self, slug: str) -> str | None:
        """The URL that owns `slug`, `""` if unknown, or None if it is free."""
        return self._owners.get(slug)

    def slug_for_url(self, url: str) -> str | None:
        """The slug this URL already owns, if it owns one."""
        return self._by_url.get(url) if url else None

    def _free_for(self, slug: str, url: str) -> bool:
        owner = self._owners.get(slug)
        return owner is None or owner == url

    def allocate(self, base: str, url: str) -> str:
        """The filename `url` should use, given a slug derived from its title.

        Three cases, in order:

        1. **This URL already owns a file.** Reuse that name, even though the
           title may have drifted since. Writing a second file would leave two
           notes carrying one `source_url` and nothing in the system able to
           reconcile them — `consolidate` merges tags, never notes. Reusing is
           also what the URL-derived hash was for: re-running a video lands on
           its existing file rather than accumulating copies.

        2. **The base name is free, or already ours.** Take it.

        3. **Another video holds it.** Append this URL's hash — and then check
           *that* name too, rather than trusting it. A six-hex-character suffix
           is not guaranteed unique against a note whose own title happens to
           slugify to `<base>-<6 hex>`, and an unchecked claim there overwrites
           a real note, which is the exact failure this class exists to stop.
           Widening the hash is deterministic, so a given URL still resolves to
           the same filename on every run.
        """
        existing = self.slug_for_url(url)
        if existing is not None:
            return existing

        if self._free_for(base, url):
            return base

        for length in (6, 8, 12, 16, 32, 64):
            candidate = f"{base}-{url_hash(url, length)}"
            if self._free_for(candidate, url):
                return candidate
        # Unreachable short of a full SHA-256 collision, but silently returning
        # a taken name would mean overwriting someone's note.
        raise RuntimeError(f"could not allocate a free filename for {base!r}")

    def claim(self, note: Note) -> str:
        """Allocate this note's slug and record the claim."""
        slug = self.allocate(slugify(note.title), note.source_url)
        self._owners[slug] = note.source_url
        if note.source_url:
            self._by_url.setdefault(note.source_url, slug)
        return slug

    def __contains__(self, slug: object) -> bool:
        return slug in self._owners

    def __len__(self) -> int:
        return len(self._owners)


def _yaml_scalar(value: str) -> str:
    """Quote a YAML scalar. Frontmatter that breaks makes Obsidian ignore it."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", " ").strip()
    return f'"{escaped}"'


def _date(value: datetime | None) -> str | None:
    return value.date().isoformat() if value else None


def render_note(note: Note, transcript_slug: str | None = None,
                transcripts_dir: str = "_transcripts") -> str:
    """Render the full markdown file."""
    fm: list[str] = ["---"]
    fm.append(f"title: {_yaml_scalar(note.title)}")
    fm.append(f"source_url: {note.source_url}")
    fm.append(f"platform: {note.platform}")
    if note.author:
        fm.append(f"author: {_yaml_scalar(note.author)}")
    if (posted := _date(note.posted_at)):
        fm.append(f"posted_at: {posted}")
    if (saved := _date(note.saved_at)):
        fm.append(f"saved_at: {saved}")
    if note.topic:
        fm.append(f"topic: {note.topic}")
    fm.append("tags: [" + ", ".join(note.tags) + "]")
    fm.append(f"has_transcript: {str(note.has_transcript).lower()}")
    if note.source_order is not None:
        fm.append(f"source_order: {note.source_order}")
    fm.append(f"processed_at: {_date(note.processed_at or datetime.now())}")
    fm.append("---")

    body: list[str] = ["", "## Summary", "", note.summary.strip(), ""]

    if note.key_points:
        body += ["## Key points", ""]
        body += [f"- {p}" for p in note.key_points]
        body.append("")

    if note.links:
        body += ["## Links & products mentioned", ""]
        body += [f"- {l}" for l in note.links]
        body.append("")

    # Instagram only: the caption ships in the export, so it is recorded
    # verbatim. Other sources have no caption independent of the download.
    if note.caption:
        body += ["## Caption", "", note.caption.strip(), ""]

    footer_bits = []
    if transcript_slug and note.has_transcript:
        footer_bits.append(f"[[{transcripts_dir}/{transcript_slug}|Raw transcript]]")
    footer_bits.append(f"[Original]({note.source_url})")
    body += ["---", "", " · ".join(footer_bits), ""]

    return "\n".join(fm + body)


def render_transcript(note: Note, transcript: str) -> str:
    """Transcript file. Carries a link back so the pair is navigable both ways."""
    return "\n".join(
        [
            "---",
            f"title: {_yaml_scalar(note.title + ' (transcript)')}",
            f"source_url: {note.source_url}",
            f"platform: {note.platform}",
            "kind: transcript",
            "---",
            "",
            "> Raw automatic speech recognition. Expect errors; the summary is",
            "> the reference, this is the receipt.",
            "",
            transcript.strip(),
            "",
        ]
    )
