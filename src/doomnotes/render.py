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

from doomnotes.models import Note

MAX_SLUG_LEN = 80


def slugify(title: str) -> str:
    """Filename-safe, readable, stable."""
    text = unicodedata.normalize("NFKD", title)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text[:MAX_SLUG_LEN].strip("-") or "untitled"


def url_hash(url: str, length: int = 6) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:length]


def note_slug(note: Note, taken: set[str] | None = None) -> str:
    """Slug for this note, with a short URL hash appended on collision.

    The hash is derived from the URL rather than a counter so that re-running a
    note lands on the same filename instead of accumulating `-2`, `-3` copies.
    """
    base = slugify(note.title)
    if taken is None or base not in taken:
        return base
    return f"{base}-{url_hash(note.source_url)}"


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
