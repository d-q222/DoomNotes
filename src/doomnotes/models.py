"""Core data shapes that flow between pipeline stages.

`VideoRef.caption_source` records where a ref's caption comes from. It is
declared by the adapter that built the ref, never inferred from `platform`,
because salvageability is not a platform property: an Instagram *export* ref
carries its caption, while a manual-list ref for the same platform does not —
its caption arrives with the media, exactly as TikTok's does.

That declaration is what `has_export_caption` reads, and it is the rule
deciding whether a failed download still yields a note.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args

Platform = Literal["instagram", "tiktok"]

# Where a ref's caption comes from. "export" means the export file carries it,
# so it is in hand before any download. "media" means it arrives with the
# download itself, so a failed download leaves nothing to salvage.
CaptionSource = Literal["export", "media"]
_CAPTION_SOURCES = get_args(CaptionSource)


@dataclass(frozen=True, slots=True)
class VideoRef:
    """One saved video, as it leaves a source and before anything is fetched.

    `url` is the identity of a ref everywhere in the system — the store keys on
    it, dedup keys on it, and it is the only field guaranteed to be present.
    """

    url: str
    platform: Platform

    # Declared by the source adapter. The default is the conservative case; a
    # source that carries captions and forgets to say so cannot pass silently,
    # because __post_init__ rejects a "media" ref that has one.
    caption_source: CaptionSource = "media"

    # Instagram export only: the caption ships in the export itself, which is
    # what makes a caption-only fallback note possible when the download fails.
    caption: str | None = None
    author: str | None = None

    # TikTok export only: an exact favourite timestamp.
    saved_at: datetime | None = None

    # Position in the source's own newest-first ordering. Present for both
    # exports; it is Instagram's *only* recency signal.
    source_order: int | None = None

    def __post_init__(self) -> None:
        """Reject the one combination no source can produce.

        A caption that arrives with the media cannot be present before the
        download. This lives on the model rather than in an adapter so it holds
        for every construction path — tests and future sources included.
        """
        if self.caption_source not in _CAPTION_SOURCES:
            # `Literal` is a type-checker annotation, not a runtime constraint.
            # An unrecognised value would read as "not export" and silently make
            # a salvageable Instagram ref unsalvageable.
            raise ValueError(
                f"caption_source must be one of {_CAPTION_SOURCES}, "
                f"got {self.caption_source!r}: {self.url}"
            )
        # Deliberately NOT rejected: platform="tiktok" with caption_source="export".
        # No adapter can emit it — each declares its own provenance, asserted in
        # tests/test_sources.py — and the only rule that would reject it, "tiktok
        # implies not export", encodes a fact about TikTok's *export file format*
        # rather than about the platform. That belongs to the adapter that reads
        # the file, not to the shape every stage passes around, and it would be
        # wrong the first time either export schema grows.
        if self.caption_source == "media" and (self.caption or "").strip():
            raise ValueError(
                f"caption_source='media' but a caption is already present: {self.url}. "
                "Declare caption_source='export' if this source carries captions."
            )

    @property
    def has_export_caption(self) -> bool:
        """Whether a failed download can still produce a note.

        Needs both halves: a source that carries captions, and a non-empty one
        here. `caption_source` is what separates an export ref whose caption is
        genuinely empty from a ref whose caption was never going to be present
        — indistinguishable from the caption field alone.
        """
        return self.caption_source == "export" and bool(self.caption and self.caption.strip())


@dataclass(frozen=True, slots=True)
class Media:
    """What a successful download produced."""

    ref: VideoRef
    audio_path: Path
    posted_at: datetime | None = None
    # yt-dlp's description, for sources whose caption doesn't ship in an export.
    description: str | None = None


@dataclass(slots=True)
class Note:
    """The finished note, immediately before rendering to markdown."""

    title: str
    source_url: str
    platform: Platform
    summary: str
    key_points: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    topic: str | None = None
    tags: list[str] = field(default_factory=list)
    author: str | None = None
    posted_at: datetime | None = None
    saved_at: datetime | None = None
    caption: str | None = None
    has_transcript: bool = False
    source_order: int | None = None
    processed_at: datetime | None = None
