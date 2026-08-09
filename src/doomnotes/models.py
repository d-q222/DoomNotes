"""Core data shapes that flow between pipeline stages.

    # ── DELIBERATELY PERMISSIVE: VideoRef field optionality ─────────────────
    # CURRENT: every source-specific field is Optional; only `url` and
    #   `platform` are required.
    #
    # WHY THAT IS INSUFFICIENT: "everything is optional" pushes the problem
    #   downstream. Each consumer must re-derive which field combinations are
    #   actually possible, and nothing prevents a source emitting a ref no
    #   later stage can use. The type asserts nothing true about the data.
    #
    #   The real shape is that the two sources have OPPOSITE gaps:
    #     Instagram : caption ✓  author ✓  saved_at ✗  source_order ✓
    #     TikTok    : caption ✗  author ✗  saved_at ✓  source_order ✓
    #   That asymmetry is load-bearing — it is why a failed TikTok download is
    #   unrecoverable while a failed Instagram one still yields a caption-only
    #   note. The type does not currently encode it.
    #
    # INTENDED: either one permissive struct validated at the seam, or distinct
    #   types a consumer must narrow. A browser-scraper source would fail like
    #   TikTok rather than like the Instagram export, since its caption arrives
    #   with the media.
    # ────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

Platform = Literal["instagram", "tiktok"]


@dataclass(frozen=True, slots=True)
class VideoRef:
    """One saved video, as it leaves a source and before anything is fetched.

    `url` is the identity of a ref everywhere in the system — the store keys on
    it, dedup keys on it, and it is the only field guaranteed to be present.
    """

    url: str
    platform: Platform

    # Instagram export only: the caption ships in the export itself, which is
    # what makes a caption-only fallback note possible when the download fails.
    caption: str | None = None
    author: str | None = None

    # TikTok export only: an exact favourite timestamp.
    saved_at: datetime | None = None

    # Position in the source's own newest-first ordering. Present for both
    # exports; it is Instagram's *only* recency signal.
    source_order: int | None = None

    @property
    def has_export_caption(self) -> bool:
        """Whether a failed download can still produce a note.

        True only for Instagram export refs. TikTok, manual-list and scraper
        refs get their caption from yt-dlp metadata — the same call that fetches
        the media — so if the download fails there is nothing left.
        """
        return bool(self.caption and self.caption.strip())


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
