"""Manual URL-list source — one URL per line, `#` for comments.

This is the source the spine test runs against, and the one to reach for when
processing a single video without touching an export.

Like TikTok (and, next weekend, Playwright), a manual ref carries no caption:
it arrives with yt-dlp metadata, from the same call that fetches the media. A
failed download therefore yields no note.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

from doomnotes.models import Platform, VideoRef
from doomnotes.sources.base import canonical_url


def detect_platform(url: str) -> Platform | None:
    host = urlsplit(url).netloc.lower()
    if "instagram.com" in host:
        return "instagram"
    if "tiktok.com" in host or "tiktokv.com" in host:
        return "tiktok"
    return None


def parse(path: str | Path) -> tuple[list[VideoRef], list[tuple[str, str]]]:
    """Returns (refs, dropped) where dropped is [(line, reason)]."""
    refs: list[VideoRef] = []
    dropped: list[tuple[str, str]] = []
    seen: set[str] = set()

    text = Path(path).expanduser().read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        platform = detect_platform(line)
        if platform is None:
            # A malformed or unsupported URL is dropped at the source, not
            # handed downstream to fail as a phantom download error.
            dropped.append((line, "unsupported or malformed URL"))
            continue

        curl = canonical_url(line)
        if curl in seen:
            dropped.append((line, "duplicate"))
            continue
        seen.add(curl)

        refs.append(
            VideoRef(
                url=curl,
                platform=platform,
                # Same position as TikTok even for an instagram URL: the
                # caption arrives with the media, not with the list.
                caption_source="media",
                source_order=len(refs),
            )
        )
    return refs, dropped


class ManualListSource:
    name = "manual"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def fetch(self) -> Iterator[VideoRef]:
        refs, _ = parse(self.path)
        yield from refs
