"""TikTok `user_data_tiktok.json` favourites parser.

The export is the mirror image of Instagram's: exact save timestamps, no
captions and no authors. That asymmetry is why a failed TikTok download is a
total loss — there is nothing in the export to fall back on.

Path in the JSON: `Likes and Favorites -> Favorite Videos -> FavoriteVideoList`,
entries of the form `{"Date": "2026-08-05 08:36:26", "Link": "…"}`.

**Redirects are deliberately not resolved here.** The plan called for resolving
`tiktokv.com/share/…`, but doing it at parse time means 91 requests to TikTok
just to build a queue — before the pacing logic has any say, and for no benefit,
since yt-dlp follows the redirect itself at download time. The numeric video id
is already present in the share URL, so a stable dedup key needs no network at
all. Parsing stays fully offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from doomnotes.models import VideoRef
from doomnotes.sources.base import canonical_url

DEFAULT_EXPORT = Path("~/Downloads/user_data_tiktok.json").expanduser()

FAVORITES_PATH = ("Likes and Favorites", "Favorite Videos", "FavoriteVideoList")
LIKES_PATH = ("Likes and Favorites", "Like List", "ItemFavoriteList")

SHARE_VIDEO_ID = re.compile(r"/(?:share/)?video/(\d+)")


@dataclass
class Reconciliation:
    entries_found: int = 0
    refs_emitted: int = 0
    dropped: list[tuple[str, str]] = field(default_factory=list)
    unparsed_dates: int = 0
    duplicate_urls: int = 0
    oldest: datetime | None = None
    newest: datetime | None = None
    likes_available: int = 0

    def render(self) -> str:
        lines = [
            "TikTok export reconciliation",
            "-" * 60,
            f"  favourite entries found       : {self.entries_found}",
            f"  refs emitted                  : {self.refs_emitted}",
            f"  dropped                       : {len(self.dropped)}",
        ]
        for url, why in self.dropped:
            lines.append(f"      {why}: {url}")
        lines.append(f"  duplicate urls collapsed      : {self.duplicate_urls}")
        lines.append(f"  unparsable dates              : {self.unparsed_dates}")
        lines.append(f"  newest save                   : {self.newest}")
        lines.append(f"  oldest save                   : {self.oldest}")
        lines.append(
            f"  likes present but out of scope: {self.likes_available}"
        )
        return "\n".join(lines)


def _dig(data: dict, path: tuple[str, ...]) -> list:
    node = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return []
        node = node[key]
    return node if isinstance(node, list) else []


def video_id(url: str) -> str | None:
    """Numeric TikTok video id, extracted from the URL without a network call."""
    m = SHARE_VIDEO_ID.search(url)
    return m.group(1) if m else None


def parse(
    path: str | Path = DEFAULT_EXPORT,
) -> tuple[list[VideoRef], Reconciliation]:
    """Parse favourites. Returns refs in file order, which is newest-first."""
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    rec = Reconciliation()

    entries = _dig(data, FAVORITES_PATH)
    rec.entries_found = len(entries)
    rec.likes_available = len(_dig(data, LIKES_PATH))

    refs: list[VideoRef] = []
    seen: set[str] = set()

    for entry in entries:
        link = (entry.get("Link") or "").strip()
        if not link:
            rec.dropped.append(("<empty>", "no Link field"))
            continue
        if video_id(link) is None:
            rec.dropped.append((link, "no video id in URL"))
            continue

        saved_at: datetime | None = None
        raw_date = (entry.get("Date") or "").strip()
        if raw_date:
            try:
                saved_at = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                rec.unparsed_dates += 1
        else:
            rec.unparsed_dates += 1

        curl = canonical_url(link)
        if curl in seen:
            rec.duplicate_urls += 1
            continue
        seen.add(curl)

        if saved_at is not None:
            rec.newest = max(rec.newest or saved_at, saved_at)
            rec.oldest = min(rec.oldest or saved_at, saved_at)

        refs.append(
            VideoRef(
                url=curl,
                platform="tiktok",
                caption_source="media",
                caption=None,   # never present in this export
                author=None,    # never present in this export
                saved_at=saved_at,
                source_order=len(refs),
            )
        )

    rec.refs_emitted = len(refs)
    return refs, rec


class TikTokExportSource:
    name = "tiktok_export"

    def __init__(self, path: str | Path = DEFAULT_EXPORT) -> None:
        self.path = Path(path).expanduser()

    def fetch(self) -> Iterator[VideoRef]:
        refs, _ = parse(self.path)
        yield from refs

    def fetch_with_reconciliation(self) -> tuple[list[VideoRef], Reconciliation]:
        return parse(self.path)
