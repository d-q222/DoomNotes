"""The seam every input adapter plugs into.

    # ── The extension point ─────────────────────────────────────────────────
    # The VideoRef half of this contract is in models.py.
    #
    # A new source must be implementable against this protocol and change
    # nothing else in the codebase. If adding one requires editing pipeline.py,
    # the protocol was drawn in the wrong place — that is the test.
    # ────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

__all__ = ["canonical_url", "INSTAGRAM_POST_PATH", "is_instagram_post_url"]

# Only these Instagram path types are fetchable posts. The export also contains
# a /channel/ entry, which is not a post at all — whitelisting here means it is
# *dropped with a reason* rather than becoming a fake download failure later.
INSTAGRAM_POST_PATH = re.compile(r"^/(reel|reels|p|tv)/([A-Za-z0-9_-]+)/?")


def is_instagram_post_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if not parts.netloc.endswith("instagram.com"):
        return False
    return bool(INSTAGRAM_POST_PATH.match(parts.path))


def canonical_url(url: str) -> str:
    """Normalise a URL so the store's dedup key is stable.

    Drops query strings and fragments (Instagram appends `?igsh=…` share tokens
    that differ per copy of the same link) and normalises the trailing slash.
    Deliberately does NOT resolve redirects — that is a network call, and the
    store must be usable offline.
    """
    parts = urlsplit(url.strip())
    path = parts.path
    if not path.endswith("/"):
        path += "/"
    return urlunsplit((parts.scheme or "https", parts.netloc, path, "", ""))
