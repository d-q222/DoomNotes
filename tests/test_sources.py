"""Source adapter tests. Uses a synthetic export fixture, not real export data.

Every URL in this file is invented (doomnotes:synthetic-urls). None of them
come from the real exports, which is why the pre-commit bulk-URL guard is
told to skip this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doomnotes.sources import ig_export, manual, tiktok_export
from doomnotes.sources.base import canonical_url, is_instagram_post_url

# A miniature export reproducing the structure that actually matters: nested
# Owner and Hashtags sub-blocks that BOTH use a "Name" label.
IG_FIXTURE = """
<html><body><main>
<table style="table-layout: fixed;">
  <tr><td colspan="2" class="_a6_q">URL<div><a href="https://www.instagram.com/reel/AAAA1111/">x</a></div></td></tr>
  <tr><td class="_a6_q">Caption</td><td class="_a6_r">First caption</td></tr>
  <tr><td colspan="2" class="_a6_q"><div><div><h2>Owner</h2><div><div><table>
        <tr><td colspan="2"><div><div><table>
            <tr><td>URL</td><td>https://creator.example.com</td></tr>
            <tr><td>Name</td><td>Real Creator Name</td></tr>
            <tr><td>Username</td><td>real.creator</td></tr>
        </table></div></div></td></tr>
  </table></div></div></div></td></tr>
  <tr><td colspan="2" class="_a6_q"><div><div><h2>Hashtags</h2><div>
        <div class="_a6_q">Name</div><div><div><div>decoyhashtag</div></div></div>
  </div></div></div></td></tr>
</table>
<table style="table-layout: fixed;">
  <tr><td colspan="2" class="_a6_q">URL<div><a href="https://www.instagram.com/p/BBBB2222/">y</a></div></td></tr>
  <tr><td class="_a6_q">Caption</td><td class="_a6_r">Second caption</td></tr>
</table>
<table style="table-layout: fixed;">
  <tr><td colspan="2" class="_a6_q">URL<div><a href="https://www.instagram.com/channel/CCCC3333/">z</a></div></td></tr>
</table>
</main></body></html>
"""


@pytest.fixture()
def ig_file(tmp_path: Path) -> Path:
    p = tmp_path / "saved_posts_fixture.html"
    p.write_text(IG_FIXTURE, encoding="utf-8")
    return p


def test_ig_parses_posts_and_drops_channel(ig_file: Path) -> None:
    refs, rec = ig_export.parse(ig_file)
    assert [r.url for r in refs] == [
        "https://www.instagram.com/reel/AAAA1111/",
        "https://www.instagram.com/p/BBBB2222/",
    ]
    assert rec.refs_emitted == 2
    assert len(rec.dropped) == 1
    assert "channel" in rec.dropped[0][0]


def test_ig_author_comes_from_owner_block_not_hashtags(ig_file: Path) -> None:
    """The bug this parser exists to avoid: a hashtag becoming the author."""
    refs, _ = ig_export.parse(ig_file)
    assert refs[0].author == "@real.creator"
    assert refs[0].author != "@decoyhashtag"


def test_ig_preserves_file_order_as_source_order(ig_file: Path) -> None:
    """File order is newest-saved-first; source_order is the recency proxy."""
    refs, _ = ig_export.parse(ig_file)
    assert [r.source_order for r in refs] == [0, 1]


def test_ig_refs_have_export_caption(ig_file: Path) -> None:
    """This is what makes a caption-only fallback possible for Instagram."""
    refs, _ = ig_export.parse(ig_file)
    assert all(r.has_export_caption for r in refs)


def test_ig_post_without_owner_block_still_parses(ig_file: Path) -> None:
    refs, _ = ig_export.parse(ig_file)
    assert refs[1].author is None
    assert refs[1].caption == "Second caption"


# ── TikTok ───────────────────────────────────────────────────────────────

TT_FIXTURE = {
    "Likes and Favorites": {
        "Favorite Videos": {
            "FavoriteVideoList": [
                {"Date": "2026-08-05 08:36:26",
                 "Link": "https://www.tiktokv.com/share/video/7000000000000000001/"},
                {"Date": "2021-07-14 22:00:14",
                 "Link": "https://www.tiktokv.com/share/video/7000000000000000002/"},
                {"Date": "bad date", "Link": "https://www.tiktokv.com/share/video/1/"},
                {"Date": "2024-01-01 00:00:00", "Link": "https://example.com/nope"},
            ]
        },
        "Like List": {"ItemFavoriteList": [{"Date": "x", "Link": "y"}] * 7},
    }
}


@pytest.fixture()
def tt_file(tmp_path: Path) -> Path:
    p = tmp_path / "user_data_fixture.json"
    p.write_text(json.dumps(TT_FIXTURE), encoding="utf-8")
    return p


def test_tiktok_parses_favorites(tt_file: Path) -> None:
    refs, rec = tiktok_export.parse(tt_file)
    assert rec.entries_found == 4
    assert rec.refs_emitted == 3
    assert len(rec.dropped) == 1  # the non-video URL


def test_tiktok_never_has_caption_or_author(tt_file: Path) -> None:
    """The asymmetry that makes a failed TikTok download unrecoverable."""
    refs, _ = tiktok_export.parse(tt_file)
    assert all(r.caption is None and r.author is None for r in refs)
    assert not any(r.has_export_caption for r in refs)


def test_tiktok_keeps_save_timestamps(tt_file: Path) -> None:
    refs, rec = tiktok_export.parse(tt_file)
    assert refs[0].saved_at is not None
    assert refs[0].saved_at.year == 2026
    assert rec.unparsed_dates == 1
    assert rec.newest.year == 2026 and rec.oldest.year == 2021


def test_tiktok_reports_likes_without_parsing_them(tt_file: Path) -> None:
    _, rec = tiktok_export.parse(tt_file)
    assert rec.likes_available == 7


def test_tiktok_parsing_makes_no_network_call(tt_file: Path) -> None:
    """Video ids come from the URL; redirects are yt-dlp's job at fetch time."""
    assert tiktok_export.video_id(
        "https://www.tiktokv.com/share/video/7000000000000000001/"
    ) == "7000000000000000001"


# ── shared helpers ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.instagram.com/reel/ABC123/", True),
        ("https://www.instagram.com/p/ABC123/", True),
        ("https://www.instagram.com/tv/ABC123/", True),
        ("https://www.instagram.com/channel/ABC123/", False),
        ("https://example.com/reel/ABC123/", False),
    ],
)
def test_instagram_post_whitelist(url: str, expected: bool) -> None:
    assert is_instagram_post_url(url) is expected


def test_canonical_url_strips_share_tokens() -> None:
    """Instagram appends ?igsh=… per copy of a link; the store must not see it."""
    a = canonical_url("https://www.instagram.com/reel/ABC123/?igsh=xyz123")
    b = canonical_url("https://www.instagram.com/reel/ABC123")
    assert a == b == "https://www.instagram.com/reel/ABC123/"


def test_manual_drops_unsupported_and_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "urls.txt"
    p.write_text(
        "# comment\n"
        "https://www.instagram.com/reel/AAA111/\n"
        "\n"
        "garbage\n"
        "https://www.instagram.com/reel/AAA111/\n"
    )
    refs, dropped = manual.parse(p)
    assert len(refs) == 1
    assert {why for _, why in dropped} == {"unsupported or malformed URL", "duplicate"}


def test_manual_detects_platform(tmp_path: Path) -> None:
    p = tmp_path / "urls.txt"
    p.write_text(
        "https://www.instagram.com/reel/AAA111/\n"
        "https://www.tiktokv.com/share/video/123456789/\n"
    )
    refs, _ = manual.parse(p)
    assert [r.platform for r in refs] == ["instagram", "tiktok"]
