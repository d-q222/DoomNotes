"""Download + failure-taxonomy tests.

    # REAL_ERRORS below is the corpus: actual yt-dlp stderr strings, each with
    # the outcome it should map to. The xfail tests show what the deliberately
    # naive classifier gets wrong; implementing a real taxonomy in
    # download.classify() makes them pass and the markers removable.
    #
    # The checkpoint case is already correct: it must never be retried, because
    # retrying is what escalates it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doomnotes.download import (
    DownloadResult,
    failure_reason,
    Outcome,
    RunnerResult,
    build_command,
    classify,
    download,
    fetch_url,
    sleep_seconds,
)
from doomnotes.models import VideoRef

IG = VideoRef("https://www.instagram.com/reel/AAA/", "instagram", caption_source="export", caption="c")
TT = VideoRef("https://www.tiktokv.com/share/video/111/", "tiktok")

# (stderr, expected_outcome, note)
REAL_ERRORS = [
    ("ERROR: [Instagram] AAA: Video unavailable", Outcome.TERMINAL, "deleted"),
    ("ERROR: [Instagram] AAA: This post is private", Outcome.TERMINAL, "private"),
    ("ERROR: [TikTok] 111: content isn't available", Outcome.TERMINAL, "region/removed"),
    ("ERROR: Unsupported URL: https://example.com/x", Outcome.TERMINAL, "not fetchable"),
    ("ERROR: unable to download video data: HTTP Error 404: Not Found", Outcome.TERMINAL, "404"),
    ("ERROR: unable to download webpage: HTTP Error 429: Too Many Requests", Outcome.RETRYABLE, "rate limited"),
    ("ERROR: unable to download webpage: HTTP Error 503: Service Unavailable", Outcome.RETRYABLE, "5xx"),
    ("ERROR: Unable to download webpage: <urlopen error timed out>", Outcome.RETRYABLE, "network"),
    ("ERROR: [Instagram] Requested content is not available, login_required", Outcome.STOP, "session dead"),
    ("ERROR: [Instagram] challenge_required: checkpoint", Outcome.STOP, "checkpoint"),
]


def test_success_is_ok() -> None:
    assert classify(0, "") is Outcome.OK


@pytest.mark.parametrize(
    "stderr,expected,note",
    [c for c in REAL_ERRORS if c[1] is Outcome.STOP and "checkpoint" in c[0]],
)
def test_checkpoint_always_stops(stderr: str, expected: Outcome, note: str) -> None:
    """Ships correct: retrying a checkpoint is how accounts get flagged."""
    assert classify(1, stderr) is Outcome.STOP


@pytest.mark.parametrize(
    "stderr,expected,note",
    [
        pytest.param(stderr, expected, note, id=note)
        for stderr, expected, note in REAL_ERRORS
        if expected is not Outcome.STOP
    ],
)
def test_classify_real_yt_dlp_errors(stderr: str, expected: Outcome, note: str) -> None:
    assert classify(1, stderr) is expected


def test_an_unrecognised_error_is_retryable_not_terminal() -> None:
    """The default that cannot lose a video.

    TikTok's export ships no caption, so calling a transient failure terminal
    loses the video permanently with nothing to fall back on. Calling a
    terminal failure retryable only costs a wasted request.
    """
    assert classify(1, "ERROR: something nobody has seen before") is Outcome.RETRYABLE


def test_a_dead_session_stops_the_run() -> None:
    """Continuing on a dead session spends the whole batch on 401s.

    Worse, on Instagram it looks identical to a run where every video happens
    to be private — so the failures land in the store as if the videos were the
    problem.
    """
    stderr = next(c[0] for c in REAL_ERRORS if "login_required" in c[0])
    assert classify(1, stderr) is Outcome.STOP


# ── the wrapper itself (SUPERVISE — these should pass) ───────────────────


def test_build_command_never_embeds_cookie_values(tmp_path: Path) -> None:
    """A cookie must reach yt-dlp as a path, never as a value on the CLI."""
    secret = tmp_path / "cookies.txt"
    secret.write_text("sessionid=SUPERSECRETVALUE")  # doomnotes:allow-fixture
    cmd = build_command(IG, tmp_path, {"mode": "file", "cookies_file": str(secret)})
    joined = " ".join(cmd)
    assert "SUPERSECRETVALUE" not in joined
    assert "--cookies" in cmd and str(secret) in cmd


def test_build_command_browser_mode(tmp_path: Path) -> None:
    cmd = build_command(IG, tmp_path, {"mode": "browser", "browser": "chrome"})
    assert "--cookies-from-browser" in cmd and "chrome" in cmd


def test_build_command_per_platform_auth_can_differ(tmp_path: Path) -> None:
    """Tier-1 #1: TikTok may land on a different answer than Instagram."""
    ig = build_command(IG, tmp_path, {"mode": "browser", "browser": "chrome"})
    tt = build_command(TT, tmp_path, {"mode": "none"})
    assert "--cookies-from-browser" in ig
    assert "--cookies-from-browser" not in tt


def test_download_failure_never_raises(tmp_path: Path) -> None:
    def boom(cmd, timeout):
        raise OSError("no such binary")

    result = download(IG, tmp_path, {}, runner=boom)
    assert isinstance(result, DownloadResult)
    assert result.outcome is Outcome.RETRYABLE


def test_download_reports_missing_audio_as_failure(tmp_path: Path) -> None:
    """yt-dlp exiting 0 without producing a file must not look like success."""
    def liar(cmd, timeout):
        return RunnerResult(0, '{"id": "AAA", "title": "t"}', "")

    result = download(IG, tmp_path, {}, runner=liar)
    assert result.outcome is Outcome.RETRYABLE
    assert "no audio file" in (result.error or "")


def test_download_extracts_posted_at(tmp_path: Path) -> None:
    audio = tmp_path / "AAA.m4a"

    def ok(cmd, timeout):
        audio.write_bytes(b"\x00")
        return RunnerResult(0, '{"id": "AAA", "upload_date": "20260714", "title": "t"}', "")

    result = download(IG, tmp_path, {}, runner=ok)
    assert result.outcome is Outcome.OK
    assert result.media is not None
    assert result.media.posted_at is not None
    assert result.media.posted_at.year == 2026


# ── identity URL vs fetch URL ────────────────────────────────────────────


def test_instagram_is_requested_exactly_as_the_export_gave_it() -> None:
    """Instagram's export URLs are already canonical. Do not touch them."""
    assert fetch_url(IG) == IG.url


def test_tiktok_is_requested_via_the_form_yt_dlp_actually_supports() -> None:
    """The export's host has no yt-dlp extractor; this form matches TikTokIE.

    Checked against yt-dlp 2026.07.04's extractor table rather than assumed:
    `www.tiktokv.com/share/video/<id>/` matches nothing and falls through to
    the generic extractor, which has to fetch and follow a redirect. The id is
    already in the URL, so the supported form costs no network to build — and
    `@_` is yt-dlp's own placeholder for an unknown uploader.
    """
    assert fetch_url(TT) == "https://www.tiktok.com/@_/video/111"


def test_the_fetch_url_does_not_become_the_refs_identity() -> None:
    """The store keys on `ref.url`. Rewriting it would orphan every existing row."""
    before = TT.url
    fetch_url(TT)
    assert TT.url == before


def test_an_unparseable_tiktok_url_is_passed_through_unchanged() -> None:
    """Better to let yt-dlp report a real error than to invent a URL."""
    odd = VideoRef("https://www.tiktok.com/t/ZTRabcdef/", "tiktok")
    assert fetch_url(odd) == odd.url


def test_build_command_requests_the_fetch_url(tmp_path: Path) -> None:
    cmd = build_command(TT, tmp_path, {"mode": "none"})
    assert cmd[-1] == "https://www.tiktok.com/@_/video/111"


def test_sleep_stays_within_configured_bounds() -> None:
    import random

    rng = random.Random(0)
    for _ in range(200):
        assert 20.0 <= sleep_seconds(20, 90, rng) <= 90.0


# ── descriptive failure reasons (3.2) ────────────────────────────────────


@pytest.mark.parametrize(
    "stderr,expected",
    [
        ("ERROR: [Instagram] AAA: Video unavailable", "deleted"),
        ("ERROR: [Instagram] AAA: This post is private", "private"),
        ("ERROR: [TikTok] 111: content isn't available", "unavailable_here"),
        ("ERROR: unable to download video data: HTTP Error 404: Not Found", "not_found"),
        ("ERROR: unable to download webpage: HTTP Error 429: Too Many Requests", "rate_limited"),
        ("ERROR: unable to download webpage: HTTP Error 503: Service Unavailable", "server_error"),
        ("ERROR: Unable to download webpage: <urlopen error timed out>", "timeout"),
        ("ERROR: [Instagram] Requested content is not available, login_required", "session_dead"),
        ("ERROR: [Instagram] challenge_required: checkpoint", "checkpoint"),
        ("ERROR: something nobody has seen before", "unknown"),
    ],
)
def test_failure_reason_is_descriptive(stderr: str, expected: str) -> None:
    """'the video failed' is not a diagnosis. The store records this instead."""
    assert failure_reason(stderr) == expected


@pytest.mark.parametrize("stderr,expected,note", REAL_ERRORS)
def test_reason_and_outcome_cannot_disagree(stderr: str, expected: Outcome, note: str) -> None:
    """Both read the same tables, so a reason always implies its own bucket.

    If these could drift, a failure could be reported as "deleted" while being
    retried, or as "rate_limited" while being written off.
    """
    reason = failure_reason(stderr)
    assert reason != "unknown", f"{note} should have a named reason"
    assert classify(1, stderr) is expected


@pytest.mark.parametrize(
    "stderr,reason",
    [
        ("ERROR: unable to download webpage: HTTP Error 401: Unauthorized", "unauthorized"),
        ("ERROR: [Instagram] Login required to access this content", "session_dead"),
        ("ERROR: [Instagram] redirected to /consent/", "consent_redirect"),
    ],
)
def test_an_auth_failure_halts_rather_than_being_retried(stderr: str, reason: str) -> None:
    """A dead session is not a property of the video.

    Retried instead of halted, it spends the whole batch on identical 401s and
    files each one in the store as though the video were the problem. Only the
    `login_required` token form was matched, so the prose form, the bare HTTP
    status and a consent redirect all fell through to RETRYABLE.
    """
    assert classify(1, stderr) is Outcome.STOP
    assert failure_reason(stderr) == reason


def test_an_ambiguous_403_stays_retryable() -> None:
    """403 is geo-blocking as often as it is auth, and the default is the one
    that cannot lose a video. Pinned so widening STOP stays deliberate."""
    stderr = "ERROR: unable to download webpage: HTTP Error 403: Forbidden"
    assert classify(1, stderr) is Outcome.RETRYABLE
