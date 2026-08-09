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
    Outcome,
    RunnerResult,
    build_command,
    classify,
    download,
    sleep_seconds,
)
from doomnotes.models import VideoRef

IG = VideoRef("https://www.instagram.com/reel/AAA/", "instagram", caption="c")
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


# The naive classifier calls every non-zero exit RETRYABLE, so the RETRYABLE
# rows of the table pass — for the wrong reason. A table-wide xfail therefore
# has to be non-strict, which is the same defect `test_store.py` documents:
# a marker that cannot distinguish "not implemented" from "accidentally right".
#
# Marking per case fixes it. Only the rows the baseline actually gets wrong are
# xfail(strict=True); the rest are ordinary passing tests. When classify() grows
# a real taxonomy, every marker below comes off together and nothing xpasses.
NAIVE_CLASSIFIER = pytest.mark.xfail(
    reason="HANDS-ON #3.2: the naive classifier treats every non-zero exit as "
           "RETRYABLE, so a deleted video is retried three times against a "
           "rate limit that matters.",
    strict=True,
)


@pytest.mark.parametrize(
    "stderr,expected,note",
    [
        pytest.param(
            stderr, expected, note,
            id=note,
            marks=[NAIVE_CLASSIFIER] if expected is Outcome.TERMINAL else [],
        )
        for stderr, expected, note in REAL_ERRORS
        if expected is not Outcome.STOP
    ],
)
def test_classify_real_yt_dlp_errors(stderr: str, expected: Outcome, note: str) -> None:
    assert classify(1, stderr) is expected


@pytest.mark.xfail(
    reason="HANDS-ON #3.2: `login_required` means the session is dead. The "
           "STOP carve-out only matches checkpoint/challenge_required/captcha, "
           "so a dead session is retried instead of halting the run.",
    strict=True,
)
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


def test_sleep_stays_within_configured_bounds() -> None:
    import random

    rng = random.Random(0)
    for _ in range(200):
        assert 20.0 <= sleep_seconds(20, 90, rng) <= 90.0
