"""yt-dlp wrapper: audio extraction, metadata, failure classification, pacing.

The subprocess call is injectable (`runner=`) so the pipeline can be exercised
end to end with zero platform traffic. That is how the offline spine test runs.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from doomnotes.models import Media, VideoRef


class Outcome(StrEnum):
    OK = "ok"
    RETRYABLE = "retryable"
    TERMINAL = "terminal"
    STOP = "stop"          # checkpoint / captcha — halt the run, notify, do not retry


@dataclass
class DownloadResult:
    ref: VideoRef
    outcome: Outcome
    media: Media | None = None
    error: str | None = None
    # Descriptive tag for a non-OK outcome: "deleted", "rate_limited", ...
    reason: str | None = None


@dataclass
class RunnerResult:
    returncode: int
    stdout: str
    stderr: str


class Runner(Protocol):
    def __call__(self, cmd: list[str], timeout: int) -> RunnerResult: ...


def subprocess_runner(cmd: list[str], timeout: int) -> RunnerResult:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    return RunnerResult(proc.returncode, proc.stdout, proc.stderr)


# ── Failure taxonomy (decision 3.2) ──────────────────────────────────────
# Classification is on yt-dlp's stderr, because the exit code is 1 for every
# failure and carries no information.
#
# The default for an unrecognised error is RETRYABLE, not TERMINAL, and the
# asymmetry is why: a TikTok export ships no caption, so calling a transient
# failure terminal loses that video permanently with nothing to fall back on.
# Calling a terminal failure retryable only costs a wasted request.
# ──────────────────────────────────────────────────────────────────────────

# (marker, reason). The reason is the descriptive tag recorded in the store and
# the run journal, so a failure reads as "deleted" or "rate_limited" rather than
# as a bucket name. Match order within a table is the order listed.

# Halt the whole run. A challenge is escalated by retrying it, and a dead
# session turns the rest of the batch into identical 401s that land in the
# store as though the videos were the problem.
STOP_REASONS = (
    ("checkpoint", "checkpoint"),
    ("challenge_required", "challenge_required"),
    ("captcha", "captcha"),
    ("login_required", "session_dead"),
    ("rate-limit reached", "rate_limit_hard"),
    ("/accounts/login", "login_redirect"),
)

# Deliberately NOT here: bare "http error 401", the prose "login required", and
# "/consent".
#
# STOP is not a per-video outcome. `pipeline.run` breaks without calling
# mark_failed, so the triggering URL stays PENDING and is offered FIRST on the
# next run. A per-video condition misclassified as STOP therefore wedges the
# queue permanently: every future run halts on the same video and nothing after
# it is ever attempted. On TikTok, with no caption to salvage, that loses the
# entire remaining backlog rather than one video.
#
# All three of those are per-video as often as they are session-wide. A single
# private or geo-restricted video can return 401; one consent interstitial can
# appear for one fetch. That is the same ambiguity for which 403 is deliberately
# left RETRYABLE, and it deserves the same answer.
#
# A genuinely dead session shows up as REPEATED failures across videos, not as
# one error string. Detecting that is decision 7.1 (systemic outage vs item
# failure), which has its own xfail. The classifier cannot see across videos and
# should not pretend to.

# The video is gone or was never fetchable. Retrying spends requests against a
# rate limit that matters and cannot succeed.
TERMINAL_REASONS = (
    ("video unavailable", "deleted"),
    ("this post is private", "private"),
    ("content isn't available", "unavailable_here"),
    ("unsupported url", "unsupported_url"),
    ("http error 404", "not_found"),
    ("http error 410", "gone"),
)

# Named retryable causes. Anything unmatched is retryable too, as "unknown" --
# these exist to describe the failure, not to decide it.
RETRYABLE_REASONS = (
    ("http error 429", "rate_limited"),
    ("http error 5", "server_error"),
    ("timed out", "timeout"),
    ("timeout", "timeout"),
    ("name resolution", "dns"),
    ("nodename nor servname", "dns"),
    # Generic, so it comes last: "unable to download webpage: <urlopen error
    # timed out>" contains both this and the timeout marker, and "timeout" is
    # the more useful diagnosis.
    ("unable to download webpage", "fetch_failed"),
)

# Kept as flat tuples: several callers and tests ask "is this a stop marker?"
STOP_MARKERS = tuple(marker for marker, _ in STOP_REASONS)
TERMINAL_MARKERS = tuple(marker for marker, _ in TERMINAL_REASONS)




def _match(stderr: str) -> tuple[Outcome, str]:
    """Outcome and descriptive reason from one pass, so they cannot disagree.

    Order is load-bearing. STOP is checked first because a dead session's
    message also contains wording that reads as terminal, and halting is always
    the safer reading. TERMINAL is checked before the retryable default so an
    explicit 404 is not swept up by the fallback.
    """
    lowered = stderr.lower()
    for table, outcome in (
        (STOP_REASONS, Outcome.STOP),
        (TERMINAL_REASONS, Outcome.TERMINAL),
        (RETRYABLE_REASONS, Outcome.RETRYABLE),
    ):
        for marker, reason in table:
            if marker in lowered:
                return outcome, reason
    # Everything else is retryable, including errors not seen before. The cost
    # is asymmetric: a TikTok export ships no caption, so calling a transient
    # failure terminal loses that video permanently with nothing to fall back
    # on, while calling a terminal failure retryable costs one wasted request.
    return Outcome.RETRYABLE, "unknown"


def classify(returncode: int, stderr: str) -> Outcome:
    """Map a yt-dlp exit to an outcome. See `_match` for the ordering rule."""
    if returncode == 0:
        return Outcome.OK
    return _match(stderr)[0]


def failure_reason(stderr: str) -> str:
    """The descriptive tag for a failure, e.g. "deleted", "rate_limited".

    Always agrees with `classify` because both read the same tables.
    """
    return _match(stderr)[1]


# ── DELIBERATELY NAIVE: pacing ───────────────────────────────────────────
# CURRENT: uniform random sleep between sleep_min_s and sleep_max_s, applied
#   identically to both platforms; queues drained strictly in the configured
#   order, Instagram fully before TikTok.
#
# WHY THAT IS INSUFFICIENT: a uniform distribution does not resemble human
#   browsing — no short bursts, no long gaps, so the inter-request histogram
#   is flatter than any real user's. It also sleeps as long after a *failed*
#   request as a successful one, spending the run's budget on videos deleted
#   years ago. And draining Instagram entirely first means TikTok's oldest
#   backlog — the most likely to have rotted away — is touched last, when it
#   teaches least.
#
# INTENDED: a distribution with bursts and gaps, failure-aware sleeping, and
#   a deliberate split between the two queues. This is a risk decision about
#   a real account, so it is left explicit rather than guessed.
# ──────────────────────────────────────────────────────────────────────────


def sleep_seconds(cfg_min: float, cfg_max: float, rng: random.Random | None = None) -> float:
    """Uniform, deliberately. See the block above."""
    r = rng or random
    return r.uniform(cfg_min, cfg_max)


TIKTOK_VIDEO_ID = re.compile(r"/(?:share/)?video/(\d+)")


def fetch_url(ref: VideoRef) -> str:
    """The URL to hand yt-dlp, which is not always the ref's identity URL.

    The ref's `url` is the store's primary key and stays exactly as the export
    gave it. This is only what gets requested.

    Why they differ for TikTok: the favourites export stores links as
    `www.tiktokv.com/share/video/<id>/`, and yt-dlp 2026.07.04 has **no
    extractor matching that host** — checked against its extractor table, not
    assumed. It falls through to the generic extractor, which fetches the URL,
    follows the redirect and re-dispatches. That costs an extra request per
    video and depends on the redirect landing on a page the TikTok extractor
    recognises rather than on a login or consent interstitial.

    The id is already in the URL, so the supported form is constructible with
    no network at all. `www.tiktok.com/@<user>/video/<id>` matches TikTokIE,
    and the handle is optional — yt-dlp's own `TikTokBaseIE._create_url` builds
    `https://www.tiktok.com/@{user_id or "_"}/video/{video_id}` when it does not
    know the uploader, which is exactly our situation. So this is yt-dlp's
    convention, not an invention of ours.

    It matters asymmetrically: a failed TikTok download has no export caption to
    fall back on, so it produces no note at all. Spending 91 first-batch
    requests on a path that is merely probable is the expensive kind of wrong.

    To revert: return `ref.url` unconditionally. Nothing else depends on this.
    """
    if ref.platform != "tiktok":
        return ref.url
    match = TIKTOK_VIDEO_ID.search(ref.url)
    return f"https://www.tiktok.com/@_/video/{match.group(1)}" if match else ref.url


def build_command(
    ref: VideoRef,
    audio_dir: Path,
    auth: dict,
    audio_format: str = "m4a",
) -> list[str]:
    """Construct the yt-dlp invocation.

    Cookies are passed as a FILE PATH or a browser name — never as a value, so
    nothing sensitive lands in the process table or shell history.
    """
    out_template = str(audio_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-progress",
        "--print-json",
        "--no-simulate",
        "-f", "ba/b",
        "-x", "--audio-format", audio_format,
        "-o", out_template,
    ]

    mode = (auth or {}).get("mode", "none")
    if mode == "browser":
        cmd += ["--cookies-from-browser", auth.get("browser", "chrome")]
    elif mode == "file":
        cookie_path = Path(str(auth.get("cookies_file", ""))).expanduser()
        if cookie_path.is_file():
            cmd += ["--cookies", str(cookie_path)]

    cmd.append(fetch_url(ref))
    return cmd


def _parse_metadata(stdout: str) -> dict:
    """yt-dlp --print-json emits one JSON object per line."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def _posted_at(meta: dict) -> datetime | None:
    ts = meta.get("timestamp")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts)
    upload = meta.get("upload_date")
    if isinstance(upload, str) and len(upload) == 8:
        try:
            return datetime.strptime(upload, "%Y%m%d")
        except ValueError:
            return None
    return None


def download(
    ref: VideoRef,
    audio_dir: Path,
    auth: dict,
    *,
    timeout_s: int = 180,
    audio_format: str = "m4a",
    runner: Runner = subprocess_runner,
) -> DownloadResult:
    """Fetch audio + metadata for one ref. Never raises for a failed download."""
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("yt-dlp") is None and runner is subprocess_runner:
        return DownloadResult(
            ref, Outcome.STOP, error="yt-dlp not installed", reason="yt_dlp_missing"
        )

    cmd = build_command(ref, audio_dir, auth, audio_format)
    try:
        result = runner(cmd, timeout_s)
    except subprocess.TimeoutExpired:
        return DownloadResult(
            ref, Outcome.RETRYABLE, error=f"timeout after {timeout_s}s", reason="timeout"
        )
    except Exception as exc:  # noqa: BLE001 - a source must never crash the run
        return DownloadResult(
            ref, Outcome.RETRYABLE, error=f"{type(exc).__name__}: {exc}", reason="runner_error"
        )

    outcome = classify(result.returncode, result.stderr)
    if outcome is not Outcome.OK:
        tail = (result.stderr or "").strip().splitlines()
        return DownloadResult(
            ref,
            outcome,
            error=tail[-1] if tail else "unknown error",
            reason=failure_reason(result.stderr or ""),
        )

    meta = _parse_metadata(result.stdout)
    audio_path = _locate_audio(meta, audio_dir, audio_format)
    if audio_path is None:
        return DownloadResult(
            ref,
            Outcome.RETRYABLE,
            error="yt-dlp reported success but no audio file found",
            reason="no_audio_file",
        )

    return DownloadResult(
        ref,
        Outcome.OK,
        media=Media(
            ref=ref,
            audio_path=audio_path,
            posted_at=_posted_at(meta),
            description=meta.get("description"),
        ),
    )


def _locate_audio(meta: dict, audio_dir: Path, audio_format: str) -> Path | None:
    candidates = []
    vid = meta.get("id")
    if vid:
        candidates.append(audio_dir / f"{vid}.{audio_format}")
    for key in ("filepath", "_filename"):
        value = meta.get(key)
        if value:
            candidates.append(Path(value))
    for c in candidates:
        if c.is_file():
            return c
    return None


