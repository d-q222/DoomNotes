"""Orchestration: refs -> download -> transcribe -> summarize -> vault.

Every collaborator is injected, which is what makes the offline spine test
possible: swap the downloader for a fixture and the whole pipeline runs with
zero platform traffic.

    # ── DELIBERATELY NAIVE: failure isolation granularity ───────────────────
    # CURRENT: one try/except around each video. A video that throws anywhere
    #   is logged and the loop continues.
    #
    # WHY THAT IS INSUFFICIENT: "the video failed" is not a diagnosis. A
    #   yt-dlp 404, an ffmpeg crash, an out-of-memory transcription and a
    #   malformed model response all land in the same bucket with the same
    #   store state, so the failure log cannot identify which stage is broken.
    #   At a few hundred videos that is the difference between fixing one
    #   thing and re-running everything.
    #
    #   Concretely: if the summarizer dies at video 12, this loop dutifully
    #   logs the rest as identical failures, marks them processed, and burns
    #   the batch. Nothing notices they share a cause.
    #
    #   This is also the constraint that stops a new source from breaking the
    #   batch pipeline: a scraper that throws must not take the run down.
    #   Source-level isolation is the seam that guarantees it, and there is
    #   none here — `source.fetch()` is outside the try.
    #
    # INTENDED: decide the granularity (per-stage vs per-video), which
    #   failures abort the run rather than the item, and whether repeated
    #   same-stage failures should trip a circuit breaker.
    #
    # INVARIANT — do not change: VaultGuardError is re-raised, never
    #   swallowed. It means a write was about to land outside the vault. That
    #   is a stop, not an item failure, and no isolation policy may downgrade
    #   it to a logged warning.
    # ────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from doomnotes.download import (
    DownloadResult,
    Outcome,
    download,
    is_auth_failure,
    sleep_seconds,
)
from doomnotes.journal import Journal, NullJournal
from doomnotes.models import Media, Note, VideoRef
from doomnotes.render import SlugIndex, render_note, render_transcript
from doomnotes.store import MAX_ATTEMPTS, Store
from doomnotes.summarize import SummarizeError, summarize
from doomnotes.tags import TagRegistry
from doomnotes.transcribe import transcribe
from doomnotes.vault import VaultGuardError, VaultWriter

log = logging.getLogger(__name__)

# How many videos in one run must fail on auth before the run is treated as
# having a credentials problem rather than a video problem.
#
# Two, because the distinction only needs one corroborating case: a single 401
# among successes is a property of that video, and the second identical failure
# is the first evidence that it is not. Setting it higher would spend real
# attempts on a dead session, which is the thing being avoided.
AUTH_SYSTEMIC_THRESHOLD = 2


@dataclass
class Deps:
    """Injected collaborators. Defaults are the real ones."""

    downloader: Callable[..., DownloadResult] = download
    transcriber: Callable[..., str | None] = transcribe
    summarizer: Callable[..., Note] = summarize


@dataclass
class RunResult:
    attempted: int = 0
    notes_written: int = 0
    caption_only: int = 0
    failed: int = 0
    # Of `failed`, the ones the store will offer again next run.
    retryable: int = 0
    # Of `failed`, auth failures — offered again AND costing no attempt.
    blocked: int = 0
    stopped: bool = False
    stop_reason: str | None = None
    per_stage_failures: dict[str, int] = field(default_factory=dict)
    written_paths: list[Path] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Run summary",
            "-" * 50,
            f"  attempted        : {self.attempted}",
            f"  notes written    : {self.notes_written}",
            f"    of which caption-only: {self.caption_only}",
            f"  failed           : {self.failed}",
            f"    of which retryable: {self.retryable}",
        ]
        if self.blocked:
            # Said plainly rather than buried in the failure count: these cost
            # no attempt, so nothing will ever escalate on its own, and the
            # thing that needs fixing is not in the queue.
            lines.append(
                f"    of which auth-blocked: {self.blocked}  "
                f"— no attempt consumed; check cookies, then re-run"
            )
        for stage, n in sorted(self.per_stage_failures.items()):
            lines.append(f"      {stage:<12} {n}")
        if self.stopped:
            lines.append(f"  STOPPED          : {self.stop_reason}")
        return "\n".join(lines)


def process_one(
    ref: VideoRef,
    *,
    writer: VaultWriter,
    registry: TagRegistry,
    deps: Deps,
    audio_dir: Path,
    auth: dict,
    transcripts_dir: str,
    taken_slugs: SlugIndex,
    keep_audio: bool = False,
    last_chance: bool = True,
) -> tuple[str, str | None, Path | None]:
    """Run one ref through every stage.

    Returns (status, detail, note_path) where status is one of:
    "written", "caption_only", "retry", "blocked", "failed", "stop".

    "blocked" is a retryable auth failure: requeued like "retry", but it does
    not consume one of the video's attempts.

    `last_chance` says whether a retryable failure here would exhaust the
    video's attempts. It defaults to True — salvage now — because a caller
    without a store has no way to record a retry, so deferring would lose the
    video rather than postpone it.
    """
    # -- download ---------------------------------------------------------
    result = deps.downloader(ref, audio_dir, auth)

    if result.outcome is Outcome.STOP:
        return "stop", f"download:{result.reason or "unknown"}: {result.error}", None

    media: Media | None = result.media
    transcript: str | None = None
    # Why the download failed, carried to the journal on the salvage path. A
    # caption-only note with no recorded cause cannot be told apart from one
    # for a video that was simply deleted, and `mark_done` clears the store's
    # error column, so the journal is the only durable record of it.
    salvaged_from: str | None = None

    if result.outcome is Outcome.OK and media is not None:
        # -- transcribe ---------------------------------------------------
        try:
            transcript = deps.transcriber(media.audio_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("transcribe failed for %s: %s", ref.url, exc)
            transcript = None
    else:
        # A download failure is only survivable when the caption is already in
        # hand. Instagram's export carries it; TikTok's does not, and neither do
        # manual or Playwright refs — their caption arrives with the media.
        # A retryable cause returns to the queue while attempts remain, even
        # when a caption is in hand: settling for a caption-only note now would
        # trade the real transcript for a degraded note the video may not need.
        # The caption is still there on the last attempt, so nothing is lost by
        # waiting — only by settling early.
        # An auth failure is usually not about this video, so it is held rather
        # than charged — but NOT on the final attempt. An exemption that also
        # applied there would make queue residency unbounded: a video that only
        # ever fails on auth could never resolve, never take the caption-only
        # escape hatch, and would occupy a batch slot on every future run.
        #
        # Whether it is charged is decided at the END of the run, once it is
        # known how many other videos failed the same way. See the flush below.
        if (
            result.outcome is Outcome.RETRYABLE
            and is_auth_failure(result.reason)
            and not last_chance
        ):
            return "blocked", f"download:{result.reason}: {result.error}", None

        if result.outcome is Outcome.RETRYABLE and not last_chance:
            return "retry", f"download:{result.reason or "unknown"}: {result.error}", None

        # Terminal, or out of attempts. Salvage the caption if there is one.
        if not ref.has_export_caption:
            return "failed", f"download:{result.reason or "unknown"}: {result.error}", None

        salvaged_from = f"download:{result.reason or "unknown"}: {result.error}"

    # -- summarize --------------------------------------------------------
    try:
        note = deps.summarizer(ref, transcript, media, registry)
    except SummarizeError as exc:
        return "failed", f"summarize:{exc}", None

    # -- write ------------------------------------------------------------
    slug = taken_slugs.claim(note)

    transcript_rel = f"{transcripts_dir}/{slug}.md" if transcript else None
    note_md = render_note(note, slug if transcript else None, transcripts_dir)
    transcript_md = render_transcript(note, transcript) if transcript else None

    note_path, _ = writer.write_pair(f"{slug}.md", note_md, transcript_rel, transcript_md)

    registry.observe(note.tags)

    if media is not None and not keep_audio:
        # Audio is an intermediate. Transcripts are the durable artefact.
        Path(media.audio_path).unlink(missing_ok=True)

    return ("written" if transcript else "caption_only"), salvaged_from, note_path


def run(
    refs: Sequence[VideoRef],
    *,
    store: Store,
    writer: VaultWriter,
    registry: TagRegistry,
    deps: Deps | None = None,
    audio_dir: Path,
    auth_for: Callable[[str], dict],
    transcripts_dir: str = "_transcripts",
    limit: int | None = None,
    sleep_range: tuple[float, float] | None = None,
    keep_audio: bool = False,
    journal: "Journal | None" = None,
) -> RunResult:
    """Process a batch. Never raises for an individual video."""
    d = deps or Deps()
    out = RunResult()
    # Records what happened. Deliberately has no say in what happens next —
    # see journal.py. Defaults to a no-op so nothing depends on it working.
    jrn = journal or NullJournal()
    # Seeded from the vault, not empty: runs are separate processes days apart,
    # so an in-memory set cannot see notes an earlier run wrote. See SlugIndex.
    taken = SlugIndex.from_vault(writer.root, subdirs=(transcripts_dir,))

    # Blocked videos, held until the batch ends. Whether an auth failure costs
    # an attempt depends on how many OTHER videos failed the same way, and that
    # is not known while the video is being processed.
    #
    # Deferring is safe in the direction that matters: if the process dies
    # mid-run, these rows are simply not updated, so the videos keep their
    # previous state and are offered again. Nothing is written off by a crash.
    blocked_pending: list[tuple[str, str]] = []

    store.register(refs)
    batch = store.queue(list(refs), limit)
    jrn.write("run_started", queued=len(batch), offered=len(refs), limit=limit)

    for i, ref in enumerate(batch):
        out.attempted += 1
        started = time.monotonic()

        # ── Naive isolation: one boundary, per video. See the note above.
        try:
            status, detail, path = process_one(
                ref,
                writer=writer,
                registry=registry,
                deps=d,
                audio_dir=audio_dir,
                auth=auth_for(ref.platform),
                transcripts_dir=transcripts_dir,
                taken_slugs=taken,
                keep_audio=keep_audio,
                last_chance=store.attempts_for(ref.url) >= MAX_ATTEMPTS - 1,
            )
        except VaultGuardError:
            # Never swallowed. A write was about to land outside the vault.
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("unhandled failure on %s", ref.url)
            status, detail, path = "failed", f"unhandled:{type(exc).__name__}: {exc}", None

        # `detail` is prefixed with the stage that produced it, so recording it
        # verbatim is what lets a reader group failures by cause afterwards.
        jrn.write(
            "video",
            # The pipeline's own position in the batch, not a count of records.
            # A reader that numbers surviving records renumbers everything after
            # a dropped one — and since a record CAN be dropped (an unwritable
            # log, a torn line), three scattered failures would then read as one
            # unbroken incident. Which is the opposite of the thing this exists
            # to tell you.
            n=i + 1,
            of=len(batch),
            url=ref.url,
            platform=ref.platform,
            status=status,
            stage=(detail or "").split(":", 1)[0] or None,
            detail=detail,
            note=path.name if path else None,
            seconds=round(time.monotonic() - started, 3),
            source_order=ref.source_order,
        )

        if status == "stop":
            out.stopped = True
            out.stop_reason = detail
            log.error("STOP: %s — halting run, not retrying. %s", ref.url, detail)
            break

        if status in ("failed", "retry", "blocked"):
            out.failed += 1
            if status == "retry":
                out.retryable += 1
            stage = (detail or "unknown:").split(":", 1)[0]
            out.per_stage_failures[stage] = out.per_stage_failures.get(stage, 0) + 1
            if status == "blocked":
                blocked_pending.append((ref.url, detail or "unknown"))
            else:
                store.mark_failed(
                    ref.url,
                    detail or "unknown",
                    terminal=status == "failed",
                )
        else:
            out.notes_written += 1
            if status == "caption_only":
                out.caption_only += 1
            if path:
                out.written_paths.append(path)
            store.mark_done(ref.url)

        if sleep_range and i < len(batch) - 1:
            slept = sleep_seconds(*sleep_range)
            # Recorded because #3.3 is a decision about what the request pattern
            # should look like, and the configured range is not the same thing
            # as the spacing a run actually produced.
            jrn.write("slept", seconds=round(slept, 3))
            time.sleep(slept)

    # ── flush the held auth failures ──────────────────────────────────────
    # "This is a credentials problem" is now a measurement rather than an
    # assumption. Several videos failing the same way is evidence about the
    # session; one video failing while others succeed is evidence about that
    # video, and charging it is what stops a permanently-401 item from sitting
    # in the queue forever, never resolving and never taking the caption-only
    # escape hatch.
    if blocked_pending:
        systemic = len(blocked_pending) >= AUTH_SYSTEMIC_THRESHOLD
        for url, detail in blocked_pending:
            store.mark_failed(
                url, detail, terminal=False, counts_as_attempt=not systemic
            )
        if systemic:
            out.blocked = len(blocked_pending)
        else:
            # It behaved as an ordinary retryable failure, so it is reported as
            # one. Calling it "blocked" would point at credentials that are fine.
            out.retryable += len(blocked_pending)
        jrn.write(
            "auth_failures_resolved",
            count=len(blocked_pending),
            systemic=systemic,
            charged=not systemic,
        )

    jrn.write(
        "run_finished",
        attempted=out.attempted,
        notes_written=out.notes_written,
        caption_only=out.caption_only,
        failed=out.failed,
        # Broken out of `failed` because they mean different things to a reader
        # of the journal weeks later: retryable will resolve itself, blocked
        # will not resolve without someone fixing credentials.
        retryable=out.retryable,
        blocked=out.blocked,
        stopped=out.stopped,
        stop_reason=out.stop_reason,
        per_stage_failures=out.per_stage_failures,
    )
    return out
