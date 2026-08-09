"""Pipeline orchestration tests.

Every URL in this file is invented (doomnotes:synthetic-urls). None of them come
from the real exports, which is why the pre-commit bulk-URL guard is told to
skip this file.

    # WHAT THESE ASSERT, AND WHAT THEY DELIBERATELY DO NOT
    #
    # pipeline.py carries the HANDS-ON #7.1 decision (failure-isolation
    # granularity). A test that pinned down the *current* isolation policy
    # would break the moment that decision is made, which would turn this file
    # into an obstacle instead of a safety net.
    #
    # So everything here is one of two kinds:
    #   1. an INVARIANT that survives every one of the seven decisions — the
    #      vault guard is re-raised, a note and its transcript are written as a
    #      pair, a STOP halts the run, audio is an intermediate;
    #   2. the DESIRED behaviour of an undecided item, marked xfail(strict=True)
    #      and labelled with its HANDS-ON number.
    #
    # Nothing asserts "the baseline does X".
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from doomnotes.download import DownloadResult, Outcome
from doomnotes.models import Media, Note, VideoRef
from doomnotes.pipeline import Deps, RunResult, process_one, run
from doomnotes.store import State, Store
from doomnotes.summarize import SummarizeError
from doomnotes.tags import TagRegistry
from doomnotes.vault import VaultGuardError, VaultWriter

# Synthetic identifiers only — never a real saved post.
IG = VideoRef(
    "https://www.instagram.com/reel/AAAAAAAAAAA/",
    "instagram",
    caption="A caption that shipped in the export.",
    author="@someone",
    source_order=0,
)
TT = VideoRef(
    "https://www.tiktokv.com/share/video/1111111111111111111/",
    "tiktok",
    saved_at=datetime(2026, 8, 5, 8, 36, 26),
    source_order=0,
)


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "ai-notes-vault"
    root.mkdir()
    return root


@pytest.fixture()
def writer(vault: Path) -> VaultWriter:
    return VaultWriter(vault)


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    with Store(tmp_path / "state.db") as s:
        yield s


def a_note(ref: VideoRef, **over) -> Note:
    fields = dict(
        title="Postgres partial indexes for soft-deleted rows",
        source_url=ref.url,
        platform=ref.platform,
        summary="It explains partial indexes.",
        key_points=["Index only live rows"],
        tags=["coding", "databases"],
        author=ref.author,
        caption=ref.caption,
        saved_at=ref.saved_at,
        source_order=ref.source_order,
    )
    fields.update(over)
    return Note(**fields)


def downloader_ok(audio_name: str = "AAA.m4a"):
    """A downloader that produces a real (empty) audio file, like yt-dlp would."""

    def inner(ref: VideoRef, audio_dir: Path, auth: dict) -> DownloadResult:
        audio_dir.mkdir(parents=True, exist_ok=True)
        path = Path(audio_dir) / audio_name
        path.write_bytes(b"\x00")
        return DownloadResult(
            ref, Outcome.OK, media=Media(ref=ref, audio_path=path, posted_at=datetime(2026, 7, 14))
        )

    return inner


def downloader_fails(outcome: Outcome = Outcome.TERMINAL, error: str = "Video unavailable"):
    def inner(ref: VideoRef, audio_dir: Path, auth: dict) -> DownloadResult:
        return DownloadResult(ref, outcome, error=error)

    return inner


def deps(**over) -> Deps:
    base = dict(
        downloader=downloader_ok(),
        transcriber=lambda p: "a transcript long enough to be real content",
        summarizer=lambda ref, tr, media, reg: a_note(ref, has_transcript=bool(tr)),
    )
    base.update(over)
    return Deps(**base)


def process(ref: VideoRef, writer: VaultWriter, tmp_path: Path, **over):
    return process_one(
        ref,
        writer=writer,
        registry=over.pop("registry", TagRegistry()),
        deps=deps(**over.pop("deps", {})),
        audio_dir=tmp_path / "audio",
        auth={},
        transcripts_dir="_transcripts",
        taken_slugs=over.pop("taken_slugs", set()),
        **over,
    )


# ── invariants: the vault guard ──────────────────────────────────────────


def test_vault_guard_error_is_never_swallowed(writer: VaultWriter, store: Store, tmp_path: Path) -> None:
    """Documented INVARIANT in pipeline.py — no isolation policy may downgrade it.

    A write about to land outside the vault is a stop, not an item failure.
    This is the one exception that must survive HANDS-ON #7.1 whatever is
    decided, so it is asserted at the `run()` level where the catch-all lives.
    """

    def guard_tripping_summarizer(ref, tr, media, reg):
        raise VaultGuardError("refusing to write outside the vault")

    with pytest.raises(VaultGuardError):
        run(
            [IG],
            store=store,
            writer=writer,
            registry=TagRegistry(),
            deps=deps(summarizer=guard_tripping_summarizer),
            audio_dir=tmp_path / "audio",
            auth_for=lambda p: {},
            sleep_range=None,
        )


def test_ordinary_exceptions_are_isolated_but_guard_errors_are_not(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """The catch-all exists; it just must not be reachable by VaultGuardError."""

    def boom(ref, tr, media, reg):
        raise MemoryError("whisper ate the machine")

    result = run(
        [IG],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(summarizer=boom),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )
    assert result.failed == 1
    assert result.notes_written == 0


# ── invariants: the note/transcript pair ─────────────────────────────────


def test_note_and_transcript_are_written_as_a_pair(writer: VaultWriter, tmp_path: Path) -> None:
    """A wikilink must never dangle — Tier-1 #2's whole point."""
    status, detail, path = process(IG, writer, tmp_path)
    assert status == "written", detail
    assert path is not None

    body = path.read_text(encoding="utf-8")
    assert "[[_transcripts/" in body

    slug = path.stem
    assert (writer.root / "_transcripts" / f"{slug}.md").is_file(), (
        "the note links to a transcript that was never written"
    )


def test_caption_only_note_carries_no_transcript_link(writer: VaultWriter, tmp_path: Path) -> None:
    """Instagram's export caption is what makes a failed download survivable."""
    status, detail, path = process(
        IG, writer, tmp_path, deps={"downloader": downloader_fails()}
    )
    assert status == "caption_only", detail
    assert path is not None

    body = path.read_text(encoding="utf-8")
    assert "has_transcript: false" in body
    assert "[[_transcripts/" not in body
    assert not (writer.root / "_transcripts").exists() or not list(
        (writer.root / "_transcripts").glob("*.md")
    )


def test_failed_tiktok_download_produces_no_note(writer: VaultWriter, tmp_path: Path) -> None:
    """The load-bearing asymmetry: TikTok has no export caption, so nothing survives.

    Not a baseline detail — this is the ruled consequence of the two exports
    having opposite gaps, and it is why misclassifying a transient TikTok
    failure as terminal loses the video permanently.
    """
    status, detail, path = process(
        TT, writer, tmp_path, deps={"downloader": downloader_fails()}
    )
    assert status == "failed"
    assert path is None
    assert list(writer.root.glob("*.md")) == []


def test_a_ref_with_neither_caption_nor_transcript_is_not_written(
    writer: VaultWriter, tmp_path: Path
) -> None:
    """A note invented from nothing is worse than a logged failure."""

    def refuses(ref, tr, media, reg):
        raise SummarizeError("nothing to summarise: no caption and no transcript")

    status, detail, path = process(
        TT,
        writer,
        tmp_path,
        deps={"transcriber": lambda p: None, "summarizer": refuses},
    )
    assert status == "failed"
    assert (detail or "").startswith("summarize:")
    assert list(writer.root.glob("*.md")) == []


# ── invariants: STOP semantics ───────────────────────────────────────────


def test_stop_halts_the_run_and_leaves_the_rest_untouched(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """Hard constraint: on captcha/checkpoint, stop and notify. Never retry.

    The remaining refs must stay PENDING — a halted run that marked them
    processed would silently drop them from every future queue.
    """
    second = replace(IG, url="https://www.instagram.com/reel/BBBBBBBBBBB/", source_order=1)
    third = replace(IG, url="https://www.instagram.com/reel/CCCCCCCCCCC/", source_order=2)

    result = run(
        [IG, second, third],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(downloader=downloader_fails(Outcome.STOP, "challenge_required: checkpoint")),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )

    assert result.stopped is True
    assert result.attempted == 1, "a STOP must not be followed by more requests"
    assert store.state_of(second.url) is State.PENDING
    assert store.state_of(third.url) is State.PENDING


def test_stop_records_the_reason(writer: VaultWriter, store: Store, tmp_path: Path) -> None:
    result = run(
        [IG],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(downloader=downloader_fails(Outcome.STOP, "challenge_required: checkpoint")),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )
    assert "checkpoint" in (result.stop_reason or "")


def test_a_stopped_video_is_not_marked_processed(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """The video that tripped the checkpoint was never actually judged.

    Marking it FAILED would mean clearing the challenge and re-running silently
    skips it — the one video you know nothing about.
    """
    run(
        [IG],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(downloader=downloader_fails(Outcome.STOP, "captcha")),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )
    assert store.state_of(IG.url) is State.PENDING


# ── invariants: audio is an intermediate ─────────────────────────────────


def test_audio_is_deleted_after_a_successful_note(writer: VaultWriter, tmp_path: Path) -> None:
    """Ruled: audio extracted then discarded; the transcript is the artefact."""
    audio_dir = tmp_path / "audio"
    process(IG, writer, tmp_path)
    assert list(audio_dir.glob("*.m4a")) == []


def test_keep_audio_opts_out(writer: VaultWriter, tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    process(IG, writer, tmp_path, keep_audio=True)
    assert list(audio_dir.glob("*.m4a")) != []


def test_audio_never_lands_in_the_vault(writer: VaultWriter, tmp_path: Path) -> None:
    """Only notes and transcripts belong in the vault Obsidian indexes."""
    process(IG, writer, tmp_path, keep_audio=True)
    assert list(writer.root.rglob("*.m4a")) == []


# ── invariants: store bookkeeping ────────────────────────────────────────


def test_only_written_notes_are_marked_done(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    failing = replace(TT, url="https://www.tiktokv.com/share/video/2222222222222222222/")
    run(
        [IG, failing],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=Deps(
            downloader=lambda ref, d, a: (
                downloader_ok()(ref, d, a)
                if ref.platform == "instagram"
                else downloader_fails()(ref, d, a)
            ),
            transcriber=lambda p: "a transcript long enough to be real content",
            summarizer=lambda ref, tr, media, reg: a_note(ref, has_transcript=bool(tr)),
        ),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )
    assert store.state_of(IG.url) is State.DONE
    assert store.state_of(failing.url) is State.FAILED


def test_rerun_does_not_reprocess_and_does_not_recreate_deleted_notes(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """The isolation test, as a unit test.

    Deleting a note must NOT bring it back on the next run: the store is
    authoritative and never consults the vault. If this ever inverts, deleting
    a bad note silently re-downloads and re-summarises it.
    """
    common = dict(
        store=store,
        writer=writer,
        registry=TagRegistry(),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )
    first = run([IG], deps=deps(), **common)
    assert first.notes_written == 1

    for note in writer.root.glob("*.md"):
        note.unlink()

    second = run([IG], deps=deps(), **common)
    assert second.attempted == 0
    assert list(writer.root.glob("*.md")) == []


def test_registry_observes_the_tags_of_written_notes(writer: VaultWriter, tmp_path: Path) -> None:
    registry = TagRegistry()
    process(IG, writer, tmp_path, registry=registry)
    assert registry.counts()["coding"] == 1
    assert registry.counts()["databases"] == 1


def test_run_result_counts_reconcile(writer: VaultWriter, store: Store, tmp_path: Path) -> None:
    """attempted == written + failed, or the run summary is lying."""
    ok = IG
    bad = replace(TT, url="https://www.tiktokv.com/share/video/3333333333333333333/")
    result: RunResult = run(
        [ok, bad],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=Deps(
            downloader=lambda ref, d, a: (
                downloader_ok()(ref, d, a)
                if ref.platform == "instagram"
                else downloader_fails()(ref, d, a)
            ),
            transcriber=lambda p: "a transcript long enough to be real content",
            summarizer=lambda ref, tr, media, reg: a_note(ref, has_transcript=bool(tr)),
        ),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )
    assert result.attempted == result.notes_written + result.failed
    assert "Run summary" in result.summary()


def test_limit_caps_the_batch(writer: VaultWriter, store: Store, tmp_path: Path) -> None:
    """Pacing is a risk decision (#3.3); that the cap is *honoured* is not."""
    refs = [
        replace(IG, url=f"https://www.instagram.com/reel/AAAAAAAAA{i:02d}/", source_order=i)
        for i in range(5)
    ]
    result = run(
        refs,
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        limit=2,
        sleep_range=None,
    )
    assert result.attempted == 2


def test_auth_is_resolved_per_platform(writer: VaultWriter, store: Store, tmp_path: Path) -> None:
    """Tier-1 #1: Instagram and TikTok may need different answers."""
    seen: list[tuple[str, dict]] = []

    def recording_downloader(ref, audio_dir, auth):
        seen.append((ref.platform, auth))
        return downloader_fails()(ref, audio_dir, auth)

    run(
        [IG, TT],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(downloader=recording_downloader),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {"mode": "browser"} if p == "instagram" else {"mode": "none"},
        sleep_range=None,
    )
    assert dict(seen)["instagram"]["mode"] == "browser"
    assert dict(seen)["tiktok"]["mode"] == "none"


def test_a_transcription_failure_still_yields_a_caption_note(
    writer: VaultWriter, tmp_path: Path
) -> None:
    """Whisper dying is not the same as the video being unavailable."""

    def boom(path):
        raise RuntimeError("ct2 died")

    status, detail, path = process(IG, writer, tmp_path, deps={"transcriber": boom})
    assert status == "caption_only", detail
    assert path is not None


# ── known bug, fixed in the next PR ──────────────────────────────────────


def _same_title_summarizer(title: str = "Five AI tools you can replace with free ones"):
    def inner(ref, tr, media, reg):
        return a_note(ref, title=title, has_transcript=bool(tr))

    return inner


@pytest.mark.xfail(
    reason="BUG: the taken-slug set is rebuilt empty on every run, so collision "
           "detection only sees notes written by the CURRENT run. A second run "
           "producing the same title overwrites the first run's note in place — "
           "no error, and the store still says the lost video is done, so it is "
           "never regenerated. Paced runs are 30-40/day, so cross-run is the "
           "normal case rather than the edge one.",
    strict=True,
)
def test_a_later_run_does_not_overwrite_an_earlier_runs_note(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """Two different videos, same generated title, two separate runs."""
    second = replace(IG, url="https://www.instagram.com/reel/BBBBBBBBBBB/", source_order=1)
    common = dict(
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(
            downloader=downloader_fails(),
            summarizer=_same_title_summarizer(),
        ),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )
    run([IG], **common)
    run([second], **common)

    urls = {
        line.split("source_url: ", 1)[1].splitlines()[0]
        for line in (p.read_text(encoding="utf-8") for p in writer.root.glob("*.md"))
        if "source_url: " in line
    }
    assert urls == {IG.url, second.url}, (
        "one video's note was silently replaced by the other's"
    )


@pytest.mark.xfail(
    reason="BUG: same root cause, applied to the transcript. Two runs, same "
           "title, and the second run's raw ASR replaces the first's — while "
           "the first note's wikilink still points at it, so the link resolves "
           "to the wrong video's transcript rather than dangling visibly.",
    strict=True,
)
def test_a_later_run_does_not_overwrite_an_earlier_runs_transcript(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    second = replace(IG, url="https://www.instagram.com/reel/BBBBBBBBBBB/", source_order=1)

    def common(transcript: str):
        return dict(
            store=store,
            writer=writer,
            registry=TagRegistry(),
            deps=deps(
                transcriber=lambda p: transcript,
                summarizer=_same_title_summarizer(),
            ),
            audio_dir=tmp_path / "audio",
            auth_for=lambda p: {},
            sleep_range=None,
        )

    run([IG], **common("the first video said this, at length, with detail"))
    run([second], **common("the second video said something else entirely here"))

    transcripts = list((writer.root / "_transcripts").glob("*.md"))
    assert len(transcripts) == 2, (
        f"expected one transcript per video, found {[p.name for p in transcripts]}"
    )


def test_reprocessing_the_same_url_reuses_its_filename(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """The other half of the fix, asserted now so it cannot regress into it.

    Re-running one video must land on the SAME file, not accumulate a
    hash-suffixed duplicate. This is why the collision suffix is derived from
    the URL rather than from a counter — and it is the constraint that makes
    "just seed the taken-set from the vault" the wrong fix.
    """
    common = dict(
        writer=writer,
        registry=TagRegistry(),
        deps=deps(downloader=downloader_fails(), summarizer=_same_title_summarizer()),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )
    with Store(tmp_path / "one.db") as s1:
        run([IG], store=s1, **common)
    with Store(tmp_path / "two.db") as s2:  # a fresh store == a forced reprocess
        run([IG], store=s2, **common)

    assert len(list(writer.root.glob("*.md"))) == 1


# ── the gap the baseline has ─────────────────────────────────────────────
#
# HANDS-ON #7.1. This asserts the OUTCOME the decision has to prevent, not the
# mechanism — circuit breaker, per-stage abort or something else is your call.


@pytest.mark.xfail(
    reason="HANDS-ON #7.1: one try/except per video cannot tell a systemic "
           "outage from an item failure, so a dead summariser consumes the "
           "whole batch and marks every video processed.",
    strict=True,
)
def test_a_systemic_stage_outage_does_not_consume_the_batch(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """If Ollama dies at video 1, videos 2..30 must not be spent finding out.

    The baseline logs 29 more identical failures and burns the batch. What
    replaces it is yours to decide; this asserts only that the run stops
    somewhere short of the end.
    """
    refs = [
        replace(IG, url=f"https://www.instagram.com/reel/AAAAAAAAA{i:02d}/", source_order=i)
        for i in range(10)
    ]

    def dead_ollama(ref, tr, media, reg):
        raise SummarizeError("connection refused: localhost:11434")

    result = run(
        refs,
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(summarizer=dead_ollama),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )
    assert result.attempted < len(refs), (
        "every video failed the same way and the run never noticed they shared a cause"
    )


# The other half of #7.1 — "a Playwright source that throws must not take down
# the batch pipeline" — is deliberately NOT tested here.
#
# Sources are collected in cli.py before `run()` is ever called, so there is no
# seam at this layer to assert against. Any test would have to invent one
# (a `sources=` parameter, a collect-and-isolate helper, a per-source try), and
# an xfail(strict=True) that can only be satisfied by one specific API shape
# stops being a specification and becomes a mechanism the decision has to obey.
# The gap is documented in pipeline.py's HANDS-ON block and HANDS_ON.md §7.1;
# it stays prose until the shape of the seam is chosen.
