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

from doomnotes.download import DownloadResult, Outcome, failure_reason
from doomnotes.models import Media, Note, VideoRef
from doomnotes.pipeline import Deps, RunResult, process_one, run
from doomnotes.render import SlugIndex
from doomnotes.store import MAX_ATTEMPTS, State, Store
from doomnotes.summarize import SummarizeError
from doomnotes.tags import TagRegistry
from doomnotes.vault import VaultGuardError, VaultWriter

# Synthetic identifiers only — never a real saved post.
IG = VideoRef(
    "https://www.instagram.com/reel/AAAAAAAAAAA/",
    "instagram",
    caption_source="export",
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
        # Derive the reason exactly as download() does. A fixture that omits it
        # tests a DownloadResult shape production never produces.
        return DownloadResult(ref, outcome, error=error, reason=failure_reason(error))

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
        taken_slugs=over.pop("taken_slugs", SlugIndex.from_vault(writer.root)),
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


# ── cross-run filename identity ──────────────────────────────────────────
#
# These three are a set and only make sense together. The first two say two
# different videos must never share a file; the third says one video re-run
# must never gain a second file. Satisfying either pair alone is easy and
# wrong — always suffix loses nothing but duplicates on every re-run, always
# reuse never duplicates but silently destroys a note.


def _same_title_summarizer(title: str = "Five AI tools you can replace with free ones"):
    def inner(ref, tr, media, reg):
        return a_note(ref, title=title, has_transcript=bool(tr))

    return inner


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


def test_a_transient_download_failure_is_left_retryable(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """The TikTok case the whole decision is for.

    No export caption means nothing to salvage, so the only thing standing
    between a rate limit and a permanently lost video is that the store offers
    it again. Asserting the state, not just the counter, because the counter
    would look identical if it were written off.
    """
    result = run(
        [TT],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(downloader=downloader_fails(
            Outcome.RETRYABLE, "HTTP Error 429: Too Many Requests"
        )),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )

    assert result.failed == 1
    assert result.retryable == 1
    assert store.state_of(TT.url) is State.RETRYABLE
    assert [r.url for r in store.filter_unprocessed([TT])] == [TT.url]


def test_a_transient_failure_does_not_settle_for_the_caption_yet(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """An Instagram ref has a caption, so it *could* be salvaged immediately.

    It should not be. A 429 says nothing about the video, and writing the
    caption-only note now trades the real transcript for a degraded note while
    attempts remain. The caption is not going anywhere.
    """
    result = run(
        [IG],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(downloader=downloader_fails(
            Outcome.RETRYABLE, "HTTP Error 429: Too Many Requests"
        )),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )

    assert result.notes_written == 0, "no note while a retry is still possible"
    assert result.caption_only == 0
    assert result.retryable == 1
    assert store.state_of(IG.url) is State.RETRYABLE
    assert list(writer.root.glob("*.md")) == []


def test_the_caption_is_salvaged_on_the_last_attempt(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """Deferring must not become losing.

    Once the attempts are spent the video is written off either way, so the
    caption-only note is strictly better than nothing — which is the whole
    reason the salvage path exists.
    """
    store.register([IG])
    for _ in range(MAX_ATTEMPTS - 1):
        store.mark_failed(IG.url, "HTTP Error 429: Too Many Requests", terminal=False)

    result = run(
        [IG],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(downloader=downloader_fails(
            Outcome.RETRYABLE, "HTTP Error 429: Too Many Requests"
        )),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )

    assert result.caption_only == 1, "the last attempt falls back to the caption"
    assert store.state_of(IG.url) is State.DONE
    assert len(list(writer.root.glob("*.md"))) == 1


def test_a_terminal_failure_salvages_immediately(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """A deleted video is not coming back, so there is nothing to wait for."""
    result = run(
        [IG],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(downloader=downloader_fails(Outcome.TERMINAL, "Video unavailable")),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )

    assert result.caption_only == 1
    assert result.retryable == 0
    assert store.state_of(IG.url) is State.DONE


def test_a_caption_only_note_records_why_the_download_failed(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """Otherwise a salvaged note is indistinguishable from an ordinary one.

    `mark_done` clears the store's error column, so the journal is the only
    durable record that this note exists because a download failed, and of
    which failure it was.
    """
    records = []

    class Recorder:
        """Matches the Journal protocol: write() and close(), nothing else."""

        def write(self, event: str, **fields) -> None:
            records.append({"event": event, **fields})

        def close(self) -> None:
            return None

    run(
        [IG],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(downloader=downloader_fails(Outcome.TERMINAL, "Video unavailable")),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
        journal=Recorder(),
    )

    videos = [r for r in records if r["event"] == "video"]
    assert len(videos) == 1
    assert videos[0]["status"] == "caption_only"
    assert "deleted" in (videos[0]["detail"] or ""), videos[0]["detail"]


def test_a_stop_reports_its_descriptive_reason(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """"the run stopped" is not a diagnosis: a checkpoint and a dead session
    need different responses from the person reading the report."""
    result = run(
        [IG],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(downloader=downloader_fails(
            Outcome.STOP, "ERROR: [Instagram] Requested content is not available, login_required"
        )),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )
    assert result.stopped is True
    assert "session_dead" in (result.stop_reason or ""), result.stop_reason


def test_the_whole_retry_lifecycle_across_consecutive_runs(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """Four runs against one store, as days apart would look.

    The unit tests each pin one moment. What actually matters is the sequence,
    and it is the thing a reader has to trust: retries must be bounded AND must
    happen, and the caption must be held back until there is nothing left to
    wait for. An off-by-one anywhere shows up here as a fourth attempt or as a
    note written on run 1.
    """
    err = "ERROR: unable to download webpage: HTTP Error 429: Too Many Requests"
    d = deps(downloader=downloader_fails(Outcome.RETRYABLE, err))

    seen = []
    for _ in range(4):
        result = run(
            [IG, TT],
            store=store,
            writer=writer,
            registry=TagRegistry(),
            deps=d,
            audio_dir=tmp_path / "audio",
            auth_for=lambda p: {},
            sleep_range=None,
        )
        seen.append((
            result.attempted,
            store.state_of(IG.url),
            store.state_of(TT.url),
            len(list(writer.root.glob("*.md"))),
        ))

    assert seen == [
        (2, State.RETRYABLE, State.RETRYABLE, 0),
        (2, State.RETRYABLE, State.RETRYABLE, 0),
        # Last attempt: Instagram falls back to its caption, TikTok has nothing.
        (2, State.DONE, State.FAILED, 1),
        # Both are processed now, so the batch is empty. Bounded at three.
        (0, State.DONE, State.FAILED, 1),
    ], seen

    assert store.attempts_for(IG.url) == MAX_ATTEMPTS
    assert store.attempts_for(TT.url) == MAX_ATTEMPTS


def test_an_auth_failure_costs_the_video_nothing(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """A stale cookie is not the video's fault, so it is not the video's cost.

    Three runs against a dead session would otherwise spend every video's whole
    budget proving the cookie is still stale, and write off the backlog — on
    TikTok, permanently — while the actual fault sat in a file on disk.
    """
    d = deps(downloader=downloader_fails(
        Outcome.RETRYABLE, "ERROR: unable to download webpage: HTTP Error 401: Unauthorized"
    ))

    for _ in range(MAX_ATTEMPTS * 2):
        result = run(
            [TT],
            store=store,
            writer=writer,
            registry=TagRegistry(),
            deps=d,
            audio_dir=tmp_path / "audio",
            auth_for=lambda p: {},
            sleep_range=None,
        )

    assert result.blocked == 1
    assert result.retryable == 0
    assert store.state_of(TT.url) is State.RETRYABLE
    assert store.attempts_for(TT.url) == 0, "no attempt may be consumed"
    assert [r.url for r in store.filter_unprocessed([TT])] == [TT.url]


def test_an_auth_failure_does_not_force_the_degraded_note(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """The ordering that makes the whole thing work.

    This ref is already one attempt from the cap, so `last_chance` is true and
    an ordinary retryable failure would salvage the caption and close the book.
    An auth failure must not: the transcript is still reachable once the cookie
    is fixed, and settling now would trade it away for someone else's problem.
    """
    store.register([IG])
    for _ in range(MAX_ATTEMPTS - 1):
        store.mark_failed(IG.url, "HTTP Error 429", terminal=False)
    assert store.attempts_for(IG.url) == MAX_ATTEMPTS - 1

    result = run(
        [IG],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(downloader=downloader_fails(
            Outcome.RETRYABLE, "ERROR: unable to download webpage: HTTP Error 401: Unauthorized"
        )),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )

    assert result.caption_only == 0, "no degraded note on an auth failure"
    assert result.blocked == 1
    assert list(writer.root.glob("*.md")) == []
    assert store.state_of(IG.url) is State.RETRYABLE
    assert store.attempts_for(IG.url) == MAX_ATTEMPTS - 1, "budget untouched"


def test_the_run_summary_names_an_auth_problem(
    writer: VaultWriter, store: Store, tmp_path: Path
) -> None:
    """Nothing escalates on its own now, so the report has to say it.

    An auth-blocked video costs no attempt and never becomes FAILED, so it will
    sit in the queue indefinitely and no counter will ever cross a threshold.
    The run summary is the only place this surfaces.
    """
    result = run(
        [TT],
        store=store,
        writer=writer,
        registry=TagRegistry(),
        deps=deps(downloader=downloader_fails(
            Outcome.RETRYABLE, "ERROR: unable to download webpage: HTTP Error 401: Unauthorized"
        )),
        audio_dir=tmp_path / "audio",
        auth_for=lambda p: {},
        sleep_range=None,
    )

    rendered = result.summary()
    assert "auth-blocked: 1" in rendered, rendered
    assert "cookies" in rendered
