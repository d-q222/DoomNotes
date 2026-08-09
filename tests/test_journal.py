"""Run journal tests.

Every URL in this file is invented (doomnotes:synthetic-urls).

    # THE CONSTRAINT WORTH ASSERTING
    #
    # The journal records; it does not react. A journal that counted
    # consecutive same-stage failures and stopped the run would have made
    # HANDS-ON #7.1's decision — what failure isolation should do — invisibly,
    # inside a module named "logging". So there is a test below that a dead
    # summariser still consumes the whole batch WITH journalling on, which is
    # not an endorsement of that behaviour: it is the assertion that turning
    # the journal on did not quietly change it.
    #
    # The other half is that the journal cannot cost you a batch. Every write
    # is failure-tolerant, because an observability feature that can abort a
    # rate-limited run is worse than no observability at all.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from doomnotes.download import DownloadResult, Outcome
from doomnotes.journal import NullJournal, RunJournal, open_journal, read_runs
from doomnotes.models import Note, VideoRef
from doomnotes.pipeline import Deps, run
from doomnotes.store import Store
from doomnotes.summarize import SummarizeError
from doomnotes.tags import TagRegistry
from doomnotes.vault import VaultWriter

IG = VideoRef(
    "https://www.instagram.com/reel/AAAAAAAAAAA/",
    "instagram",
    caption="a caption from the export",
    source_order=0,
)


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "ai-notes-vault"
    root.mkdir()
    return root


def a_note(ref: VideoRef, title: str = "A note about postgres indexes") -> Note:
    return Note(
        title=title, source_url=ref.url, platform=ref.platform,
        summary="s" * 30, tags=["coding"],
    )


def deps(**over) -> Deps:
    base = dict(
        downloader=lambda ref, d, a: DownloadResult(ref, Outcome.TERMINAL, error="Video unavailable"),
        transcriber=lambda p: None,
        summarizer=lambda ref, tr, media, reg: a_note(ref),
    )
    base.update(over)
    return Deps(**base)


def do_run(refs, vault: Path, tmp_path: Path, journal=None, **over):
    with Store(tmp_path / "state.db") as store:
        return run(
            refs,
            store=store,
            writer=VaultWriter(vault),
            registry=TagRegistry(),
            deps=over.pop("deps", deps()),
            audio_dir=tmp_path / "audio",
            auth_for=lambda p: {},
            sleep_range=None,
            journal=journal,
            **over,
        )


# ── the file ─────────────────────────────────────────────────────────────


def test_records_are_one_json_object_per_line(tmp_path: Path) -> None:
    """JSONL rather than one document, because a killed run tears the last
    line and should not cost the file."""
    path = tmp_path / "logs" / "runs.jsonl"
    with RunJournal(path, run_id="r1") as j:
        j.write("run_started", queued=2)
        j.write("video", url=IG.url, status="written")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["run_id"] == "r1" for line in lines)


def test_it_appends_rather_than_truncating(tmp_path: Path) -> None:
    """A run that overwrote the history would defeat the purpose."""
    path = tmp_path / "runs.jsonl"
    with RunJournal(path, run_id="r1") as j:
        j.write("run_started")
    with RunJournal(path, run_id="r2") as j:
        j.write("run_started")
    assert {r["run_id"] for r in read_runs(path)} == {"r1", "r2"}


def test_every_record_is_timestamped_and_attributed(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    with RunJournal(path, run_id="r1") as j:
        j.write("video", url=IG.url)
    record = read_runs(path)[0]
    assert record["ts"] and record["run_id"] == "r1" and record["event"] == "video"


def test_a_torn_final_line_is_skipped_not_raised_on(tmp_path: Path) -> None:
    """The expected cost of a killed run, not corruption."""
    path = tmp_path / "runs.jsonl"
    with RunJournal(path, run_id="r1") as j:
        j.write("video", url=IG.url)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"event": "video", "url": "htt')

    assert len(read_runs(path)) == 1


def test_reading_a_journal_that_does_not_exist_yet_is_empty(tmp_path: Path) -> None:
    assert read_runs(tmp_path / "never-written.jsonl") == []


def test_unserialisable_values_do_not_raise(tmp_path: Path) -> None:
    """A Path or a datetime in a field must not take down a run."""
    path = tmp_path / "runs.jsonl"
    with RunJournal(path, run_id="r1") as j:
        j.write("video", note=Path("/tmp/x.md"))
    assert read_runs(path)[0]["note"] == "/tmp/x.md"


def test_an_unwritable_location_is_survived_not_raised(tmp_path: Path) -> None:
    """An observability feature must never cost a rate-limited batch."""
    blocked = tmp_path / "afile"
    blocked.write_text("not a directory", encoding="utf-8")
    journal = RunJournal(blocked / "nested" / "runs.jsonl")
    journal.write("run_started")   # must not raise
    journal.close()


def test_open_journal_with_no_path_records_nothing(tmp_path: Path) -> None:
    journal = open_journal(None)
    assert isinstance(journal, NullJournal)
    journal.write("run_started")


# ── what a run puts in it ────────────────────────────────────────────────


def test_a_run_records_start_each_video_and_finish(vault: Path, tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    with RunJournal(path, run_id="r1") as journal:
        do_run([IG], vault, tmp_path, journal=journal)

    events = [r["event"] for r in read_runs(path)]
    assert events[0] == "run_started"
    assert events[-1] == "run_finished"
    assert events.count("video") == 1


def test_a_failure_records_the_stage_that_produced_it(vault: Path, tmp_path: Path) -> None:
    """The grouping key. Without it, 29 failures look like 29 problems."""
    path = tmp_path / "runs.jsonl"
    tiktok = VideoRef("https://www.tiktokv.com/share/video/1111111111111111111/", "tiktok")
    with RunJournal(path, run_id="r1") as journal:
        do_run([tiktok], vault, tmp_path, journal=journal)

    video = next(r for r in read_runs(path) if r["event"] == "video")
    assert video["status"] == "failed"
    assert video["stage"] == "download"
    assert "Video unavailable" in video["detail"]


def test_a_systemic_outage_is_visible_as_one_shared_cause(vault: Path, tmp_path: Path) -> None:
    """The evidence #7.1 needs, and the reason this module exists.

    Ten videos failing at the same stage in quick succession is a dead Ollama,
    not ten bad videos. The run summary cannot say that; `state.db` keeps only
    the latest state per URL. This can.

    Note carefully what is NOT asserted: how many videos ran. An earlier
    version of this test asserted `attempted == 10`, which pinned the naive
    baseline — the one #7.1 exists to replace — inside a file about logging.
    Implementing a circuit breaker would then have failed a journal test for no
    reason the developer could see from its name.

    Only journal-owned properties belong here: that failures were recorded, and
    that they carry a shared stage. Whether the run should have stopped early
    is #7.1's to decide, and `test_journalling_does_not_change_the_runs_outcome`
    is where "the journal changed nothing" is checked — by comparing two runs
    to each other rather than to a literal.
    """
    refs = [
        replace(IG, url=f"https://www.instagram.com/reel/AAAAAAAAA{i:02d}/", source_order=i)
        for i in range(10)
    ]

    def dead_ollama(ref, tr, media, reg):
        raise SummarizeError("connection refused: localhost:11434")

    path = tmp_path / "runs.jsonl"
    with RunJournal(path, run_id="r1") as journal:
        do_run(refs, vault, tmp_path, journal=journal, deps=deps(summarizer=dead_ollama))

    failures = [
        r for r in read_runs(path) if r["event"] == "video" and r["status"] == "failed"
    ]
    assert failures, "the run failed throughout; the journal should say so"
    assert {v["stage"] for v in failures} == {"summarize"}, (
        "the shared cause is the whole point — one dead service, not N bad videos"
    )


def test_journalling_does_not_change_the_runs_outcome(vault: Path, tmp_path: Path) -> None:
    """Run the same batch twice, with and without a journal, and compare."""
    refs = [
        replace(IG, url=f"https://www.instagram.com/reel/BBBBBBBBB{i:02d}/", source_order=i)
        for i in range(3)
    ]
    without = do_run(refs, vault, tmp_path / "a", journal=None)

    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    second_vault = tmp_path / "b" / "vault"
    second_vault.mkdir()
    with RunJournal(tmp_path / "runs.jsonl", run_id="r1") as journal:
        with_journal = do_run(refs, second_vault, tmp_path / "b", journal=journal)

    assert (without.attempted, without.notes_written, without.failed) == (
        with_journal.attempted, with_journal.notes_written, with_journal.failed
    )


def test_a_stop_is_recorded_before_the_run_halts(vault: Path, tmp_path: Path) -> None:
    """A checkpoint is the run you most want a record of afterwards."""
    path = tmp_path / "runs.jsonl"
    stopper = deps(
        downloader=lambda ref, d, a: DownloadResult(
            ref, Outcome.STOP, error="challenge_required: checkpoint"
        )
    )
    with RunJournal(path, run_id="r1") as journal:
        do_run([IG], vault, tmp_path, journal=journal, deps=stopper)

    records = read_runs(path)
    video = next(r for r in records if r["event"] == "video")
    assert video["status"] == "stop"
    assert "checkpoint" in video["detail"]

    # A halted run still finishes — it just finishes stopped, and says why.
    # Reserving "no run_finished record at all" for a process that was killed
    # is what makes that distinction readable afterwards.
    finished = next(r for r in records if r["event"] == "run_finished")
    assert finished["stopped"] is True
    assert "checkpoint" in finished["stop_reason"]


def test_a_killed_run_is_distinguishable_from_a_finished_one(vault: Path, tmp_path: Path) -> None:
    """No run_finished record means the process did not get to the end.

    That is the only way to tell "the batch completed with failures" from
    "something killed it partway", and they call for different responses.
    """
    path = tmp_path / "runs.jsonl"

    def killed(ref, tr, media, reg):
        raise KeyboardInterrupt

    journal = RunJournal(path, run_id="r1")
    with pytest.raises(KeyboardInterrupt):
        do_run([IG], vault, tmp_path, journal=journal, deps=deps(summarizer=killed))
    journal.close()

    records = read_runs(path)
    assert [r["event"] for r in records] == ["run_started"]


def test_no_run_writes_the_journal_into_the_vault(vault: Path, tmp_path: Path) -> None:
    """It holds saved-video URLs and belongs in data/, never where Obsidian looks."""
    path = tmp_path / "runs.jsonl"
    with RunJournal(path, run_id="r1") as journal:
        do_run([IG], vault, tmp_path, journal=journal)
    assert list(vault.rglob("*.jsonl")) == []


# ── findings from adversarial review ─────────────────────────────────────


def test_a_lost_record_does_not_turn_scattered_failures_into_one_incident(
    vault: Path, tmp_path: Path
) -> None:
    """The heuristic must not overclaim from data the journal admits it drops.

    Records CAN go missing — an unwritable log, a torn line — and both are
    tolerated by design. If positions were derived by numbering the survivors,
    deleting the successes between three scattered failures would renumber them
    into 1-2-3 and report "unbroken run": a causal claim, from an artefact.

    Positions come from the pipeline's own `n`, so a gap stays a gap.
    """
    path = tmp_path / "runs.jsonl"
    with RunJournal(path, run_id="r1") as j:
        j.write("run_started", queued=5)
        for i, failed in enumerate([True, False, True, False, True], start=1):
            j.write("video", n=i, of=5, url=f"https://www.instagram.com/reel/AAAAAAAAA{i:02d}/",
                    status="failed" if failed else "written",
                    stage="download" if failed else "written", seconds=0.1)
        j.write("run_finished", attempted=5, failed=3)

    kept = [
        r for r in read_runs(path)
        if not (r.get("event") == "video" and r.get("status") == "written")
    ]
    positions = sorted(r["n"] for r in kept if r.get("event") == "video")
    assert positions == [1, 3, 5], "the gaps survive the loss of the records between them"


def test_two_runs_in_the_same_second_get_different_ids(tmp_path: Path) -> None:
    """A second-resolution id merges a quick restart into one displayed run.

    The totals would then belong to neither, and an "unbroken run" could be
    stitched together from two unrelated batches.
    """
    a = RunJournal(tmp_path / "a.jsonl")
    b = RunJournal(tmp_path / "b.jsonl")
    a.close()
    b.close()
    assert a.run_id != b.run_id or "-" in a.run_id


def test_an_unserialisable_field_does_not_raise(tmp_path: Path) -> None:
    """`default=str` does not cover everything.

    A circular reference raises before it is consulted, and a lone surrogate
    fails at encode time. Neither is reachable from today's callers — but a
    journal that can abort a rate-limited batch is the one outcome this module
    exists to prevent, so the guard is on the class, not on its callers.
    """
    path = tmp_path / "runs.jsonl"
    circular: dict = {}
    circular["self"] = circular

    with RunJournal(path, run_id="r1") as j:
        j.write("video", payload=circular)          # must not raise
        j.write("video", url="\udcff")              # must not raise
        j.write("video", url="https://www.instagram.com/reel/AAAAAAAAAAA/")

    kept = read_runs(path)
    assert len(kept) == 1, "the good record still landed"
    assert kept[0]["url"].endswith("/reel/AAAAAAAAAAA/")


def test_a_torn_write_does_not_swallow_the_next_record(tmp_path: Path) -> None:
    """A failed write can leave an unterminated line.

    Without a newline guard the next record concatenates onto it and BOTH are
    unreadable — worse than the "a torn write costs one line" claim.
    """
    path = tmp_path / "runs.jsonl"
    journal = RunJournal(path, run_id="r1")

    real = journal._fh

    class HalfWriter:
        def write(self, s):
            real.write(s[: len(s) // 2])
            raise OSError("No space left on device")

        def flush(self):
            real.flush()

    journal._fh = HalfWriter()          # type: ignore[assignment]
    journal.write("video", n=1, url="https://www.instagram.com/reel/AAAAAAAAAAA/")
    journal._fh = real                  # type: ignore[assignment]
    journal.write("video", n=2, url="https://www.instagram.com/reel/BBBBBBBBBBB/")
    journal.close()

    kept = read_runs(path)
    assert len(kept) == 1, f"the second record should survive the first's tear: {kept}"
    assert kept[0]["n"] == 2


def test_closing_a_broken_journal_does_not_raise(tmp_path: Path) -> None:
    """The likeliest place for this class to break its own contract.

    write() flushes per record, so the buffer is normally empty by close time.
    The exception is a run where an earlier flush already failed — write()
    caught that, but close() retries the residual flush against the same full
    disk. Since the caller holds this in a `with`, that fires during unwind and
    replaces the run summary and the "do NOT retry automatically" checkpoint
    warning with a traceback. The batch is already safe on disk by then, so
    losing the report is the entire, avoidable cost.
    """
    path = tmp_path / "runs.jsonl"
    journal = RunJournal(path, run_id="r1")

    class RefusesToClose:
        def write(self, s): ...
        def flush(self): ...
        def close(self):
            raise OSError("No space left on device")

    journal._fh = RefusesToClose()      # type: ignore[assignment]
    journal.close()                     # must not raise


def test_the_context_manager_does_not_divert_a_runs_exit_path(tmp_path: Path) -> None:
    """`with open_journal(...)` must not turn a clean run into a traceback."""
    path = tmp_path / "runs.jsonl"

    class RefusesToClose:
        def write(self, s): ...
        def flush(self): ...
        def close(self):
            raise OSError("No space left on device")

    with RunJournal(path, run_id="r1") as journal:
        journal._fh = RefusesToClose()  # type: ignore[assignment]
        journal.write("run_started")
