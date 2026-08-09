"""Run journal — an append-only record of what each run did.

Task 7.2, and the last unfinished SUPERVISE item in the build order.

    # IT RECORDS. IT DOES NOT REACT.
    #
    # That is the whole design constraint, and it is deliberate rather than
    # minimalist. A journal that counted consecutive same-stage failures and
    # tripped a breaker would have made HANDS-ON #7.1's decision — what failure
    # isolation should actually do — and made it invisibly, inside a module
    # called "logging". So this writes, and nothing else. Deciding what to do
    # about what it records is 7.1's job.

Why it exists at all, concretely. #7.1 asks which failures share a cause, and
right now there is nothing to answer that from:

  - the run summary gives totals, so 29 failures look like 29 problems;
  - `state.db` keeps the latest state per URL and overwrites it each attempt,
    so the shape of a run is gone as soon as the next one starts.

Neither records that videos 12 through 30 all failed at the summarize stage
within 400 ms of each other, which is what a dead Ollama looks like from the
outside. This does.

It is also the evidence for two other decisions: #4.2 wants real per-video
transcription times on real audio rather than synthetic clips, and #3.3 wants to
know what the actual inter-request spacing looked like once failures are in the
mix, not what the configured range says it should be.

**Format: JSON Lines**, one object per event, opened in append mode. Not a
table and not a single JSON document, because a run can be killed at any moment
and a torn write should cost one line rather than the file. Everything a reader
needs is on each line, so `grep` and `jq` both work without a schema.

**Privacy: it contains your saved-video URLs**, exactly like `state.db` does. It
lives under `data/`, which is gitignored, and the pre-commit hook blocks
`data/*` by path. It is never written into the vault.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

__all__ = ["Journal", "RunJournal", "NullJournal", "open_journal", "read_runs"]


class Journal(Protocol):
    """What the pipeline needs. Two methods, neither of which returns anything.

    A Protocol rather than a base class so a test can pass a plain list-appender
    without importing anything from here.
    """

    def write(self, event: str, **fields: Any) -> None: ...

    def close(self) -> None: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _ends_mid_line(path: Path) -> bool:
    """Whether `path` exists and its last byte is not a newline.

    One seek and one byte, so it costs nothing at open time. Unreadable or
    absent counts as "fine": the caller is about to try opening it anyway and
    will report that failure properly.
    """
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with path.open("rb") as fh:
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) != b"\n"
    except OSError:
        return False


class NullJournal:
    """Records nothing. The default, so nothing depends on journalling working."""

    run_id = ""

    def write(self, event: str, **fields: Any) -> None:
        return

    def close(self) -> None:
        return

    def __enter__(self) -> "NullJournal":
        return self

    def __exit__(self, *exc: object) -> None:
        return


class RunJournal:
    """Append-only JSONL writer for one run.

    Every method is failure-tolerant on purpose. A journal that can abort a
    batch is worse than no journal: it would let an observability feature cost
    you the download budget it exists to explain.
    """

    def __init__(self, path: str | os.PathLike[str], run_id: str | None = None) -> None:
        self.path = Path(path).expanduser()
        # The pid disambiguates two runs started in the same second — a quick
        # restart after Ctrl-C, or two terminal tabs. Without it they share an
        # id, and a reader merges two unrelated batches into one "run" whose
        # totals belong to neither.
        self.run_id = run_id or (
            f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{os.getpid():d}"
        )
        self._fh = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Checked against the FILE, not carried in memory. The guard against
            # a torn line has to survive the process that tore it: a run killed
            # with SIGKILL, or one whose final flush failed, leaves an
            # unterminated last line, and the next run appends straight onto it.
            # That silently loses the new run's opening record — and if the torn
            # line was a `run_finished`, a completed batch reads back as one that
            # never finished, which is the one distinction this file exists to
            # preserve. Runs are separate processes days apart, so inheriting a
            # torn file is the ordinary case, not an exotic one.
            self._needs_newline = _ends_mid_line(self.path)
            self._fh = self.path.open("a", encoding="utf-8")
        except OSError as exc:
            self._needs_newline = False
            log.warning("run journal unavailable at %s: %s", self.path, exc)

    def write(self, event: str, **fields: Any) -> None:
        if self._fh is None:
            return
        record = {"ts": _now(), "run_id": self.run_id, "event": event, **fields}
        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 - a logger must never raise
            # `default=str` does not cover everything: a circular reference
            # raises before it is consulted, and a lone surrogate fails at
            # encode time. Neither is reachable from today's callers, but a
            # journal that can abort a rate-limited batch is the one outcome
            # this module exists to avoid.
            log.warning("could not serialise journal record %s: %s", event, exc)
            return

        try:
            # A previous write may have failed partway through, leaving an
            # unterminated line. Prefixing a newline costs one blank line and
            # keeps the torn record from swallowing this one too — otherwise
            # both are lost, not just the torn one.
            self._fh.write(("\n" if self._needs_newline else "") + line + "\n")
            self._needs_newline = False
            # Flushed per record, so a killed PROCESS still has every video it
            # finished — the bytes are in the kernel's hands. Deliberately not
            # fsync'd: that would only additionally cover a power loss or kernel
            # panic, which is not the failure this is protecting against.
            self._fh.flush()
        except Exception as exc:  # noqa: BLE001 - a logger must never raise
            self._needs_newline = True
            log.warning("could not journal %s: %s", event, exc)

    def close(self) -> None:
        """Closing must not raise either — and this is the likeliest place to.

        `write()` flushes per record, so the buffer is normally empty by now.
        The exception is a run where an earlier flush already failed: `write()`
        caught and logged that, but `close()` retries the residual flush against
        the same full disk and raises.

        Because the caller holds this in a `with`, that exception fires during
        unwind and diverts the exit path — the run summary never prints, the
        "a checkpoint was hit, do NOT retry automatically" warning never
        prints, and the documented exit codes are replaced by a traceback. The
        batch itself is already safe on disk by then, which makes losing the
        report the entire cost, and an avoidable one.
        """
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception as exc:  # noqa: BLE001 - a logger must never raise
                log.warning("could not close the run journal at %s: %s", self.path, exc)
            finally:
                self._fh = None

    def __enter__(self) -> "RunJournal":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_journal(path: str | os.PathLike[str] | None) -> RunJournal | NullJournal:
    """A journal for `path`, or a no-op when journalling is switched off."""
    return NullJournal() if path is None else RunJournal(path)


def read_runs(path: str | os.PathLike[str]) -> list[dict]:
    """Every record in the journal, skipping torn lines.

    A partial final line is the expected cost of a killed run, not corruption,
    so it is dropped rather than raised on.

    Read as BYTES and decoded per line. Decoding the whole file at once looks
    equivalent and is not: a tear can land inside a multi-byte character — any
    non-ASCII text in a caption, handle or URL makes that reachable — and a
    single whole-file decode then raises before any line is examined. The
    per-line tolerance below would never run, and one torn tail would cost
    every good record before it. Which is the opposite of the promise.

    `errors="replace"` keeps a mangled line parseable-or-skippable rather than
    fatal; a line whose bytes were damaged fails JSON parsing and is dropped,
    exactly as a truncated one is.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return []
    out: list[dict] = []
    for raw in p.read_bytes().split(b"\n"):
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            # RecursionError needs deeply nested JSON, which a torn write can
            # never produce — a tear truncates a flat record. It is caught
            # anyway because the promise made above is "a bad line is dropped",
            # and a promise with an exception nobody can name is not one.
            continue
        # The signature says list[dict], so make that true here rather than
        # leaving every caller to re-derive it. A valid-JSON line that is not
        # an object is not a record.
        if isinstance(record, dict):
            out.append(record)
    return out
