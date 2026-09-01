"""Processed-URL store. SQLite, keyed on canonical URL.

States and what each costs (decision 2.1):

    PENDING    never attempted, or attempted in a run that halted before
               reaching it. Enters every queue.
    RETRYABLE  failed for a cause that may not recur. Re-enters the queue on
               the NEXT run -- there is no in-run retry, because the condition
               that caused the failure almost always outlives the run, and
               retrying a 429 sooner is precisely the wrong move. The daily
               cadence is the backoff.
    FAILED     terminal, or retried MAX_ATTEMPTS times without success. Never
               re-enters a queue.
    DONE       a note was written.

`processed_urls()` is DONE and FAILED. RETRYABLE is deliberately absent, which
is the whole change: a transient error no longer buries a video.

Retries are bounded AND they happen. A cap that works by never retrying is not
a cap, which is what the previous baseline had -- `attempts` was written and
never read.

INVARIANT -- do not change: the store is authoritative and keys on URL only. It
must never consult the vault to decide what is processed. If it did, deleting a
note would silently re-download and re-summarise it. The isolation test
("delete a note, re-run, it must not reappear") holds that line.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from doomnotes.models import VideoRef


class State(StrEnum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    RETRYABLE = "retryable"


# Attempts before a retryable failure is written off. Counted across runs, not
# within one: under next-run-only retrying, three attempts is three runs.
MAX_ATTEMPTS = 3


SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    url          TEXT PRIMARY KEY,
    platform     TEXT NOT NULL,
    state        TEXT NOT NULL,
    error        TEXT,
    attempts     INTEGER NOT NULL DEFAULT 0,
    source_order INTEGER,
    first_seen   TEXT NOT NULL,
    last_attempt TEXT,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_videos_state ON videos(state);
CREATE INDEX IF NOT EXISTS idx_videos_platform_order ON videos(platform, source_order);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- registration -----------------------------------------------------

    def register(self, refs: Iterable[VideoRef]) -> int:
        """Record refs as PENDING. Existing rows are left untouched.

        `INSERT OR IGNORE` is what makes re-running a parse harmless: a URL
        already marked DONE keeps its state instead of resetting to PENDING.
        """
        rows = [
            (r.url, r.platform, State.PENDING, r.source_order, _now())
            for r in refs
        ]
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO videos "
            "(url, platform, state, source_order, first_seen) VALUES (?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return cur.rowcount

    # -- queueing ---------------------------------------------------------

    def processed_urls(self) -> set[str]:
        """URLs this store considers finished with — DONE or FAILED.

        NOTE: FAILED counts as processed forever. See the module docstring.
        """
        cur = self.conn.execute(
            "SELECT url FROM videos WHERE state IN (?, ?)",
            (State.DONE, State.FAILED),
        )
        return {row["url"] for row in cur}

    def filter_unprocessed(self, refs: Iterable[VideoRef]) -> list[VideoRef]:
        """Drop refs the store considers finished. Preserves input order."""
        done = self.processed_urls()
        return [r for r in refs if r.url not in done]

    def queue(
        self,
        refs: Sequence[VideoRef],
        limit: int | None = None,
    ) -> list[VideoRef]:
        """Newest-first batch for one run.

        Input order IS newest-first — both exports are newest-first on disk and
        the parsers preserve file order. There is no shared key on which to
        merge Instagram and TikTok (Instagram has no save timestamps at all), so
        callers drain each source as its own queue.
        """
        pending = self.filter_unprocessed(refs)
        return pending[:limit] if limit is not None else pending

    # -- outcomes ---------------------------------------------------------

    def mark_done(self, url: str) -> None:
        self.conn.execute(
            "UPDATE videos SET state=?, error=NULL, completed_at=?, "
            "last_attempt=?, attempts=attempts+1 WHERE url=?",
            (State.DONE, _now(), _now(), url),
        )
        self.conn.commit()

    def mark_failed(
        self,
        url: str,
        error: str,
        terminal: bool = True,
        counts_as_attempt: bool = True,
    ) -> None:
        """Record a failure.

        A terminal failure is final immediately. A retryable one returns to the
        queue until it has been attempted `MAX_ATTEMPTS` times, after which it
        becomes terminal — otherwise a permanently broken video would be
        retried on every run forever.

        `counts_as_attempt=False` records the failure without spending one of
        those attempts, for causes that are not about this video at all — a
        stale cookie, an expired session. The request failed on a precondition,
        so the video was never really tried, and a video that is never tried
        must not be written off. Such a row can never reach the cap from this
        path, which is the point: the fault is in the credentials and gets
        fixed there, not by exhausting the queue.

        The attempt is counted whatever the cause, including a batch-wide one
        such as a rate limit. That is deliberate: it is what bounds the retry.
        The cost is that a run which is rate-limited throughout spends one
        attempt on every video in it, so the write-off is surfaced by state in
        `stats()` rather than left to be discovered.
        """
        state = State.FAILED
        if not terminal:
            if counts_as_attempt:
                attempted = self.attempts_for(url) + 1
                state = State.FAILED if attempted >= MAX_ATTEMPTS else State.RETRYABLE
            else:
                # Never written off on this path: an uncounted failure cannot
                # reach the cap, however many times it happens.
                state = State.RETRYABLE

        self.conn.execute(
            "UPDATE videos SET state=?, error=?, last_attempt=?, "
            "attempts=attempts+? WHERE url=?",
            (state, error[:500], _now(), 1 if counts_as_attempt else 0, url),
        )
        self.conn.commit()

    def attempts_for(self, url: str) -> int:
        """How many times this URL has already been attempted.

        The pipeline reads this to know whether the attempt it is about to make
        is the last one, which decides whether a retryable failure holds the
        video for another run or falls back to a caption-only note.
        """
        row = self.conn.execute(
            "SELECT attempts FROM videos WHERE url=?", (url,)
        ).fetchone()
        return row["attempts"] if row else 0

    def state_of(self, url: str) -> State | None:
        cur = self.conn.execute("SELECT state FROM videos WHERE url=?", (url,))
        row = cur.fetchone()
        return State(row["state"]) if row else None

    # -- reporting --------------------------------------------------------

    def stats(self) -> dict[str, dict[str, int]]:
        cur = self.conn.execute(
            "SELECT platform, state, COUNT(*) AS n FROM videos "
            "GROUP BY platform, state"
        )
        out: dict[str, dict[str, int]] = {}
        for row in cur:
            out.setdefault(row["platform"], {})[row["state"]] = row["n"]
        return out

    def failures(self) -> list[sqlite3.Row]:
        """Every video that failed, written off or still waiting to retry.

        RETRYABLE rows are included because a queue you cannot see is the
        problem this decision exists to fix. `state` distinguishes them.
        """
        cur = self.conn.execute(
            "SELECT url, platform, state, error, attempts, last_attempt FROM videos "
            "WHERE state IN (?, ?) ORDER BY last_attempt DESC",
            (State.FAILED, State.RETRYABLE),
        )
        return list(cur)
