"""Processed-URL store. SQLite, keyed on canonical URL.

    # ── DELIBERATELY NAIVE: failure state handling ──────────────────────────
    # CURRENT: three states, but `mark_failed` writes FAILED for every error
    #   regardless of cause, and FAILED counts as processed forever.
    #
    # WHY THAT IS INSUFFICIENT: a transient error is indistinguishable from a
    #   permanent one. One network blip, one rate-limit, one interrupted run,
    #   and that video is excluded from every future queue — silently, because
    #   it looks exactly like a deleted video. On Instagram the caption note
    #   survives; on TikTok there is no caption, so the video is simply absent
    #   and nothing reports it. The `attempts` column is written but never
    #   read, which is the tell.
    #
    # INTENDED: decide whether RETRYABLE is a distinct state or a predicate
    #   over (state, attempts, last_attempt); whether retries need backoff
    #   before re-entering the queue; and what a STOPPED run leaves behind so
    #   resuming is correct. This interlocks with the download failure taxonomy
    #   — that decides which bucket an error lands in, this decides what each
    #   bucket costs. See tests/test_store.py.
    #
    # INVARIANT — do not change: the store is authoritative and keys on URL
    #   only. It must never consult the vault to decide what is processed. If
    #   it did, deleting a note would silently re-download and re-summarise it.
    #   The isolation test ("delete a note, re-run, it must not reappear")
    #   holds that line.
    # ────────────────────────────────────────────────────────────────────────
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

    def mark_failed(self, url: str, error: str, terminal: bool = True) -> None:
        """Record a failure.

        NOTE: `terminal` is accepted and then ignored — every failure is
        written as FAILED, which `processed_urls()` treats as final. The
        parameter exists so a real taxonomy has something to call.
        """
        self.conn.execute(
            "UPDATE videos SET state=?, error=?, last_attempt=?, "
            "attempts=attempts+1 WHERE url=?",
            (State.FAILED, error[:500], _now(), url),
        )
        self.conn.commit()

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
        cur = self.conn.execute(
            "SELECT url, platform, error, attempts, last_attempt FROM videos "
            "WHERE state=? ORDER BY last_attempt DESC",
            (State.FAILED,),
        )
        return list(cur)
