"""Store tests.

    # The two tests marked `xfail` below describe behaviour the deliberately
    # naive store does not have. They are not bugs to file — they are the
    # specification for the retry policy. Implementing it makes them pass and
    # the markers removable. Until then they document the gap rather than
    # pretending the baseline is complete.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doomnotes.models import VideoRef
from doomnotes.store import State, Store

A = VideoRef("https://www.instagram.com/reel/AAA/", "instagram", caption_source="export", caption="a", source_order=0)
B = VideoRef("https://www.instagram.com/reel/BBB/", "instagram", caption_source="export", caption="b", source_order=1)
C = VideoRef("https://www.tiktokv.com/share/video/111/", "tiktok", source_order=0)


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    with Store(tmp_path / "state.db") as s:
        yield s


def test_register_is_idempotent(store: Store) -> None:
    assert store.register([A, B]) == 2
    assert store.register([A, B]) == 0
    assert store.state_of(A.url) is State.PENDING


def test_register_does_not_reset_completed_state(store: Store) -> None:
    """Re-parsing an export must not resurrect finished work."""
    store.register([A])
    store.mark_done(A.url)
    store.register([A])
    assert store.state_of(A.url) is State.DONE


def test_filter_unprocessed_excludes_done(store: Store) -> None:
    store.register([A, B, C])
    store.mark_done(A.url)
    remaining = store.filter_unprocessed([A, B, C])
    assert [r.url for r in remaining] == [B.url, C.url]


def test_queue_preserves_newest_first_order(store: Store) -> None:
    """Input order IS newest-first; the store must not reorder it."""
    store.register([A, B])
    assert [r.url for r in store.queue([A, B])] == [A.url, B.url]


def test_queue_respects_limit(store: Store) -> None:
    store.register([A, B, C])
    assert len(store.queue([A, B, C], limit=2)) == 2


def test_store_never_consults_the_filesystem(store: Store, tmp_path: Path) -> None:
    """The property the isolation test depends on.

    Deleting a note must not make the store forget. If `filter_unprocessed` ever
    checks whether a note file exists, deleting one silently re-downloads and
    re-summarises it — which is the opposite of what re-run semantics promise.
    """
    store.register([A])
    store.mark_done(A.url)
    (tmp_path / "some-note.md").write_text("a note")
    (tmp_path / "some-note.md").unlink()
    assert store.filter_unprocessed([A]) == []


def test_failure_is_recorded_with_message(store: Store) -> None:
    store.register([C])
    store.mark_failed(C.url, "Video unavailable")
    assert store.state_of(C.url) is State.FAILED
    rows = store.failures()
    assert rows[0]["url"] == C.url
    assert "unavailable" in rows[0]["error"]


def test_stats_group_by_platform_and_state(store: Store) -> None:
    store.register([A, B, C])
    store.mark_done(A.url)
    store.mark_failed(C.url, "gone")
    stats = store.stats()
    assert stats["instagram"][State.DONE] == 1
    assert stats["tiktok"][State.FAILED] == 1


# ── the retry policy (decision 2.1) ──────────────────────────────────────


def test_retryable_failure_returns_to_the_queue(store: Store) -> None:
    store.register([C])
    store.mark_failed(C.url, "HTTP Error 429: Too Many Requests", terminal=False)
    assert [r.url for r in store.filter_unprocessed([C])] == [C.url]


def test_retries_are_bounded_but_do_happen(store: Store) -> None:
    """Both halves matter, and the baseline only satisfies the second.

    A cap that works by never retrying is not a cap. Asserting the requeue
    *and* the eventual stop is what distinguishes a real retry policy from the
    baseline's accidental one.
    """
    store.register([C])
    store.mark_failed(C.url, "network blip", terminal=False)
    assert [r.url for r in store.filter_unprocessed([C])] == [C.url], (
        "a retryable failure should return to the queue"
    )

    for _ in range(5):
        store.mark_failed(C.url, "network blip", terminal=False)
    assert store.filter_unprocessed([C]) == [], "retries should stop at a cap"


def test_a_terminal_failure_is_never_retried(store: Store) -> None:
    """A deleted video must not consume three runs proving it is still deleted."""
    store.register([C])
    store.mark_failed(C.url, "Video unavailable", terminal=True)
    assert store.filter_unprocessed([C]) == []
    assert store.state_of(C.url) is State.FAILED


def test_the_retry_backlog_is_visible(store: Store) -> None:
    """A queue you cannot see is the failure mode this decision exists to fix.

    `stats()` is what `doomnotes status` prints, and `failures()` is the list
    under it. A video waiting to retry has to appear in both, or the only way
    to learn it is pending again is to run and watch.
    """
    store.register([C])
    store.mark_failed(C.url, "HTTP Error 429: Too Many Requests", terminal=False)

    assert store.stats()["tiktok"][State.RETRYABLE] == 1
    rows = store.failures()
    assert [r["state"] for r in rows] == [State.RETRYABLE]
    assert rows[0]["attempts"] == 1
