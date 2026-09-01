"""The pre-commit secret guard, exercised against a real index.

Every URL in this file is invented (doomnotes:synthetic-urls). None of them
come from the real exports.

The guard had three defects at once, and none of them were visible from
reading it:

  * the synthetic-fixtures pragma stopped being honoured above the pipe buffer,
  * a file staged and then deleted from the working tree skipped the scan,
  * the canonical `tiktok.com/@handle/video/<id>` form — the exact URL the
    pipeline now requests — was not matched at all.

Two of those fail open. A guard nobody tests is a guard that quietly stops
guarding, so these run it as git does: against a throwaway repository's index.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parent.parent / "scripts" / "check_no_secrets.sh"

# Six of each, one over the guard's threshold of five.
SHARE_FORM = "\n".join(
    f"https://www.tiktokv.com/share/video/741234567890123456{i}/" for i in range(6)
)
CANONICAL_FORM = "\n".join(
    f"https://www.tiktok.com/@someone/video/741234567890123456{i}/" for i in range(6)
)
INSTAGRAM_FORM = "\n".join(
    f"https://www.instagram.com/reel/AAAAAAAAAA{i}/" for i in range(6)
)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A throwaway git repository. No network, no hooks, no real data."""
    if shutil.which("git") is None:  # pragma: no cover - git is a hard dependency
        pytest.skip("git not available")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    return tmp_path


def run_guard(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(GUARD)], cwd=repo, capture_output=True, text=True
    )


def stage(repo: Path, name: str, body: str) -> Path:
    path = repo / name
    path.write_text(body, encoding="utf-8")
    subprocess.run(["git", "add", name], cwd=repo, check=True)
    return path


@pytest.mark.parametrize(
    "corpus,form",
    [(SHARE_FORM, "share"), (CANONICAL_FORM, "canonical"), (INSTAGRAM_FORM, "instagram")],
)
def test_a_bulk_corpus_is_blocked_in_every_url_form(repo: Path, corpus: str, form: str) -> None:
    """The canonical TikTok form is the one `fetch_url` requests.

    It went unmatched while the share form was caught, so the guard would have
    passed a corpus of exactly the URLs this project generates.
    """
    stage(repo, "leak.md", corpus)
    result = run_guard(repo)
    assert result.returncode == 1, f"{form} form was not blocked:\n{result.stdout}"
    assert "BLOCKED" in result.stdout


def test_a_file_staged_then_deleted_is_still_scanned(repo: Path) -> None:
    """What gets committed is the staged blob, not the working tree.

    Skipping files missing from the working tree let a corpus into a commit
    while the guard reported success — the one failure mode a secret guard
    must not have.
    """
    path = stage(repo, "leak.md", SHARE_FORM)
    path.unlink()

    result = run_guard(repo)
    assert result.returncode == 1, f"staged blob escaped the scan:\n{result.stdout}"
    assert "BLOCKED" in result.stdout


def test_the_synthetic_pragma_survives_a_large_file(repo: Path) -> None:
    """The pragma sits at the top; the file is far past the 16KB pipe buffer.

    `git show | grep -q PRAGMA` returned 141 here — grep exits at the match,
    git show dies of SIGPIPE, pipefail propagates it — so a successful match
    reported failure and the file was blocked.
    """
    padding = "\n".join(f"# filler line {i} " + "x" * 60 for i in range(600))
    stage(repo, "fixtures.py", f"# doomnotes:synthetic-urls\n{SHARE_FORM}\n{padding}\n")

    result = run_guard(repo)
    assert len(SHARE_FORM) + len(padding) > 16384, "must exceed the pipe buffer to be a test"
    assert result.returncode == 0, f"pragma was not honoured:\n{result.stdout}"
    assert "skipped" in result.stdout


def test_a_handful_of_example_urls_is_allowed(repo: Path) -> None:
    """A threshold, not a ban — documentation may cite an example."""
    stage(repo, "README.md", "See https://www.instagram.com/reel/AAAAAAAAAAA/ for one.")
    assert run_guard(repo).returncode == 0
