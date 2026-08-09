"""Tests for the vault write guard.

These are the tests that back the one non-negotiable constraint. Each case maps
to a specific way a naive guard leaks writes outside the vault.

Read alongside the commentary at the top of src/doomnotes/vault.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from doomnotes.vault import (
    VaultGuardError,
    VaultWriter,
    check_vault_root,
    resolve_in_vault,
)

MAIN_VAULT = Path(
    "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Starting Vault"
).expanduser()


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "ai-notes-vault"
    (root / "_transcripts").mkdir(parents=True)
    (root / "_meta").mkdir()
    return root


# --------------------------------------------------------------------------
# The happy path, so the guard is proven to permit as well as refuse.
# --------------------------------------------------------------------------


def test_writes_note_inside_vault(vault: Path) -> None:
    w = VaultWriter(vault)
    p = w.write_text("some-note.md", "hello")
    assert p.read_text() == "hello"
    assert p.parent == vault.resolve()


def test_writes_into_subdirectory(vault: Path) -> None:
    w = VaultWriter(vault)
    p = w.write_text("_transcripts/clip.md", "raw asr")
    assert p.read_text() == "raw asr"


# --------------------------------------------------------------------------
# Escapes. One test per mechanism.
# --------------------------------------------------------------------------


def test_rejects_dotdot_escape(vault: Path) -> None:
    with pytest.raises(VaultGuardError, match="escapes vault"):
        resolve_in_vault(vault, "../outside.md")


def test_rejects_deep_dotdot_into_main_vault(vault: Path) -> None:
    """Textually 'under' the root; on disk it is the main vault."""
    escape = os.path.relpath(MAIN_VAULT / "stolen.md", vault)
    assert escape.startswith("..")
    with pytest.raises(VaultGuardError):
        resolve_in_vault(vault, escape)


def test_rejects_absolute_main_vault_path(vault: Path) -> None:
    with pytest.raises(VaultGuardError, match="outside vault"):
        resolve_in_vault(vault, MAIN_VAULT / "stolen.md")


def test_rejects_symlink_pointing_out_of_vault(vault: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (vault / "escape-hatch").symlink_to(outside, target_is_directory=True)
    with pytest.raises(VaultGuardError, match="escapes vault"):
        resolve_in_vault(vault, "escape-hatch/note.md")


def test_rejects_writing_to_vault_root_itself(vault: Path) -> None:
    with pytest.raises(VaultGuardError, match="vault root itself"):
        resolve_in_vault(vault, ".")


def test_sibling_prefix_directory_is_not_inside_the_vault(tmp_path: Path) -> None:
    """The bug that defeats `str.startswith`.

    '/…/ai-notes-vault-backup' starts with the string '/…/ai-notes-vault', so a
    prefix check accepts it. It is a sibling directory, not a child.
    """
    vault = tmp_path / "ai-notes-vault"
    vault.mkdir()
    sibling = tmp_path / "ai-notes-vault-backup"
    sibling.mkdir()

    assert str(sibling).startswith(str(vault))  # the naive check would pass

    with pytest.raises(VaultGuardError):
        resolve_in_vault(vault, sibling / "note.md")


# --------------------------------------------------------------------------
# Root-level refusals — the isolation test lives here.
# --------------------------------------------------------------------------


def test_rejects_main_vault_as_root() -> None:
    """Isolation test: point config at the main vault, get a hard stop."""
    with pytest.raises(VaultGuardError, match="iCloud-synced"):
        check_vault_root(MAIN_VAULT)


def test_rejects_any_icloud_obsidian_vault_as_root() -> None:
    other = Path(
        "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Claude Code Learning"
    ).expanduser()
    with pytest.raises(VaultGuardError, match="iCloud-synced"):
        check_vault_root(other)


@pytest.mark.parametrize("bad", ["/", "~", "~/Documents", "~/Downloads", "~/Desktop"])
def test_rejects_toplevel_directories_as_root(bad: str) -> None:
    with pytest.raises(VaultGuardError, match="not a dedicated vault"):
        check_vault_root(bad)


def test_rejects_nonexistent_root(tmp_path: Path) -> None:
    with pytest.raises(VaultGuardError, match="does not exist"):
        check_vault_root(tmp_path / "typo-vault")


# --------------------------------------------------------------------------
# A refused write must leave nothing behind — not even a directory.
# --------------------------------------------------------------------------


def test_refused_write_creates_nothing(vault: Path, tmp_path: Path) -> None:
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    w = VaultWriter(vault)
    with pytest.raises(VaultGuardError):
        w.write_text("../../escaped/deep/note.md", "should never exist")

    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert before == after, "guard ran after mkdir — directories were created"


def test_guard_runs_before_mkdir_for_main_vault_target(vault: Path) -> None:
    """The specific case worth being certain about."""
    w = VaultWriter(vault)
    target_dir = MAIN_VAULT / "doomnotes-should-never-appear"
    with pytest.raises(VaultGuardError):
        w.write_text(target_dir / "note.md", "nope")
    assert not target_dir.exists()


# --------------------------------------------------------------------------
# Atomicity.
# --------------------------------------------------------------------------


def test_no_temp_files_left_behind(vault: Path) -> None:
    w = VaultWriter(vault)
    w.write_text("note.md", "content")
    leftovers = [p.name for p in vault.rglob(".doomnotes-*")]
    assert leftovers == []


def test_overwrite_is_atomic_replace(vault: Path) -> None:
    w = VaultWriter(vault)
    w.write_text("note.md", "v1")
    p = w.write_text("note.md", "v2")
    assert p.read_text() == "v2"


def test_write_pair_writes_transcript_and_note(vault: Path) -> None:
    w = VaultWriter(vault)
    note, transcript = w.write_pair(
        "my-note.md",
        "[[_transcripts/my-note|Raw transcript]]",
        "_transcripts/my-note.md",
        "the raw asr text",
    )
    assert transcript is not None
    assert transcript.read_text() == "the raw asr text"
    assert note.read_text().startswith("[[_transcripts/my-note")


def test_write_pair_rejects_bad_transcript_path_without_orphaning_note(
    vault: Path,
) -> None:
    """Both paths are guarded up front, so no note is written for a bad pair."""
    w = VaultWriter(vault)
    with pytest.raises(VaultGuardError):
        w.write_pair("my-note.md", "note body", "../../evil.md", "transcript")
    assert not (vault / "my-note.md").exists()


def test_write_pair_without_transcript(vault: Path) -> None:
    """Caption-only Instagram notes have no transcript and no wikilink."""
    w = VaultWriter(vault)
    note, transcript = w.write_pair("caption-only.md", "body", None, None)
    assert transcript is None
    assert note.read_text() == "body"
