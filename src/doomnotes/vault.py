"""Vault write guard + atomic writes.

This module is the single enforcement point for the project's one non-negotiable
constraint: **never write outside the configured vault**.

Everything that touches the filesystem under the vault goes through `VaultWriter`.
There is deliberately no other write path.

    # ── Three rejected approaches, and why each fails ───────────────────────
    # tests/test_vault.py has a case for each.
    #
    # WHY NOT `str.startswith`:
    #     str(candidate).startswith(str(root))
    #   accepts  ~/Documents/ai-notes-vault-backup/x.md  when the root is
    #   ~/Documents/ai-notes-vault  — "ai-notes-vault-backup" starts with
    #   "ai-notes-vault". String prefixes don't respect path boundaries.
    #   This is the sibling-prefix bug and it is the most common version of it.
    #
    # WHY NOT checking the path before resolving:
    #     root / "../../Library/Mobile Documents/.../Starting Vault/x.md"
    #   is *textually* under the root. It is not under it on disk. Resolution
    #   must happen first, then comparison of the real locations.
    #
    # WHY NOT trusting resolution alone:
    #   a symlink inside the vault pointing anywhere else resolves out of the
    #   vault — which `.relative_to()` then catches. That is why both sides are
    #   resolved and the resolved forms compared, not one or the other.
    # ────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["VaultGuardError", "VaultWriter", "check_vault_root", "resolve_in_vault"]


class VaultGuardError(Exception):
    """Raised when a write would land outside the configured vault.

    This is a hard stop, never a warning. Callers must not catch it to continue.
    """


# Any vault root under one of these is refused outright. iCloud Mobile Documents
# covers the main vault and every other Obsidian vault synced through iCloud, so
# a mistyped config cannot quietly point the pipeline at real notes.
FORBIDDEN_ROOT_PREFIXES: tuple[Path, ...] = (
    Path("~/Library/Mobile Documents").expanduser(),
    Path("~/Library/CloudStorage").expanduser(),
)

# Refusing these outright stops a misconfigured root from turning the whole home
# directory into "the vault", where the containment check would pass trivially.
FORBIDDEN_EXACT: tuple[Path, ...] = (
    Path("/"),
    Path.home(),
    Path("~/Documents").expanduser(),
    Path("~/Downloads").expanduser(),
    Path("~/Desktop").expanduser(),
)


def check_vault_root(vault_root: str | os.PathLike[str]) -> Path:
    """Validate the configured vault root itself, before any path is joined to it.

    Returns the resolved root. Raises `VaultGuardError` if the root is unusable.

    This is the layer the isolation test exercises: pointing `config.toml` at a
    protected vault must fail here, before a single directory is created.
    """
    root = Path(vault_root).expanduser()
    resolved = root.resolve(strict=False)

    for exact in FORBIDDEN_EXACT:
        if resolved == exact.resolve(strict=False):
            raise VaultGuardError(
                f"Refusing to use {resolved} as a vault root: it is a top-level "
                f"directory, not a dedicated vault."
            )

    for prefix in FORBIDDEN_ROOT_PREFIXES:
        prefix_resolved = prefix.resolve(strict=False)
        if resolved == prefix_resolved or _is_within(resolved, prefix_resolved):
            raise VaultGuardError(
                f"Refusing to use {resolved} as a vault root: it is inside "
                f"{prefix_resolved}, which holds iCloud-synced Obsidian vaults "
                f"including the main vault. This pipeline never writes there."
            )

    # Requiring the root to already exist means a mistyped path fails loudly
    # instead of silently creating a brand-new tree somewhere unexpected.
    if not resolved.is_dir():
        raise VaultGuardError(
            f"Vault root {resolved} does not exist (or is not a directory). "
            f"Create it deliberately before running."
        )

    return resolved


def _is_within(candidate: Path, root: Path) -> bool:
    """True if `candidate` is at or below `root`. Both must already be resolved.

    Uses `relative_to`, which compares path *components*, so the sibling-prefix
    bug that defeats `str.startswith` cannot occur.
    """
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_in_vault(vault_root: str | os.PathLike[str], relative: str | os.PathLike[str]) -> Path:
    """Resolve `relative` against the vault and prove the result stays inside it.

    Returns the absolute, resolved target path. Raises `VaultGuardError` if the
    target escapes the vault by any means: `..`, an absolute path, or a symlink.

    Note the ordering — this function creates nothing. Validation completes
    before any caller is allowed to mkdir.
    """
    root = check_vault_root(vault_root)

    rel = Path(relative).expanduser()
    if rel.is_absolute():
        # pathlib's `root / absolute` silently discards the root. Rejecting it
        # explicitly gives a better error than a confusing containment failure.
        candidate = rel.resolve(strict=False)
        if not _is_within(candidate, root):
            raise VaultGuardError(
                f"Refusing to write to {candidate}: absolute path outside vault {root}."
            )
        return candidate

    candidate = (root / rel).resolve(strict=False)
    if not _is_within(candidate, root):
        raise VaultGuardError(
            f"Refusing to write to {candidate}: escapes vault {root} "
            f"(via '..' or a symlink in {rel})."
        )
    if candidate == root:
        raise VaultGuardError(f"Refusing to write to the vault root itself: {root}.")
    return candidate


class VaultWriter:
    """The only sanctioned way to write into the vault.

    Every write is guarded, then staged to a temp file in the *destination
    directory* and moved into place with `os.replace`.

    Why the temp file must be a sibling of the destination, not in /tmp:
    `os.replace` is only atomic within a single filesystem. A cross-device move
    degrades to copy-then-delete, which is exactly the half-written file the
    staging was meant to prevent.

    Why the temp file is not named `*.md`: Obsidian watches the vault directory.
    A visible half-written `.md` would be indexed mid-write.
    """

    def __init__(self, vault_root: str | os.PathLike[str]) -> None:
        self.root = check_vault_root(vault_root)

    def resolve(self, relative: str | os.PathLike[str]) -> Path:
        return resolve_in_vault(self.root, relative)

    def write_text(self, relative: str | os.PathLike[str], text: str) -> Path:
        """Guard, then atomically write `text` to `relative` inside the vault."""
        # Guard first. Nothing below this line runs for a rejected path — in
        # particular no mkdir, so a refused write leaves zero trace on disk.
        target = self.resolve(relative)

        target.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(
            dir=target.parent, prefix=".doomnotes-", suffix=".tmp"
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return target

    def write_pair(
        self,
        note_rel: str | os.PathLike[str],
        note_text: str,
        transcript_rel: str | os.PathLike[str] | None,
        transcript_text: str | None,
    ) -> tuple[Path, Path | None]:
        """Write a note and its transcript so a wikilink can never dangle.

        The transcript lands first. If the note references `[[_transcripts/x]]`,
        the target already exists by the time the note is visible to Obsidian.
        Both paths are guarded before either is written, so a bad transcript path
        cannot leave an orphan note behind.
        """
        self.resolve(note_rel)
        if transcript_rel is not None:
            self.resolve(transcript_rel)

        transcript_path: Path | None = None
        if transcript_rel is not None and transcript_text is not None:
            transcript_path = self.write_text(transcript_rel, transcript_text)
        note_path = self.write_text(note_rel, note_text)
        return note_path, transcript_path
