#!/usr/bin/env python
"""Cross-reference the repository against the real platform exports.

The pre-commit hook is a shape check: it blocks credential-shaped strings and
bulk URL lists without knowing what the actual saved videos are. This is the
identity check — it reads the real exports and asks whether any genuine
shortcode, video id or creator handle has ended up in a tracked file.

It is deliberately NOT a hook: it needs the exports present, and a hook that
breaks when a file in ~/Downloads is missing is a hook people disable. Run it
before making a repo public, or after a large documentation change.

Run: make audit-leaks
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from doomnotes.sources import ig_export, tiktok_export  # noqa: E402


def tracked_files() -> list[Path]:
    # splitlines(), not split(): git emits one path per line, and a filename
    # containing a space would otherwise be torn into two non-existent paths
    # and silently dropped from the scan.
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    staged = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--cached", "--name-only"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return [REPO / f for f in sorted(set(out) | set(staged)) if (REPO / f).is_file()]


def main() -> int:
    identifiers: dict[str, str] = {}
    # A security tool that cannot complete its scan must not report success.
    # Every degraded path below flips this, and an incomplete scan exits non-zero
    # even when it found nothing — "checked 0 identifiers" is not "clean".
    complete = True

    try:
        ig_refs, _ = ig_export.parse()
        for r in ig_refs:
            identifiers[r.url.rstrip("/").rsplit("/", 1)[-1]] = "instagram shortcode"
            if r.author:
                identifiers[r.author.lstrip("@")] = "creator handle"
    except FileNotFoundError:
        print("  ! Instagram export NOT FOUND — that half of the scan did not run")
        complete = False

    # The owner's OWN identity, which the exports never contain as data but
    # encode in their directory name: `instagram-<account>-<date>-<token>`.
    # Deriving identifiers only from creator handles inside the export misses
    # the account name and archive token entirely — that gap put both into a
    # published commit once already.
    for directory in (Path("~/Downloads").expanduser(), REPO.parent):
        if not directory.is_dir():
            continue
        for match in directory.glob("instagram-*"):
            parts = match.name.split("-")
            if len(parts) >= 4:
                identifiers[parts[1]] = "account name (from export dir)"
                identifiers[parts[-1]] = "export archive token"

    try:
        tt_refs, _ = tiktok_export.parse()
        for r in tt_refs:
            vid = tiktok_export.video_id(r.url)
            if vid:
                identifiers[vid] = "tiktok video id"
    except FileNotFoundError:
        print("  ! TikTok export NOT FOUND — that half of the scan did not run")
        complete = False

    # Very short identifiers would match ordinary prose, so they are excluded —
    # but reported, because a real 4-character handle silently dropped from a
    # leak scan is exactly the failure this tool exists to prevent.
    short = sorted(k for k in identifiers if len(k) < 7)
    identifiers = {k: v for k, v in identifiers.items() if len(k) >= 7}
    if short:
        print(f"  ! {len(short)} identifier(s) under 7 chars NOT scanned "
              f"(would match ordinary prose): {short}")

    print(f"Auditing tracked files against {len(identifiers)} real identifiers")
    print("=" * 70)

    files = tracked_files()
    hits: list[tuple[str, str, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for ident, kind in identifiers.items():
            if ident in text:
                hits.append((str(path.relative_to(REPO)), ident, kind))

    print(f"WORKING TREE — {len(files)} tracked files")
    if hits:
        for f, ident, kind in hits:
            print(f"  LEAK  {f}\n        {kind}: {ident}")
    else:
        print("  clean")

    # ── history ──────────────────────────────────────────────────────────
    # Deleting a secret in a later commit does NOT remove it from the repo.
    # Anyone who can read the repo can read every commit, so the only honest
    # question is whether the identifier appears ANYWHERE in history.
    print()
    print("HISTORY — every commit, all branches")
    history: dict[str, list[str]] = {}
    for ident, kind in identifiers.items():
        try:
            proc = subprocess.run(
                ["git", "-C", str(REPO), "log", "--all", "--format=%h", "-S", ident],
                capture_output=True, text=True, timeout=600,
            )
            if proc.returncode != 0:
                # Empty stdout from a failed git is indistinguishable from
                # "not found" — treat it as an incomplete scan, not a pass.
                print(f"  ! git log failed for an identifier: {proc.stderr.strip()[:80]}")
                complete = False
            out = proc.stdout.splitlines()
        except subprocess.TimeoutExpired:
            print("  ! history scan TIMED OUT — remaining identifiers unchecked")
            complete = False
            break
        if out:
            history[f"{kind}: {ident}"] = out

    if history:
        for label, commits in history.items():
            print(f"  LEAK  {label}")
            print(f"        commits: {' '.join(commits)}")
    else:
        print("  clean")

    if not hits and not history:
        if not complete:
            print("\n  INCOMPLETE — no leaks found, but the scan did not finish.")
            print("  This is NOT a clean result. Fix the cause above and re-run.")
            return 2
        print("\n  CLEAN — nothing in the working tree, nothing in history")
        return 0

    print()
    print(f"  {len(hits)} working-tree leak(s), {len(history)} in history.")
    if history:
        print("  Removing a file in a later commit does not remove it from history.")
        print("  Rewriting history is a Tier-1 decision — do not do it automatically.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
