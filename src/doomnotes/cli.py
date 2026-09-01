"""DoomNotes command line.

    doomnotes parse        # run both export parsers, print reconciliations
    doomnotes status       # what the store thinks is done
    doomnotes run          # process a batch (needs auth — see check-auth)
    doomnotes consolidate  # tag pass 2
    doomnotes journal      # what recent runs actually did
    doomnotes check-auth   # single auth probe, one video per platform
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from doomnotes import config as config_mod
from doomnotes.consolidate import consolidate
from doomnotes.download import download, fetch_url
from doomnotes.journal import open_journal, read_runs
from doomnotes.pipeline import Deps, run as run_batch
from doomnotes.sources import ig_export, manual, tiktok_export
from doomnotes.store import Store
from doomnotes.summarize import settings_from_config as summarize_settings, summarize
from doomnotes.tags import TagRegistry
from doomnotes.transcribe import settings_from_config as transcribe_settings, transcribe
from doomnotes.vault import VaultGuardError, VaultWriter, check_vault_root

log = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _auth_for(cfg):
    def inner(platform: str) -> dict:
        return dict(cfg.get("auth", platform, default={}) or {})
    return inner


DEFAULT_RUN_LOG = "data/logs/runs.jsonl"


def _run_log(cfg, *, for_writing: bool) -> Path | None:
    """The journal path, or None when journalling is switched off.

    An empty string means off, deliberately — a missing key falls back to the
    default rather than silently disabling the record, because a config written
    before this existed should still get one.

    `for_writing` is the whole reason this takes a flag. Refusing a path inside
    the vault protects the vault from gaining a file that is not a note; that is
    a concern about *writing*. Reading a journal that is already on disk puts
    nothing anywhere, so applying the same hard stop there would only mean
    `doomnotes journal` cannot show you a file you can see in Finder — a refusal
    that protects nothing.
    """
    if not cfg.get("paths", "run_log", default=DEFAULT_RUN_LOG):
        return None
    path = cfg.path("paths", "run_log", default=DEFAULT_RUN_LOG)
    if not for_writing:
        return path
    try:
        vault_root = check_vault_root(cfg.vault_root)
    except VaultGuardError:
        return path  # the vault is unusable; cmd_run reports that on its own
    if path.resolve(strict=False).is_relative_to(vault_root):
        raise VaultGuardError(
            f"paths.run_log ({path}) is inside the vault at {vault_root}. "
            f"The run journal holds saved-video URLs and is not a note — keep "
            f"it under data/."
        )
    return path


def cmd_parse(args, cfg) -> int:
    ig_refs, ig_rec = ig_export.parse(args.instagram) if args.instagram else ig_export.parse()
    print(ig_rec.render())
    print()
    tt_refs, tt_rec = (
        tiktok_export.parse(args.tiktok) if args.tiktok else tiktok_export.parse()
    )
    print(tt_rec.render())
    print()
    print(f"TOTAL v1 scope: {len(ig_refs) + len(tt_refs)} videos "
          f"({len(ig_refs)} instagram + {len(tt_refs)} tiktok)")

    if args.register:
        with Store(cfg.path("paths", "state_db")) as store:
            n = store.register(ig_refs + tt_refs)
            print(f"registered {n} new URLs in the store")
    return 0


def cmd_status(args, cfg) -> int:
    with Store(cfg.path("paths", "state_db")) as store:
        stats = store.stats()
        if not stats:
            print("store is empty — run `doomnotes parse --register` first")
            return 0
        print("Store")
        print("-" * 40)
        for platform, states in sorted(stats.items()):
            total = sum(states.values())
            detail = ", ".join(f"{k}={v}" for k, v in sorted(states.items()))
            print(f"  {platform:<10} {total:>4}   {detail}")
        failures = store.failures()
        if failures:
            print(f"\n  {len(failures)} failures (most recent first):")
            for row in failures[:10]:
                print(f"    {row['url']}\n      {row['error']}")
    return 0


def cmd_run(args, cfg) -> int:
    try:
        writer = VaultWriter(cfg.vault_root)
    except VaultGuardError as exc:
        print(f"vault guard refused the configured vault: {exc}", file=sys.stderr)
        return 2

    if args.urls:
        refs, dropped = manual.parse(args.urls)
        for line, why in dropped:
            print(f"  dropped {line}: {why}")
    else:
        ig_refs, _ = ig_export.parse()
        tt_refs, _ = tiktok_export.parse()
        order = cfg.get("pacing", "queue_order", default=["instagram", "tiktok"])
        by_platform = {"instagram": ig_refs, "tiktok": tt_refs}
        refs = [r for p in order for r in by_platform.get(p, [])]

    registry_rel = cfg.get("tags", "registry_file", default="_meta/tags.json")
    registry = TagRegistry.load(writer.root / registry_rel)

    t_settings = transcribe_settings(cfg)
    s_settings = summarize_settings(cfg)
    # The downloader is wired explicitly rather than left as Deps' default,
    # because the default is `download` called with three positional arguments —
    # which means its own function defaults win and the [download] section of
    # config.toml is inert. Editing audio_format there used to change nothing.
    timeout_s = int(cfg.get("download", "timeout_s", default=180))
    audio_format = str(cfg.get("download", "audio_format", default="m4a"))
    deps = Deps(
        downloader=lambda ref, audio_dir, auth: download(
            ref, audio_dir, auth, timeout_s=timeout_s, audio_format=audio_format
        ),
        transcriber=lambda p: transcribe(p, t_settings),
        summarizer=lambda ref, tr, media, reg: summarize(
            ref, tr, media, reg, s_settings
        ),
    )

    sleep_range = (
        float(cfg.get("pacing", "sleep_min_s", default=20)),
        float(cfg.get("pacing", "sleep_max_s", default=90)),
    )
    limit = args.limit if args.limit is not None else int(
        cfg.get("pacing", "batch_cap", default=35)
    )

    try:
        run_log = _run_log(cfg, for_writing=True)
    except VaultGuardError as exc:
        print(f"vault guard refused the run journal path: {exc}", file=sys.stderr)
        return 2

    try:
        with open_journal(run_log) as journal, \
                Store(cfg.path("paths", "state_db")) as store:
            result = run_batch(
                refs,
                store=store,
                writer=writer,
                registry=registry,
                deps=deps,
                audio_dir=cfg.path("paths", "audio_dir"),
                auth_for=_auth_for(cfg),
                transcripts_dir=cfg.get("vault", "transcripts_dir", default="_transcripts"),
                limit=limit,
                sleep_range=None if args.no_sleep else sleep_range,
                keep_audio=args.keep_audio,
                journal=journal,
            )
    finally:
        # Persisted even when the run does not finish. A paced batch spends
        # roughly half an hour asleep between downloads, so Ctrl-C partway
        # through is an ordinary way for it to end — and the notes it already
        # wrote are on disk either way. Without this, the vocabulary those
        # notes contributed is dropped, and the next run's pass-1 prompt is
        # handed a registry that disagrees with the vault, so it mints
        # duplicates of tags that already exist.
        #
        # Not data loss: `doomnotes consolidate` rebuilds the registry from
        # note frontmatter. It is a quality loss until someone does.
        #
        # Guarded, because an exception raised inside a `finally` REPLACES the
        # one being propagated. A disk-full error here would otherwise hide the
        # VaultGuardError or checkpoint that actually ended the run — swapping
        # the diagnosis for a symptom at the exact moment it is needed.
        try:
            writer.write_text(registry_rel, registry.to_json())
        except Exception:  # noqa: BLE001 - must never displace the real error
            log.exception("could not persist the tag registry to %s", registry_rel)

    print()
    print(result.summary())
    if result.stopped:
        print("\n  A checkpoint/captcha was hit. Clear it manually in the app,")
        print("  then re-run. Do NOT retry automatically.")
        return 3
    return 0


def cmd_consolidate(args, cfg) -> int:
    writer = VaultWriter(cfg.vault_root)
    plan = consolidate(
        writer,
        cfg.get("tags", "registry_file", default="_meta/tags.json"),
        merge_threshold=float(cfg.get("tags", "merge_similarity", default=0.85)),
        split_threshold=int(cfg.get("tags", "split_threshold", default=15)),
        dry_run=args.dry_run,
    )
    print(plan.render())
    if args.dry_run:
        print("\n(dry run — nothing was written)")
    return 0


def cmd_journal(args, cfg) -> int:
    """Read the run journal back. Reports; decides nothing.

    The grouping below is the whole point of 7.2: the run summary says "29
    failed", and `state.db` keeps only the latest state per URL. Neither can
    tell you that videos 12 through 30 all failed at the same stage within
    400 ms of each other, which is what a dead Ollama looks like from outside.
    """
    path = _run_log(cfg, for_writing=False)
    if path is None:
        print("journalling is off — set paths.run_log in config.toml")
        return 0

    try:
        records = read_runs(path)
    except OSError as exc:
        # An unreadable journal is a real problem worth naming. Reporting it as
        # "no runs recorded yet" would be worse than the traceback it replaces.
        print(f"could not read the run journal at {path}: {exc}", file=sys.stderr)
        return 2

    if not records:
        print(f"no runs recorded yet at {path}")
        return 0

    # Records are whatever is on disk, which may include lines this tool did
    # not write. Anything that is not an object is skipped rather than crashed
    # on, so `read_runs`'s promise of tolerance holds for its own consumer too.
    records = [r for r in records if isinstance(r, dict)]

    run_ids: list[str] = []
    for rec in records:
        run_id = rec.get("run_id") or "(no run id)"
        if run_id not in run_ids:
            run_ids.append(run_id)

    # `--last 0` must mean nothing, not everything: `run_ids[-0:]` is the whole
    # list, because Python has no negative zero.
    count = max(0, int(args.last))
    for run_id in (run_ids[-count:] if count else []):
        rows = [r for r in records if (r.get("run_id") or "(no run id)") == run_id]
        videos = [r for r in rows if r.get("event") == "video"]
        finished = next((r for r in rows if r.get("event") == "run_finished"), None)

        print(f"\nrun {run_id}   {len(videos)} video(s)"
              + ("" if finished else "   [did not finish]"))
        print("-" * 60)

        by_stage: dict[str, list[dict]] = {}
        for row in videos:
            by_stage.setdefault(row.get("stage") or row.get("status") or "?", []).append(row)

        for stage, rows_for_stage in sorted(by_stage.items(), key=lambda kv: -len(kv[1])):
            # Positions come from the pipeline's own `n`, never from counting
            # these rows. Numbering survivors renumbers everything after a
            # dropped record, which would turn scattered failures into a
            # spurious "unbroken run" — the exact claim you would act on.
            positions = sorted(r["n"] for r in rows_for_stage if isinstance(r.get("n"), int))
            seconds = [s for r in rows_for_stage if isinstance(s := r.get("seconds"), (int, float))]
            span = f"{min(seconds):>5.1f}-{max(seconds):.1f}s" if seconds else " " * 11

            where = ""
            if positions:
                # Consecutive is only assertable when every record is present.
                # If any went missing, say where they were and stop short of a
                # causal claim the data cannot support.
                complete = len(positions) == len(rows_for_stage)
                consecutive = positions == list(range(positions[0], positions[-1] + 1))
                where = (
                    f"video {positions[0]}"
                    if len(positions) == 1
                    else f"videos {positions[0]}-{positions[-1]}"
                )
                if len(positions) > 1 and not consecutive:
                    where += ", scattered"
                elif consecutive and len(positions) > 2 and complete:
                    where += "  <- unbroken run"
            print(f"  {stage:<14} {len(rows_for_stage):>4}   {span}   {where}")

        slept = [s for r in rows if r.get("event") == "slept"
                 and isinstance(s := r.get("seconds"), (int, float))]
        if slept:
            print(f"  {'(slept)':<14} {len(slept):>4}   "
                  f"{min(slept):.0f}-{max(slept):.0f}s, total {sum(slept)/60:.1f} min")

        if args.errors:
            # Prints saved-video URLs to the terminal, deliberately — you need
            # to know which video failed. Behind a flag so it is never in the
            # default output, and so it is never in a screenshot by accident.
            for row in videos:
                if row.get("detail"):
                    print(f"    {row.get('url')}\n      {row['detail']}")
    return 0


def cmd_check_auth(args, cfg) -> int:
    """Print the manual auth probe. One video per platform, nothing more."""
    print("Cookie auth probe")
    print("=" * 60)
    print("This step is deliberately manual: it needs the OS keychain and an")
    print("already-authenticated 2FA session, so it is run by hand.\n")
    print("Run these by hand, one at a time, and read the errors:\n")
    # Deliberately placeholders, not real URLs. A real shortcode here would be
    # a real saved post committed to a repository.
    for platform, example, note in (
        (
            "instagram",
            "https://www.instagram.com/reel/<SHORTCODE>/",
            "<SHORTCODE> — grep a reel URL out of the saved-posts export "
            f"({ig_export.EXPORT_GLOB}). `doomnotes parse` reports counts, "
            "not URLs",
        ),
        (
            "tiktok",
            "https://www.tiktok.com/@_/video/<ID>",
            "<ID> — the digits from a share/video/<ID>/ link in "
            f"{tiktok_export.DEFAULT_EXPORT.name}. Not a typo, see below",
        ),
    ):
        auth = cfg.get("auth", platform, default={}) or {}
        browser = auth.get("browser", "chrome")
        print(f"  # {platform}    {note}")
        print(f"  yt-dlp --cookies-from-browser {browser} -f 'ba/b' -x \\")
        print(f"    '{example}' -o 'data/authtest-{platform}.%(ext)s'\n")

    print("Why the TikTok URL above is not the one in your export. yt-dlp has")
    print("no extractor for the www.tiktokv.com host the export uses — it falls")
    print("through to the generic extractor and relies on a redirect. A real")
    print("run therefore requests `https://www.tiktok.com/@_/video/<ID>`, which")
    print("is yt-dlp's own form for an unknown uploader. Probing that form is")
    print("what makes this test what a batch will actually do.")
    print()
    print("If it fails and the share URL works, revert `fetch_url` in")
    print("download.py to return ref.url — nothing else depends on it.\n")
    print("Success = an audio file on disk. On failure, fall back to an")
    print("exported cookies.txt at")
    print("  ~/.config/doomnotes/cookies.txt   (chmod 600, outside this repo)")
    print("and set mode = \"file\" for that platform in config.toml.\n")
    print("That cookie is a bearer credential equivalent to a logged-in")
    print("session; 2FA does not protect it. Revoke via Instagram →")
    print("Settings → Security → Login activity.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="doomnotes", description=__doc__)
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("parse", help="parse both exports, print reconciliation")
    p.add_argument("--instagram", default=None)
    p.add_argument("--tiktok", default=None)
    p.add_argument("--register", action="store_true", help="record URLs in the store")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("status", help="what the store considers processed")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("run", help="process a batch")
    p.add_argument("--urls", default=None, help="manual URL list file")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-sleep", action="store_true", help="skip pacing (offline tests only)")
    p.add_argument("--keep-audio", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("consolidate", help="tag pass 2: merge + sub-cluster split")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_consolidate)

    p = sub.add_parser("journal", help="what recent runs actually did")
    p.add_argument("--last", type=int, default=3, help="how many runs to show")
    p.add_argument("--errors", action="store_true", help="list each failure's message")
    p.set_defaults(func=cmd_journal)

    p = sub.add_parser("check-auth", help="print the manual cookie-auth probe")
    p.set_defaults(func=cmd_check_auth)

    args = ap.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = config_mod.load(args.config)
    return args.func(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
