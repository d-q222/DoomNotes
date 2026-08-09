#!/usr/bin/env python
"""Trace ONE video through every stage, printing the real data at each boundary.

Prints the actual shape of the data at every boundary rather than describing it.

    python scripts/trace_one.py            # real identifiers, local only
    python scripts/trace_one.py --redact   # handle/shortcode masked, safe to paste

The download stage is not exercised — that needs the manual cookie gate.
Everything else is genuinely executed: the parse is real, the Ollama call is
real, the render is real, the guarded vault write is real.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import textwrap
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from doomnotes.render import note_slug, render_note, render_transcript  # noqa: E402
from doomnotes.sources.ig_export import DEFAULT_EXPORT, parse  # noqa: E402
from doomnotes.store import Store  # noqa: E402
from doomnotes.summarize import settings_from_config, summarize  # noqa: E402
from doomnotes import config as config_mod  # noqa: E402
from doomnotes.tags import TagRegistry  # noqa: E402
from doomnotes.vault import VaultWriter  # noqa: E402

# A real transcript captured from `make smoke-whisper`, used as the stand-in for
# the download+transcribe stages, which are gated on the manual cookie test. It is
# genuine faster-whisper output, including its error: it heard "wear clause".
SAMPLE_TRANSCRIPT = (
    "So the trick here is a partial index, you add a wear clause to the index "
    "definition, so deleted rows never enter the B-tree at all."
)

REDACTIONS = [
    (re.compile(r"instagram\.com/(reel|reels|p|tv)/[A-Za-z0-9_-]+"), r"instagram.com/\1/XXXXXXXXXXX"),
    (re.compile(r"@[A-Za-z0-9_.]{2,}"), "@creator"),
    (re.compile(r"tiktokv?\.com/(share/)?video/\d+"), r"tiktokv.com/\1video/0000000000000000000"),
]


def redact(text: str, on: bool) -> str:
    if not on:
        return text
    for pattern, repl in REDACTIONS:
        text = pattern.sub(repl, text)
    return text


def banner(n: int, title: str) -> None:
    print()
    print("=" * 74)
    print(f"  STAGE {n} — {title}")
    print("=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--redact", action="store_true")
    args = ap.parse_args()
    R = args.redact

    # ── 0. the raw export ────────────────────────────────────────────────
    banner(0, "RAW EXPORT (bytes on disk)")
    html = DEFAULT_EXPORT.read_text(encoding="utf-8", errors="replace")
    i = html.find('<table style="table-layout: fixed;">')
    fragment = html[i : i + 620]
    print(redact(textwrap.fill(fragment, 100, replace_whitespace=False), R))
    print("\n  ^ note: label text ('URL', 'Caption') is what the parser keys on.")
    print("    The class names (_a6_q, _a6_r) are Meta's obfuscated names and")
    print("    change between exports.")

    # ── 1. parse ─────────────────────────────────────────────────────────
    banner(1, "SOURCE → VideoRef")
    refs, rec = parse()
    ref = refs[args.index]
    print(f"  parsed {len(refs)} refs from the export")
    print(f"  ref[{args.index}]:")
    print(f"    url          = {redact(ref.url, R)}")
    print(f"    platform     = {ref.platform}")
    print(f"    author       = {redact(str(ref.author), R)}")
    print(f"    saved_at     = {ref.saved_at}        <- None: IG export has no dates")
    print(f"    source_order = {ref.source_order}    <- the only recency signal")
    print(f"    caption      = {redact(repr((ref.caption or '')[:90]), R)}...")
    print(f"    has_export_caption = {ref.has_export_caption}  <- makes salvage possible")

    # ── 2. store ─────────────────────────────────────────────────────────
    banner(2, "STORE (queue admission)")
    workdir = REPO / "data" / "trace"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    with Store(workdir / "state.db") as store:
        store.register([ref])
        print(f"    state_of(url)        = {store.state_of(ref.url)}")
        queue = store.queue([ref])
        print(f"    queue length         = {len(queue)}")
        store.mark_done(ref.url)
        print(f"    after mark_done      = {store.state_of(ref.url)}")
        print(f"    filter_unprocessed   = {store.filter_unprocessed([ref])}  <- re-runs skip it")
        print("\n  The store is keyed on URL and never looks at the vault. Deleting")
        print("  the note does NOT bring it back — that is the isolation test.")

    # ── 3/4. download + transcribe ───────────────────────────────────────
    banner(3, "DOWNLOAD → TRANSCRIBE  (not executed — gated on manual auth)")
    print("  yt-dlp would run here, with cookies passed as a PATH, never a value.")
    print("  Its metadata supplies posted_at, which is why caption-only notes")
    print("  have no posted_at — the field added to judge recency is missing on")
    print("  exactly the notes whose content also cannot be judged.")
    print("\n  Standing in with a real faster-whisper transcript from make smoke-whisper:")
    print(f"    {SAMPLE_TRANSCRIPT}")
    print("\n  ^ it heard 'wear clause' for 'where clause'. That error is precisely")
    print("    why transcripts are kept and linked instead of trusted silently.")

    # ── 5. summarize ─────────────────────────────────────────────────────
    banner(5, "SUMMARIZE  (real Ollama call, schema-constrained)")
    cfg = config_mod.load()
    settings = settings_from_config(cfg)
    registry = TagRegistry()
    print(f"  model={settings.model}  registry={registry.prompt_block()}")
    print("  calling...")
    started = datetime.now()
    note = summarize(ref, SAMPLE_TRANSCRIPT, None, registry, settings)
    elapsed = (datetime.now() - started).total_seconds()
    print(f"  returned in {elapsed:.1f}s\n")
    print(f"    title      = {redact(note.title, R)}")
    print(f"    topic      = {note.topic}")
    print(f"    tags       = {note.tags}")
    print(f"    summary    = {redact(note.summary[:200], R)}")
    print(f"    key_points = {len(note.key_points)}")
    for kp in note.key_points:
        print(f"        - {redact(kp[:90], R)}")
    print(f"    links      = {[redact(l, R) for l in note.links][:4]}")

    # ── 6. render + guarded write ────────────────────────────────────────
    banner(6, "RENDER → GUARDED VAULT WRITE")
    vault = workdir / "vault"
    (vault / "_transcripts").mkdir(parents=True)
    writer = VaultWriter(vault)
    slug = note_slug(note, set())
    md = render_note(note, slug)
    tmd = render_transcript(note, SAMPLE_TRANSCRIPT)
    note_path, tpath = writer.write_pair(f"{slug}.md", md, f"_transcripts/{slug}.md", tmd)
    print(f"  slug         = {slug}")
    print(f"  note         = {note_path.relative_to(vault)}")
    print(f"  transcript   = {tpath.relative_to(vault) if tpath else None}")
    print("\n  --- rendered note ---")
    print(textwrap.indent(redact(md, R), "  "))

    print("=" * 74)
    print(f"  artifacts in {workdir.relative_to(REPO)}/  (gitignored)")
    if not R:
        print("  NOTE: real identifiers above. Use --redact before pasting anywhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
