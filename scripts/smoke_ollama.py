#!/usr/bin/env python
"""Task 0.4 — prove Ollama's `format` parameter actually constrains generation.

This is a smoke test, not an open question: `format` has taken a full JSON
Schema since Ollama v0.5 and compiles it to a llama.cpp grammar. But the whole
summarization stage rests on that, so it gets verified against a real caption
from the export rather than assumed.

Run: make smoke-ollama
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomnotes import config as config_mod  # noqa: E402
from doomnotes.sources.ig_export import parse as parse_ig  # noqa: E402
from doomnotes.summarize import (  # noqa: E402
    SummarizeSettings,
    load_schema,
    settings_from_config,
    summarize,
)
from doomnotes.tags import TagRegistry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="which export entry to use")
    ap.add_argument("--count", type=int, default=1, help="how many to summarise")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cfg = config_mod.load()
    settings = settings_from_config(cfg)
    if args.model:
        settings = SummarizeSettings(**{**settings.__dict__, "model": args.model})

    schema = load_schema()
    required = schema["required"]
    print(f"schema : {len(schema['properties'])} properties, required={required}")
    print(f"model  : {settings.model}  host={settings.host}")
    print("-" * 68)

    refs, _rec = parse_ig()
    registry = TagRegistry()
    ok = 0

    for offset in range(args.count):
        ref = refs[args.index + offset]
        if not ref.has_export_caption:
            print(f"[{offset}] skipped: no caption")
            continue

        t0 = time.time()
        try:
            note = summarize(ref, transcript=None, registry=registry, settings=settings)
        except Exception as exc:  # noqa: BLE001
            print(f"[{offset}] FAILED: {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t0

        # The point of the test: every required field present, correct types,
        # and the array bounds the schema declared.
        problems = []
        if not (8 <= len(note.title) <= 90):
            problems.append(f"title length {len(note.title)} outside [8,90]")
        if not (1 <= len(note.key_points) <= 8):
            problems.append(f"key_points {len(note.key_points)} outside [1,8]")
        if not (1 <= len(note.tags) <= 5):
            problems.append(f"tags {len(note.tags)} outside [1,5]")
        if not isinstance(note.links, list):
            problems.append("links not a list")

        status = "CONFORMS" if not problems else "VIOLATION"
        print(f"[{offset}] {status}  {dt:5.1f}s  {ref.platform}")
        print(f"      title : {note.title}")
        print(f"      topic : {note.topic}   tags: {note.tags}")
        print(f"      summary: {note.summary[:150]}")
        print(f"      key_points: {len(note.key_points)}   links: {len(note.links)}")
        for p in problems:
            print(f"      !! {p}")
        registry.observe(note.tags)
        ok += not problems

    print("-" * 68)
    print(f"conforming: {ok}/{args.count}")
    print(f"registry after run: {json.dumps(registry.counts())}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
