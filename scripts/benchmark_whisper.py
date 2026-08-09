#!/usr/bin/env python
"""Benchmark Whisper model sizes and pick one.

Judge on the DIFF, not the wall clock. A model that is three times faster and
drops the product names the video was saved for is not faster, it is useless
more quickly. The script prints both, side by side.

Clips: uses real audio from `data/audio/` when present (populated by a real
run after the cookie gate). With none available it falls back to synthetic
`say` clips so the harness is exercisable now — but synthetic speech is clean
studio audio, and real reels have music, accents and clipping. Relative timings
transfer; accuracy numbers do NOT. The script says so at the end.

Run: make benchmark-whisper
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from doomnotes.transcribe import TranscribeSettings, transcribe  # noqa: E402

MODELS = ["tiny.en", "base.en", "small.en", "distil-small.en"]

SYNTHETIC = [
    "Comment database and I will send over all the resources for SQL and APIs.",
    "These five plants survive low light. Snake plant, pothos, ZZ plant, and philodendron.",
    "The trick is a partial index with a where clause on the deleted at column.",
    "I swapped Midjourney for Nano Banana and Notion AI for Obsidian with a local model.",
    "Three protein sources under two dollars per serving: lentils, eggs, and canned sardines.",
]


def gather_clips(limit: int) -> tuple[list[Path], bool]:
    audio_dir = REPO / "data" / "audio"
    real = sorted(p for p in audio_dir.glob("*") if p.suffix in {".m4a", ".mp3", ".wav", ".opus"})
    if real:
        return real[:limit], True

    workdir = REPO / "data" / "benchmark"
    workdir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, phrase in enumerate(SYNTHETIC[:limit]):
        wav = workdir / f"synthetic_{i}.wav"
        if not wav.is_file():
            aiff = workdir / f"synthetic_{i}.aiff"
            subprocess.run(["say", "-o", str(aiff), phrase], check=True)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
                 "-ar", "16000", "-ac", "1", str(wav)],
                check=True,
            )
            aiff.unlink(missing_ok=True)
        clips.append(wav)
    return clips, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    clips, real = gather_clips(args.clips)
    print(f"clips: {len(clips)} ({'REAL audio from data/audio/' if real else 'SYNTHETIC via `say`'})")
    print("device: cpu (CTranslate2 has no Metal backend on Apple Silicon)")
    print("=" * 74)

    transcripts: dict[str, list[str]] = {}
    for model in args.models:
        settings = TranscribeSettings(model=model)
        total = 0.0
        outputs = []
        try:
            for clip in clips:
                t0 = time.time()
                text = transcribe(clip, settings)
                total += time.time() - t0
                outputs.append(text or "")
        except Exception as exc:  # noqa: BLE001
            print(f"{model:<18} FAILED: {type(exc).__name__}: {exc}")
            continue
        transcripts[model] = outputs
        print(f"{model:<18} {total:6.1f}s total   {total / len(clips):5.2f}s/clip")

    if len(transcripts) > 1:
        print("\nText diff — this is the thing to judge on")
        print("=" * 74)
        baseline = args.models[-1] if args.models[-1] in transcripts else list(transcripts)[-1]
        for i in range(len(clips)):
            print(f"\nclip {i}:")
            for model, outputs in transcripts.items():
                marker = "*" if model == baseline else " "
                print(f"  {marker}{model:<17} {outputs[i][:100]}")

    print("\n" + "=" * 74)
    if not real:
        print("NOTE: these were synthetic clips. Relative SPEED transfers; accuracy")
        print("does not — real reels have music, accents and clipping. Re-run this")
        print("after the first real batch, when data/audio/ has genuine clips.")
    print("Pick the smallest model whose diff still keeps the names and numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
