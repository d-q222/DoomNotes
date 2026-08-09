#!/usr/bin/env python
"""Prove faster-whisper actually loads and transcribes on THIS machine.

Not a benchmark — that is scripts/benchmark_whisper.py. This is
the "does the stack work at all" check, and it uses macOS `say` to generate a
clip so it needs no downloaded media and no network.

Worth knowing what it is measuring: faster-whisper runs on CTranslate2, which
has NO Metal backend. This is CPU-only on Apple Silicon — the 18 cores, not the
GPU. That is why 4.2 says measure rather than assume.

Run: make smoke-whisper
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from doomnotes.transcribe import TranscribeSettings, transcribe  # noqa: E402

PHRASE = (
    "So the trick here is a partial index. You add a where clause to the index "
    "definition, so deleted rows never enter the b-tree at all."
)


def main() -> int:
    workdir = REPO / "data" / "smoke"
    workdir.mkdir(parents=True, exist_ok=True)
    aiff = workdir / "smoke.aiff"
    wav = workdir / "smoke.wav"

    print("Generating a test clip with `say` (no network, no downloads)")
    subprocess.run(["say", "-o", str(aiff), PHRASE], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
         "-ar", "16000", "-ac", "1", str(wav)],
        check=True,
    )
    print(f"  clip: {wav.name}  ({wav.stat().st_size / 1024:.0f} KB)")

    settings = TranscribeSettings()
    print(f"  model: {settings.model}  device={settings.device} "
          f"compute={settings.compute_type}  (CPU — CTranslate2 has no Metal backend)")

    t0 = time.time()
    text = transcribe(wav, settings)
    dt = time.time() - t0

    print("-" * 66)
    if text is None:
        print("FAILED: transcribe() returned None for a clip that has speech")
        return 1

    print(f"  transcribed in {dt:.1f}s")
    print(f"  spoken : {PHRASE}")
    print(f"  heard  : {text}")

    # A loose check — ASR will differ in punctuation and casing. We only want
    # evidence the model is genuinely recognising this audio, not exact match.
    spoken_words = {w.strip(".,").lower() for w in PHRASE.split()}
    heard_words = {w.strip(".,").lower() for w in text.split()}
    overlap = len(spoken_words & heard_words) / len(spoken_words)
    print(f"  word overlap: {overlap:.0%}")

    print("-" * 66)
    if overlap < 0.6:
        print("FAILED: transcript does not resemble the input")
        return 1
    print("SMOKE TEST PASSED — faster-whisper loads and transcribes on this machine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
