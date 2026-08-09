"""Speech-to-text behind a one-function interface.

    transcribe(path) -> str | None

`None` means "no usable speech", not "error" — a silent clip, a music-only reel,
or a photo post with no audio track. The caller turns that into
`has_transcript: false`, which is a legitimate note, not a failure.

    # ── UNTUNED DEFAULT: model size ─────────────────────────────────────────
    # CURRENT: config sets `base.en`, int8, beam_size 1, CPU.
    #
    # CONTEXT FOR TUNING: faster-whisper runs on CTranslate2, which has NO
    #   Metal backend, so on Apple Silicon this is CPU-only. That is usually
    #   fine for 30-90 s clips, but it is why model size matters more here than
    #   it would on CUDA. `mlx-whisper` is a Metal-native drop-in behind this
    #   same interface.
    #
    # HOW TO MEASURE: `make benchmark-whisper` times each size over a set of
    #   clips and prints seconds-per-clip alongside a diff of the text. The
    #   diff is the thing to judge on: a model 3x faster that drops the product
    #   names the video was saved for is not faster.
    # ────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscribeSettings:
    model: str = "base.en"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 1
    min_chars: int = 40


@lru_cache(maxsize=4)
def _load_model(model: str, device: str, compute_type: str):
    """Cached because loading weights costs seconds and a batch is 30–40 clips."""
    from faster_whisper import WhisperModel  # imported lazily: heavy

    return WhisperModel(model, device=device, compute_type=compute_type)


def transcribe(
    audio_path: str | Path,
    settings: TranscribeSettings | None = None,
) -> str | None:
    """Transcribe one audio file. Returns None when there is no usable speech."""
    cfg = settings or TranscribeSettings()
    path = Path(audio_path)
    if not path.is_file():
        log.warning("transcribe: no such file %s", path)
        return None

    model = _load_model(cfg.model, cfg.device, cfg.compute_type)
    segments, _info = model.transcribe(str(path), beam_size=cfg.beam_size)
    text = " ".join(seg.text.strip() for seg in segments).strip()

    if len(text) < cfg.min_chars:
        # Short output is the normal signal for music-only or silent clips.
        # Treating it as absence keeps a meaningless 6-character "transcript"
        # from being fed to the summariser as if it were content.
        log.info("transcribe: %s produced %d chars, treating as no speech", path.name, len(text))
        return None
    return text


def settings_from_config(cfg) -> TranscribeSettings:
    return TranscribeSettings(
        model=cfg.get("transcribe", "model", default="base.en"),
        device=cfg.get("transcribe", "device", default="cpu"),
        compute_type=cfg.get("transcribe", "compute_type", default="int8"),
        beam_size=int(cfg.get("transcribe", "beam_size", default=1)),
        min_chars=int(cfg.get("transcribe", "min_transcript_chars", default=40)),
    )
