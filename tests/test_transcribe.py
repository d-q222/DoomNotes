"""Transcription seam tests.

The model is never loaded here — `_load_model` is patched. That is deliberate:
these assert the *contract* around `transcribe()`, which is what lets the model
size be swapped (HANDS-ON #4.2) without touching anything downstream.

The contract in one line: `None` means "no usable speech", never "an error
occurred". A caller that cannot tell those apart will log a silent reel as a
failure and lose the note.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doomnotes import transcribe as transcribe_mod
from doomnotes.transcribe import TranscribeSettings, transcribe


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModel:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def transcribe(self, path: str, beam_size: int = 1):
        self.calls.append({"path": path, "beam_size": beam_size})
        return [FakeSegment(t) for t in self.text.split("|")], object()


@pytest.fixture()
def audio(tmp_path: Path) -> Path:
    p = tmp_path / "clip.m4a"
    p.write_bytes(b"\x00")
    return p


@pytest.fixture()
def loaded(monkeypatch):
    """Install a fake model and hand back the last one constructed."""
    holder: dict[str, FakeModel] = {}

    def install(text: str) -> FakeModel:
        model = FakeModel(text)
        holder["model"] = model
        monkeypatch.setattr(
            transcribe_mod, "_load_model", lambda *a, **k: model, raising=True
        )
        return model

    return install


# ── the contract ─────────────────────────────────────────────────────────


def test_returns_text_for_real_speech(audio: Path, loaded) -> None:
    loaded("This is a sentence long enough to clear the minimum threshold.")
    assert transcribe(audio) is not None


def test_segments_are_joined_and_trimmed(audio: Path, loaded) -> None:
    loaded("  first part of it  | second part that makes it long enough  ")
    out = transcribe(audio)
    assert out == "first part of it second part that makes it long enough"


def test_a_missing_file_is_absence_not_an_exception(audio: Path, loaded) -> None:
    """A caller must not have to wrap this in try/except to survive."""
    loaded("irrelevant")
    assert transcribe(audio.parent / "does-not-exist.m4a") is None


def test_short_output_is_treated_as_no_speech(audio: Path, loaded) -> None:
    """Music-only reels and photo posts produce a few stray characters.

    Feeding "Thanks!" to the summariser as if it were content is how a note
    about nothing gets written.
    """
    loaded("Thanks!")
    assert transcribe(audio) is None


def test_the_no_speech_threshold_is_configurable(audio: Path, loaded) -> None:
    loaded("short one")
    assert transcribe(audio, TranscribeSettings(min_chars=5)) is not None
    assert transcribe(audio, TranscribeSettings(min_chars=500)) is None


def test_settings_reach_the_model(audio: Path, loaded) -> None:
    model = loaded("a transcript long enough to clear the default threshold")
    transcribe(audio, TranscribeSettings(beam_size=5))
    assert model.calls[-1]["beam_size"] == 5


def test_faster_whisper_is_not_imported_at_module_import_time() -> None:
    """Weights load is seconds. Importing the CLI must not pay for it.

    `_load_model` does the import inside the function precisely so that
    `doomnotes status` — which never transcribes — starts instantly.
    """
    import ast
    import inspect

    src = inspect.getsource(transcribe_mod)
    tree = ast.parse(src)
    toplevel = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    names = {
        alias.name for n in toplevel if isinstance(n, ast.Import) for alias in n.names
    } | {n.module or "" for n in toplevel if isinstance(n, ast.ImportFrom)}
    assert not any("whisper" in n for n in names), names


def test_model_loading_is_cached() -> None:
    """A batch is 30-40 clips; reloading weights each time would dominate."""
    assert hasattr(transcribe_mod._load_model, "cache_info")
