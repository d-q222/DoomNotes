"""Summarization tests — the prompt seam and the model boundary.

No Ollama daemon is needed: the client is injected, which is the point of it
being injectable.

    # WHAT IS ASSERTED HERE
    #
    # HANDS-ON #5.1/#5.2 own the *content* of the schema and the registry
    # prompt — how tags are ranked, whether descriptions are injected, what
    # happens when the list is capped. Nothing here asserts any of that.
    #
    # What is asserted is the seam around it, which those decisions do not
    # move: the schema reaches Ollama as `format`, a non-conforming response is
    # an error rather than a bad note, model output never overwrites a field
    # that came from the ref, and the caption recorded verbatim is the export's
    # own — not yt-dlp's description.
    #
    # That last one is the interesting invariant. `Note.caption` is rendered
    # verbatim into the note body under "## Caption", so it must be text the
    # creator actually wrote, not a field the pipeline synthesised.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from doomnotes.models import Media, VideoRef
from doomnotes.summarize import (
    SYSTEM_PROMPT,
    SummarizeError,
    SummarizeSettings,
    build_user_prompt,
    load_schema,
    summarize,
)
from doomnotes.tags import TagRegistry

IG = VideoRef(
    "https://www.instagram.com/reel/AAAAAAAAAAA/",
    "instagram",
    caption="Three ways to index soft-deleted rows.",
    author="@someone",
    source_order=7,
)
TT = VideoRef(
    "https://www.tiktokv.com/share/video/1111111111111111111/",
    "tiktok",
    saved_at=datetime(2026, 8, 5, 8, 36, 26),
    source_order=3,
)

GOOD_RESPONSE = {
    "title": "Postgres partial indexes for soft-deleted rows",
    "summary": "It explains why a partial index beats a full one here.",
    "key_points": ["Index only rows where deleted_at is null"],
    "links": ["pgAnalyze"],
    "topic": "Coding / Databases",
    "tags": ["Coding", "databases", "postgres"],
}


class FakeClient:
    """Records the call and returns a canned response."""

    def __init__(self, payload: object = None, raw: str | None = None) -> None:
        self.payload = GOOD_RESPONSE if payload is None else payload
        self.raw = raw
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        content = self.raw if self.raw is not None else json.dumps(self.payload)
        return {"message": {"content": content}}


def factory(client: FakeClient):
    return lambda host: client


# ── the prompt ───────────────────────────────────────────────────────────


def test_caption_and_transcript_both_reach_the_prompt() -> None:
    prompt = build_user_prompt(IG, "spoken words here", IG.caption, TagRegistry(), SummarizeSettings())
    assert "Three ways to index soft-deleted rows." in prompt
    assert "spoken words here" in prompt


def test_transcript_is_labelled_as_asr() -> None:
    """The model must know the transcript is unreliable — ASR mishears."""
    prompt = build_user_prompt(IG, "wear clause", IG.caption, TagRegistry(), SummarizeSettings())
    assert "speech recognition" in prompt.lower()
    assert "expect errors" in prompt.lower()


def test_missing_transcript_is_stated_not_left_implicit() -> None:
    """Silence about a missing transcript invites the model to imagine one."""
    prompt = build_user_prompt(IG, None, IG.caption, TagRegistry(), SummarizeSettings())
    assert "no transcript" in prompt.lower()
    assert "do not speculate" in prompt.lower()


def test_nothing_to_summarise_is_an_error_not_an_empty_note() -> None:
    with pytest.raises(SummarizeError):
        build_user_prompt(TT, None, None, TagRegistry(), SummarizeSettings())


def test_caption_and_transcript_are_truncated_to_their_budgets() -> None:
    """num_ctx is finite. Truncation must happen here, not inside the model."""
    settings = SummarizeSettings(max_caption_chars=50, max_transcript_chars=80)
    prompt = build_user_prompt(IG, "t" * 5000, "c" * 5000, TagRegistry(), settings)
    assert "c" * 50 in prompt and "c" * 51 not in prompt
    assert "t" * 80 in prompt and "t" * 81 not in prompt


def test_registry_is_injected_with_the_reuse_instruction() -> None:
    """Tier-1 #3's mechanism: pass 1 reuses before it mints."""
    registry = TagRegistry()
    registry.observe(["coding", "gardening"])
    prompt = build_user_prompt(IG, None, IG.caption, registry, SummarizeSettings())
    assert "coding" in prompt and "gardening" in prompt
    assert "Create a new tag only when nothing in the list applies." in prompt


def test_empty_registry_says_so_rather_than_showing_nothing() -> None:
    """An empty list reads as 'no tags allowed' unless it is explained."""
    prompt = build_user_prompt(IG, None, IG.caption, TagRegistry(), SummarizeSettings())
    assert "registry is empty" in prompt


def test_system_prompt_carries_the_meaning_the_schema_cannot() -> None:
    """Ollama's `format` enforces types and shapes only — descriptions in the
    schema constrain nothing. So the anti-slug rule has to live in the prompt,
    and this asserts it has not drifted back into the schema alone.
    """
    assert "slug" in SYSTEM_PROMPT
    assert "Never add facts" in SYSTEM_PROMPT


# ── the model boundary ───────────────────────────────────────────────────


def test_schema_is_passed_to_ollama_as_format() -> None:
    """The load-bearing assumption of the whole stage."""
    client = FakeClient()
    summarize(IG, None, None, TagRegistry(), client_factory=factory(client))
    sent = client.calls[0]["format"]
    assert sent["required"] == ["title", "summary", "key_points", "links", "topic", "tags"]
    assert sent["additionalProperties"] is False


def test_schema_on_disk_stays_in_sync_with_the_note_body() -> None:
    """Fields the renderer emits from the ref must NOT be model-generated.

    A model asked for a URL invents one. `source_url`, `platform`, `author`,
    `posted_at`, `saved_at` and `source_order` all come from the ref or from
    yt-dlp, and the schema forbids the model from returning them at all.
    """
    schema = load_schema()
    forbidden = {
        "source_url", "platform", "author", "posted_at",
        "saved_at", "source_order", "processed_at", "caption",
    }
    assert forbidden.isdisjoint(schema["properties"]), (
        "the model must never be asked for a field that is known for certain"
    )


def test_non_json_response_is_an_error_not_a_note() -> None:
    client = FakeClient(raw="I'm sorry, I can't do that.")
    with pytest.raises(SummarizeError):
        summarize(IG, None, None, TagRegistry(), client_factory=factory(client))


def test_think_is_disabled_explicitly_when_configured_off() -> None:
    """Measured at ~20x on this machine. Left implicit it silently reverts."""
    client = FakeClient()
    summarize(
        IG, None, None, TagRegistry(),
        SummarizeSettings(think=False), client_factory=factory(client),
    )
    assert client.calls[0]["think"] is False


def test_think_true_sends_no_flag_so_any_model_still_works() -> None:
    """Passing think=True to a non-reasoning model errors. Omitting it is safe."""
    client = FakeClient()
    summarize(
        IG, None, None, TagRegistry(),
        SummarizeSettings(think=True), client_factory=factory(client),
    )
    assert "think" not in client.calls[0]


# ── what the model is not allowed to decide ──────────────────────────────


def test_ref_fields_are_never_taken_from_the_model() -> None:
    client = FakeClient({**GOOD_RESPONSE, "title": "A real title for this video"})
    note = summarize(IG, "a transcript", None, TagRegistry(), client_factory=factory(client))
    assert note.source_url == IG.url
    assert note.platform == "instagram"
    assert note.author == "@someone"
    assert note.source_order == 7


def test_caption_recorded_verbatim_is_the_export_caption_only() -> None:
    """"## Caption" claims to be what the creator wrote. It must be exactly that.

    yt-dlp's description is used to *inform* the summary for sources with no
    export caption, but it is not the export caption and must not be presented
    as one.
    """
    media = Media(ref=TT, audio_path=Path("/dev/null"), description="yt-dlp description text")
    client = FakeClient()
    note = summarize(TT, "a transcript", media, TagRegistry(), client_factory=factory(client))
    assert note.caption is None

    prompt = client.calls[0]["messages"][1]["content"]
    assert "yt-dlp description text" in prompt, (
        "the description should still inform the summary, just not be quoted as a caption"
    )


def test_export_caption_wins_over_yt_dlp_description() -> None:
    """The export caption is what was on screen when the post was saved."""
    media = Media(ref=IG, audio_path=Path("/dev/null"), description="a later edit")
    client = FakeClient()
    summarize(IG, None, media, TagRegistry(), client_factory=factory(client))
    prompt = client.calls[0]["messages"][1]["content"]
    assert "Three ways to index soft-deleted rows." in prompt
    assert "a later edit" not in prompt


def test_posted_at_comes_from_media_and_is_absent_without_it() -> None:
    """The documented interaction of two rulings: caption-only notes have no date."""
    client = FakeClient()
    without = summarize(IG, None, None, TagRegistry(), client_factory=factory(client))
    assert without.posted_at is None

    media = Media(ref=IG, audio_path=Path("/dev/null"), posted_at=datetime(2026, 7, 14))
    with_media = summarize(IG, None, media, TagRegistry(), client_factory=factory(client))
    assert with_media.posted_at == datetime(2026, 7, 14)


def test_has_transcript_reflects_reality_not_the_model() -> None:
    client = FakeClient({**GOOD_RESPONSE, "has_speech": True})
    note = summarize(IG, None, None, TagRegistry(), client_factory=factory(client))
    assert note.has_transcript is False


# ── tag hygiene at the boundary ──────────────────────────────────────────


def test_model_tags_are_normalised() -> None:
    """A model emits 'Coding' and 'ai tools / prompts'; the vault needs slugs."""
    client = FakeClient({**GOOD_RESPONSE, "tags": ["Coding", "AI Tools / Prompts", "  spaced  "]})
    note = summarize(IG, None, None, TagRegistry(), client_factory=factory(client))
    assert note.tags == ["coding", "ai-tools/prompts", "spaced"]


def test_tags_are_capped_at_the_configured_maximum() -> None:
    client = FakeClient({**GOOD_RESPONSE, "tags": [f"tag{i}" for i in range(20)]})
    note = summarize(
        IG, None, None, TagRegistry(),
        SummarizeSettings(max_tags=3), client_factory=factory(client),
    )
    assert len(note.tags) == 3


def test_a_note_always_has_at_least_one_tag() -> None:
    """An untagged note is invisible in the tag pane, which is the whole index."""
    client = FakeClient({**GOOD_RESPONSE, "tags": [], "topic": "coding"})
    note = summarize(IG, None, None, TagRegistry(), client_factory=factory(client))
    assert note.tags


def test_topic_falls_back_when_the_model_omits_it() -> None:
    client = FakeClient({**GOOD_RESPONSE, "topic": "", "tags": ["gardening"]})
    note = summarize(IG, None, None, TagRegistry(), client_factory=factory(client))
    assert note.topic == "gardening"


def test_summarize_needs_no_daemon() -> None:
    """If this ever tries to reach localhost:11434 the suite becomes flaky."""
    client = FakeClient()
    note = summarize(IG, None, None, TagRegistry(), client_factory=factory(client))
    assert note.title == GOOD_RESPONSE["title"]
