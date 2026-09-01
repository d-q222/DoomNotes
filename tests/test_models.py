"""VideoRef's salvage contract (decision 1.1).

`has_export_caption` decides whether a failed download still produces a note.
It is the only thing standing between a failed Instagram fetch and a permanent
loss, so what it reads has to be true rather than inferred.

The rule these tests pin: salvageability is declared by the source that built
the ref, never derived from `platform`.
"""

from __future__ import annotations

import pytest

from doomnotes.models import VideoRef

IG_URL = "https://www.instagram.com/reel/AAAAAAAAAAA/"
TT_URL = "https://www.tiktokv.com/share/video/1111111111111111111/"


def test_platform_does_not_decide_salvageability() -> None:
    """The whole decision, in one assertion.

    Both refs are `platform="instagram"`. One came from the export, which ships
    the caption; the other from a manual list, whose caption arrives with the
    media exactly as TikTok's does. Any rule phrased over `platform` gets one
    of these two wrong.
    """
    from_export = VideoRef(IG_URL, "instagram", caption_source="export", caption="text")
    from_manual = VideoRef(IG_URL, "instagram", caption_source="media")

    assert from_export.has_export_caption
    assert not from_manual.has_export_caption


def test_a_media_sourced_ref_cannot_carry_a_caption() -> None:
    """The impossible combination is rejected loudly, not silently tolerated.

    A caption that arrives with the download cannot exist before it. Allowing
    the combination would let the salvage branch fire for a ref that has no
    export caption to salvage from.
    """
    with pytest.raises(ValueError, match="caption_source='media'"):
        VideoRef(TT_URL, "tiktok", caption_source="media", caption="impossible")


def test_the_rejection_covers_every_construction_path() -> None:
    """The guard is on the model, not in an adapter.

    A new source in `sources/` must not be able to route around it, and neither
    should a test fixture — which is how an invariant quietly stops holding.
    """
    with pytest.raises(ValueError):
        VideoRef(IG_URL, "instagram", caption_source="media", caption="text")


def test_an_unrecognised_caption_source_is_rejected() -> None:
    """`Literal` is a type-checker annotation, not a runtime constraint.

    A typo reads as "not export", so an Instagram ref that shipped its caption
    would silently stop being salvageable: the download fails, the salvage
    branch declines, and the store records it as done. Nothing reports it.
    """
    with pytest.raises(ValueError, match="caption_source must be one of"):
        VideoRef(IG_URL, "instagram", caption_source="exprot", caption="text")


def test_media_is_the_default_so_forgetting_fails_safe() -> None:
    """An undeclared ref is treated as unsalvageable rather than assumed good.

    Combined with the rejection above, the default can only ever be wrong in
    the direction that raises: a source carrying captions cannot silently keep
    the default, because the caption itself trips the guard.
    """
    assert VideoRef(TT_URL, "tiktok").caption_source == "media"
    assert not VideoRef(TT_URL, "tiktok").has_export_caption


def test_an_export_ref_with_an_empty_caption_is_not_salvageable() -> None:
    """The 2-in-251 case, and why the field earns its place.

    An export ref whose caption is genuinely empty and a ref whose caption was
    never going to be present are indistinguishable from `caption` alone. Both
    correctly fail salvage — a caption-only note built from nothing is not a
    note — but only the first is a parser-miss candidate worth reporting.
    """
    empty = VideoRef(IG_URL, "instagram", caption_source="export", caption="")
    never = VideoRef(IG_URL, "instagram", caption_source="media")

    assert not empty.has_export_caption
    assert not never.has_export_caption
    assert empty.caption_source != never.caption_source, "the distinction survives"


@pytest.mark.parametrize("caption", ["", "   ", "\n\t ", None])
def test_whitespace_is_not_a_caption(caption: str | None) -> None:
    """Salvage needs something to summarise, not merely a non-None field."""
    assert not VideoRef(
        IG_URL, "instagram", caption_source="export", caption=caption
    ).has_export_caption


def test_the_manual_adapter_emits_instagram_refs_on_the_media_side(tmp_path) -> None:
    """The adapter half of the same rule, exercised end to end.

    This is the case that makes a platform-based rule wrong in practice: a
    manual list can contain an Instagram URL, and its caption still arrives
    with the media. If this ever emits "export", the salvage branch starts
    lying about refs it cannot recover.
    """
    from doomnotes.sources import manual

    listing = tmp_path / "urls.txt"
    listing.write_text(f"{IG_URL}\n{TT_URL}\n", encoding="utf-8")

    refs, dropped = manual.parse(listing)

    assert not dropped
    assert {r.platform for r in refs} == {"instagram", "tiktok"}
    assert {r.caption_source for r in refs} == {"media"}
    assert not any(r.has_export_caption for r in refs)
