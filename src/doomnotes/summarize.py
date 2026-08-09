"""Ollama summarization, constrained by schema/note.json.

Ollama's `format` parameter takes a full JSON Schema and compiles it to a
llama.cpp grammar, so a conforming object is what the sampler is *able* to
produce — not what the prompt politely asks for. That is the load-bearing
assumption of this whole stage, and `make smoke-ollama` proves it on real data
rather than trusting it.

The client is injectable so the pipeline can be tested without a running daemon.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from doomnotes.models import Media, Note, VideoRef
from doomnotes.tags import TagRegistry, normalise_tag

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "note.json"


def load_schema(path: str | Path = SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


SYSTEM_PROMPT = """You summarise short-form social videos into reference notes.

Rules:
- Summarise only what the source actually says. Never add facts, links, product
  names or numbers that are not present in the material you were given.
- If the material is thin, say so briefly. A short honest note beats an invented
  detailed one.
- `links` collects tools, products, books, sites or named resources that are
  actually mentioned. If none are mentioned, return an empty array.
- `title` is a normal English phrase in sentence case, with ordinary spaces and
  capitalisation, describing what the video is ABOUT. Specific enough to
  recognise months later. No emoji and no clickbait.
  Write it the way you would say it out loud:
      good: Postgres partial indexes for soft-deleted rows
      good: Fifteen paid AI tools and their free replacements
      bad:  postgres-partial-indexes-for-soft-deleted-rows   (that is a slug)
      bad:  creator_instagram_caption_summary                (that describes the post)
  Describe the CONTENT, never the post or the caption itself. Do not mention
  Instagram, TikTok, "caption", "creator" or the handle in the title.
- Creators overstate. Record the claim, not your endorsement of it.
"""


@dataclass
class SummarizeSettings:
    model: str = "qwen3.5:9b"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.2
    num_ctx: int = 8192
    max_caption_chars: int = 4000
    max_transcript_chars: int = 12000
    max_tags: int = 5
    # qwen3.5 is a reasoning model. Left on, it spends tens of thousands of
    # thinking characters deciding how to summarise a caption. Measured on this
    # machine, two real captions:
    #     think=True   147.0s / 79.2s   (25,342 / 11,888 thinking chars)
    #     think=False    4.1s /  8.6s   (0)
    # Titles were equal or better with it off. Across 342 videos that is roughly
    # 11 hours versus 35 minutes.
    # NOT measured: quality on long transcripts, which is where reasoning would
    # most plausibly pay for itself. Flip this back to true and compare if
    # summaries look shallow once real transcripts are flowing.
    think: bool = False


class SummarizeError(RuntimeError):
    """Raised when the model could not produce a usable object."""


def build_user_prompt(
    ref: VideoRef,
    transcript: str | None,
    caption: str | None,
    registry: TagRegistry,
    settings: SummarizeSettings,
) -> str:
    parts: list[str] = [f"Platform: {ref.platform}"]
    if ref.author:
        parts.append(f"Creator: {ref.author}")

    if caption:
        parts.append("\n--- CAPTION (written by the creator) ---\n"
                     + caption[: settings.max_caption_chars])
    if transcript:
        parts.append("\n--- TRANSCRIPT (automatic speech recognition; expect errors) ---\n"
                     + transcript[: settings.max_transcript_chars])

    if not caption and not transcript:
        raise SummarizeError("nothing to summarise: no caption and no transcript")

    if not transcript:
        parts.append(
            "\nNOTE: there is no transcript — the video could not be fetched or had "
            "no speech. Summarise from the caption alone and do not speculate about "
            "what the video showed."
        )

    parts.append(
        "\n--- EXISTING TAG VOCABULARY ---\n"
        + registry.prompt_block()
        + "\n\nReuse a tag from the list above when one genuinely fits. "
          "Create a new tag only when nothing in the list applies."
    )
    return "\n".join(parts)


def _default_client(host: str):
    import ollama  # imported lazily so tests need no daemon

    return ollama.Client(host=host)


def summarize(
    ref: VideoRef,
    transcript: str | None,
    media: Media | None = None,
    registry: TagRegistry | None = None,
    settings: SummarizeSettings | None = None,
    *,
    client_factory: Callable[[str], Any] = _default_client,
    schema: dict[str, Any] | None = None,
) -> Note:
    """Produce a Note. Raises SummarizeError if the model returns nothing usable."""
    cfg = settings or SummarizeSettings()
    reg = registry or TagRegistry()
    note_schema = schema if schema is not None else load_schema()

    # The Instagram export caption is preferred over yt-dlp's description
    # because it is what was on screen when the post was saved.
    caption = ref.caption or (media.description if media else None)

    user_prompt = build_user_prompt(ref, transcript, caption, reg, cfg)
    client = client_factory(cfg.host)

    kwargs: dict[str, Any] = dict(
        model=cfg.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format=note_schema,
        options={"temperature": cfg.temperature, "num_ctx": cfg.num_ctx},
    )
    if not cfg.think:
        # Only sent when disabling. Passing think=True to a non-reasoning model
        # errors, so the default path stays compatible with any model.
        kwargs["think"] = False

    response = client.chat(**kwargs)

    content = _extract_content(response)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SummarizeError(f"model returned non-JSON: {content[:200]!r}") from exc

    tags = [normalise_tag(t) for t in data.get("tags", [])]
    tags = [t for t in tags if t][: cfg.max_tags]
    topic = normalise_tag(data.get("topic", "")) or (tags[0] if tags else "unsorted")

    return Note(
        title=data["title"].strip(),
        source_url=ref.url,
        platform=ref.platform,
        summary=data["summary"].strip(),
        key_points=[p.strip() for p in data.get("key_points", []) if p.strip()],
        links=[l.strip() for l in data.get("links", []) if l.strip()],
        topic=topic,
        tags=tags or [topic],
        author=ref.author,
        posted_at=media.posted_at if media else None,
        saved_at=ref.saved_at,
        caption=ref.caption,          # verbatim, Instagram only
        has_transcript=bool(transcript),
        source_order=ref.source_order,
        processed_at=datetime.now(),
    )


def _extract_content(response: Any) -> str:
    """Tolerate both the dict and object shapes the ollama client returns."""
    if isinstance(response, dict):
        return response.get("message", {}).get("content", "")
    message = getattr(response, "message", None)
    return getattr(message, "content", "") if message is not None else ""


def settings_from_config(cfg) -> SummarizeSettings:
    return SummarizeSettings(
        model=cfg.get("summarize", "model", default="qwen3.5:9b"),
        host=cfg.get("summarize", "host", default="http://127.0.0.1:11434"),
        temperature=float(cfg.get("summarize", "temperature", default=0.2)),
        num_ctx=int(cfg.get("summarize", "num_ctx", default=8192)),
        max_caption_chars=int(cfg.get("summarize", "max_caption_chars", default=4000)),
        max_transcript_chars=int(cfg.get("summarize", "max_transcript_chars", default=12000)),
        max_tags=int(cfg.get("tags", "max_tags_per_note", default=5)),
        think=bool(cfg.get("summarize", "think", default=False)),
    )
