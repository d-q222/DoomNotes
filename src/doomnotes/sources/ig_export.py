"""Instagram `saved_posts.html` export parser.

Keys on **label text** (`URL`, `Caption`, `Username`, `Name`), never on CSS
classes: `_a6_q` / `_a6_r` are Meta's obfuscated names and change between
exports.

The non-obvious part is nesting. Each post's `<table>` contains *sub-tables*:

    <table>                          <- the post
      <tr><td>URL<div><a href=…>     <- post url
      <tr><td>Caption</td><td>…      <- post caption
      <tr><td><h2>Owner</h2>
                <table>              <- sub-table
                  <tr><td>Username</td><td>handle
                  <tr><td>Name</td><td>Display Name
      <tr><td><h2>Hashtags</h2>
                …<div>Name</div>     <- ALSO called "Name"

`Name` therefore means two different things depending on which block it sits in.
A document-wide label lookup assigns a hashtag ("bestaitools") as the post's
author. So every field is read from within its own scope: post-level rows are
read with `recursive=False`, and author fields only from inside the `Owner`
block, located by its `<h2>` heading text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from bs4 import BeautifulSoup, Tag

from doomnotes.models import VideoRef
from doomnotes.sources.base import canonical_url, is_instagram_post_url

# Discovered, never hardcoded: an Instagram export directory is named
# `instagram-<account>-<date>-<token>`, so writing one into source would commit
# the account name and archive token to the repository. Globbing keeps identity
# out of version control and survives the next export having a different name.
EXPORT_GLOB = "instagram-*/your_instagram_activity/saved/saved_posts.html"
SEARCH_DIRS = (Path("~/Downloads").expanduser(), Path.cwd())


def find_export() -> Path:
    """Locate the most recent saved-posts export, or raise with what to do."""
    matches: list[Path] = []
    for directory in SEARCH_DIRS:
        if directory.is_dir():
            matches.extend(directory.glob(EXPORT_GLOB))
    if not matches:
        raise FileNotFoundError(
            "No Instagram export found. Looked for "
            f"{EXPORT_GLOB!r} under {', '.join(str(d) for d in SEARCH_DIRS)}. "
            "Pass an explicit path instead."
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


@dataclass
class Reconciliation:
    """What the parser saw, so counts are explained rather than asserted.

    The export's own label counts disagree with each other, so any hardcoded
    expectation would fail and explain nothing. This reports instead.
    """

    tables_scanned: int = 0
    post_tables: int = 0
    refs_emitted: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    dropped: list[tuple[str, str]] = field(default_factory=list)
    missing_caption: int = 0
    missing_author: int = 0
    duplicate_urls: int = 0
    label_census: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            "Instagram export reconciliation",
            "-" * 60,
            f"  <table> elements scanned      : {self.tables_scanned}",
            f"  post tables (have a URL row)  : {self.post_tables}",
            f"  refs emitted                  : {self.refs_emitted}",
            "  by url kind:",
        ]
        for kind, n in sorted(self.by_kind.items()):
            lines.append(f"      {kind:<10} {n}")
        lines.append(f"  dropped                       : {len(self.dropped)}")
        for url, why in self.dropped:
            lines.append(f"      {why}: {url}")
        lines.append(f"  duplicate urls collapsed      : {self.duplicate_urls}")
        lines.append(f"  refs with no caption          : {self.missing_caption}")
        lines.append(f"  refs with no author           : {self.missing_author}")
        lines.append("  raw label census (whole document, all nesting levels):")
        for label, n in sorted(self.label_census.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {label:<12} {n}")
        lines.append(
            "  note: document-wide 'Name'/'Caption' counts exceed the post count\n"
            "        because sub-blocks reuse those labels. Scoped extraction is\n"
            "        why 'refs emitted' does not match them."
        )
        return "\n".join(lines)


def _direct_label(cell: Tag) -> str | None:
    """The cell's own text, ignoring nested tags.

    `<td>URL<div><a>…</a></div></td>` -> "URL"
    `<td>Caption</td>`                -> "Caption"
    `<td><div>…<h2>Owner</h2>…</td>`  -> None (no direct text)
    """
    direct = "".join(
        str(child) for child in cell.children if isinstance(child, str)
    ).strip()
    return direct or None


def _rows(table: Tag) -> list[Tag]:
    """Direct `<tr>` children only — never rows belonging to a sub-table."""
    return [tr for tr in table.find_all("tr", recursive=False)]


def _block_by_heading(scope: Tag, heading: str) -> Tag | None:
    """Find the sub-block introduced by `<h2>heading</h2>` within `scope`.

    Located by heading *text*, which is stable, rather than by class name.
    """
    for h2 in scope.find_all("h2"):
        if h2.get_text(strip=True) == heading:
            parent = h2.parent
            return parent if isinstance(parent, Tag) else None
    return None


def _labelled_values(scope: Tag, wanted: frozenset[str]) -> dict[str, str]:
    """Read `label -> value` pairs from anywhere inside `scope`.

    Searches every depth *within the given block* rather than one fixed level.
    The Owner block wraps its real table in another table, so a single
    `scope.find("table")` lands on the wrapper and finds nothing.

    Scoping is still what keeps this correct: `Hashtags` is a sibling block, not
    a descendant of `Owner`, so hashtag `Name` entries are out of reach here.
    `wanted` narrows further, so the owner's bio-link `URL` row is ignored.
    """
    out: dict[str, str] = {}
    for tr in scope.find_all("tr"):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 2:
            continue
        label = _direct_label(cells[0])
        if label in wanted:
            value = cells[1].get_text(strip=True)
            if value:
                out.setdefault(label, value)
    return out


def _is_post_table(table: Tag) -> bool:
    """A post table has a URL row containing an actual anchor.

    The anchor requirement matters: owner blocks also carry a `URL` row for the
    creator's bio link, but as plain text. Without this check, 223 owner
    sub-tables are miscounted as posts.
    """
    for tr in _rows(table):
        for td in tr.find_all("td", recursive=False):
            if _direct_label(td) == "URL" and td.find("a", href=True):
                return True
    return False


def _url_kind(url: str) -> str:
    for kind in ("reel", "reels", "p", "tv", "channel"):
        if f"/{kind}/" in url:
            return kind
    return "other"


def parse(
    path: str | Path | None = None,
) -> tuple[list[VideoRef], Reconciliation]:
    """Parse the export. Returns refs in file order (newest-saved first)."""
    path = Path(path).expanduser() if path else find_export()
    html = Path(path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    rec = Reconciliation()

    # Whole-document census, for the reconciliation only — never for extraction.
    for cell in soup.find_all(["td", "div"]):
        label = _direct_label(cell)
        if label in {"URL", "Caption", "Username", "Name"}:
            rec.label_census[label] = rec.label_census.get(label, 0) + 1

    all_tables = soup.find_all("table")
    rec.tables_scanned = len(all_tables)

    refs: list[VideoRef] = []
    seen: set[str] = set()

    for table in all_tables:
        if not _is_post_table(table):
            continue
        rec.post_tables += 1

        url: str | None = None
        caption: str | None = None

        for tr in _rows(table):
            cells = tr.find_all("td", recursive=False)
            if not cells:
                continue
            label = _direct_label(cells[0])

            if label == "URL":
                anchor = cells[0].find("a", href=True)
                if anchor:
                    url = anchor["href"].strip()
            elif label == "Caption" and len(cells) >= 2:
                caption = cells[1].get_text().strip() or None

        if not url:
            continue

        kind = _url_kind(url)
        rec.by_kind[kind] = rec.by_kind.get(kind, 0) + 1

        if not is_instagram_post_url(url):
            rec.dropped.append((url, f"not a fetchable post (/{kind}/)"))
            continue

        # Author is read ONLY from inside the Owner block. Reading "Name"
        # document-wide would pick up hashtag names instead.
        author = None
        owner = _block_by_heading(table, "Owner")
        if owner is not None:
            fields = _labelled_values(owner, frozenset({"Username", "Name"}))
            author = fields.get("Username") or fields.get("Name")

        curl = canonical_url(url)
        if curl in seen:
            rec.duplicate_urls += 1
            continue
        seen.add(curl)

        if not caption:
            rec.missing_caption += 1
        if not author:
            rec.missing_author += 1

        refs.append(
            VideoRef(
                url=curl,
                platform="instagram",
                # The export ships the caption, which is what makes a
                # caption-only note possible when the download fails.
                caption_source="export",
                caption=caption,
                author=f"@{author}" if author and not author.startswith("@") else author,
                source_order=len(refs),
            )
        )

    rec.refs_emitted = len(refs)
    return refs, rec


class InstagramExportSource:
    """Source adapter over the saved-posts HTML export."""

    name = "instagram_export"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path else find_export()

    def fetch(self) -> Iterator[VideoRef]:
        refs, _ = parse(self.path)
        yield from refs

    def fetch_with_reconciliation(self) -> tuple[list[VideoRef], Reconciliation]:
        return parse(self.path)
