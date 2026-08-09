# Architecture and design decisions

Why each seam in the pipeline is drawn where it is, and what breaks if it moves.

Ordered by data flow. Every claim here was verified by running the code, not assumed.

---

## Sources → `VideoRef`

A source is anything satisfying a two-member protocol: a `name`, and `fetch()` returning an
iterable of `VideoRef`. Nothing downstream knows which source produced a ref.

**Why the abstraction earns its keep with only three implementations:** a fourth is planned (a
browser-driven scraper for the recent delta). The test of whether the seam is in the right
place is whether adding it requires editing `pipeline.py`. If it does, the protocol was drawn
wrong.

**Why both `caption` and `saved_at` are optional:** the two exports have opposite gaps.
Instagram ships captions and authors but no dates; TikTok ships exact save timestamps and
nothing else. Neither field can be required, and `has_export_caption` is where that asymmetry
becomes a decision the pipeline can act on.

**Why ordering is per-source and never merged:** Instagram has no save timestamps at all, so
there is no shared key on which to interleave the two queues. Each source drains as its own
newest-first queue. Instagram's file order *is* newest-saved-first — its shortcodes encode a
monotonically increasing media id — but it is ordered by save time, not post time, so it is
not perfectly monotonic. It is also the only recency signal available before downloading
anything, which is why `source_order` exists.

### Parsing the Instagram HTML export

The export is HTML, and the parser keys on **label text** (`URL`, `Caption`, `Username`,
`Name`) rather than CSS class. The class names are obfuscated build artifacts (`_a6_q`,
`_a6_r`) and change between exports; a class-keyed parser breaks silently on the next
download.

**Label text alone is not sufficient, which is the non-obvious part.** Labels repeat at
different nesting depths within a single post's table:

- `Name` labels the post owner's display name **and** every hashtag.
- `URL` labels the post URL **and** the creator's bio link.

A document-wide label lookup therefore attributes a hashtag as the post's author. Extraction
is scoped per block: post-level fields come from the post table's *direct* row children, and
author fields only from inside the `Owner` block, located by its `<h2>` heading text.

Two measured consequences:

- Requiring an actual `<a>` element in the URL row matters: 223 owner bio-link rows also carry
  a plain-text `URL` label. Without that check the parser counted 474 "posts" instead of 251.
- The `Owner` block nests its real table inside a wrapper table, so a single `find("table")`
  lands on the wrapper and returns nothing.

**The parser reports a reconciliation rather than asserting a count.** The export's own label
counts disagree with each other — 474 `URL` labels, 259 `Caption`, 252 `Username`, for 251
posts — because sub-blocks reuse the labels. A hardcoded expectation would fail and explain
nothing; a reconciliation explains the difference.

### Parsing the TikTok JSON export

Favourites live at `Likes and Favorites → Favorite Videos → FavoriteVideoList`, as
`{"Date", "Link"}` pairs.

**Redirects are deliberately not resolved at parse time.** Resolving `tiktokv.com/share/…`
would mean one network request per entry just to build a queue, before any pacing logic
applies. The numeric video id is already present in the share URL, so a stable dedup key
needs no network. Parsing stays fully offline.

**But the export's URL form is not what gets requested.** An earlier version of this
document justified the above with "yt-dlp follows the redirect itself at download time".
That is true only via the generic extractor: checked against yt-dlp 2026.07.04's extractor
table offline, **no TikTok extractor matches the `tiktokv.com` host**. The URL falls through
to `GenericIE`, which fetches it, follows the redirect and re-dispatches — an extra request
per video, and it stakes every TikTok download on the redirect landing somewhere the TikTok
extractor recognises rather than on a login or consent interstitial.

So identity and fetch are separated. `ref.url` stays exactly as the export wrote it, because
it is the store's primary key and rewriting it would orphan every existing row. What gets
requested is `download.fetch_url(ref)`:

```
store key :  https://www.tiktokv.com/share/video/<id>/
requested :  https://www.tiktok.com/@_/video/<id>
```

`@_` is yt-dlp's own convention, not an invention here — `TikTokBaseIE._create_url` emits
`@{user_id or "_"}` when the uploader is unknown, which is exactly this situation. This
matters asymmetrically: a failed TikTok download has no export caption to fall back on, so
it yields no note at all.

Unverifiable without TikTok traffic, by construction. `make check-auth` prints this exact
form so the manual gate tests what a real run does, and reverting is one line.

---

## Store

SQLite, keyed on **canonical URL**, authoritative over the filesystem.

**Why canonical URL:** Instagram appends `?igsh=…` share tokens that differ per copy of the
same link. Without normalisation the same video re-enters the queue indefinitely.

**Why the store must never consult the vault:** if `filter_unprocessed` checked whether a note
file existed, deleting a note would silently re-download and re-summarise it — real requests
against a real account, triggered by tidying a vault. The store is the source of truth for
what has been processed; the vault is output. A test pins this.

**Re-run semantics:** registration uses `INSERT OR IGNORE`, so re-parsing an export never
resets a completed row back to pending.

---

## Download

**Cookies reach yt-dlp as a path or a browser name, never as a value**, so nothing sensitive
enters the process table or shell history. A test asserts the cookie's contents never appear
in the constructed argv.

**Auth is configured per platform**, not globally, because the two platforms can land on
different answers — browser-keychain extraction may work for one and not the other — and that
should not require a refactor.

**Failures classify three ways**, and the distinction is load-bearing:

| class | examples | response |
|---|---|---|
| terminal | video unavailable, private post, 404 | log, move on, never retry |
| retryable | 429, 5xx, timeouts, DNS | back off and requeue |
| **stop** | checkpoint, challenge, login required | **halt the run, report, do not retry** |

The stop class exists because retrying a challenge is what escalates it. Misclassification is
asymmetric in cost: on TikTok there is no export caption, so treating a transient failure as
terminal loses that video permanently, with no note and no second chance.

**`posted_at` comes from yt-dlp metadata**, which only exists if the download succeeded. It is
therefore absent on caption-only notes — the field intended to judge recency is missing on
exactly the notes whose content also cannot be judged. That is an interaction between two
design choices, documented rather than hidden.

---

## Transcribe

The interface is deliberately narrow: `transcribe(path) -> str | None`.

**`None` means "no usable speech", not "error"** — a silent clip, a music-only video, a photo
post with no audio track. The caller turns that into `has_transcript: false`, which is a
legitimate note rather than a failure. A minimum-character floor prevents a six-character ASR
artifact being fed to the summariser as though it were content.

**Performance note:** faster-whisper runs on CTranslate2, which has no Metal backend. On Apple
Silicon this is CPU-only. A Metal-native alternative is a drop-in behind this same interface,
which is the reason the interface is one function.

**Transcripts are kept and wikilinked rather than inlined or discarded.** Two reasons: a
schema change or model swap becomes a free local re-summarisation pass instead of
re-downloading everything, and a claim in a summary can be checked against what was actually
said. ASR is wrong often enough to matter — in a verification run the model heard "wear
clause" for "where clause".

---

## Summarize

Ollama's `format` parameter takes a full JSON Schema and compiles it to a llama.cpp grammar,
so conformance is a **constraint on generation**, not a request in the prompt.

> **JSON Schema `description` fields constrain nothing.** The grammar enforces types, shapes
> and array bounds; descriptions may never reach the model at all. Editing a description and
> expecting different output does not work — verified the hard way, when a schema saying
> "descriptive title, not a slug" still produced slugs. The actual cause was a line in the
> system prompt reading *"It becomes a filename."*
>
> **`schema/note.json` is the source of truth for structure. The system prompt is the only
> channel for meaning.**

**The model never generates `source_url`, `platform` or dates.** A model asked for a URL will
invent one. Those fields come from the ref and from download metadata, and
`additionalProperties: false` enforces the boundary.

**Reasoning mode is a measured setting, not a default.** The configured model is a reasoning
model; left enabled it spends tens of thousands of thinking characters per caption. Measured
on two real captions: 147.0 s and 79.2 s with reasoning on, versus 4.1 s and 8.6 s with it
off, for equal or better titles. Across a few hundred videos that is the difference between
hours and minutes. The knob is in `config.toml` with the measurement recorded next to it.

**The tag registry is injected into every call**, with an instruction to reuse an existing tag
when one fits and mint a new one only when nothing does. This is **order-dependent by
construction**: the first video sees an empty registry, the last sees a mature one. That is
inherent to a growing vocabulary, and the consolidation pass is what repairs it.

---

## Vault write guard

The single enforcement point for the project's one non-negotiable constraint: never write
outside the configured vault. There is deliberately no other write path.

Three ways to get this wrong, each with a test:

**1. `str.startswith` accepts sibling directories.**
```python
str("/…/ai-notes-vault-backup/x.md").startswith(str("/…/ai-notes-vault"))  # True
```
`ai-notes-vault-backup` is a sibling, not a child. `Path.relative_to()` compares path
*components*, so it rejects what a string prefix accepts. The test asserts that the naive
check *would* have passed.

**2. Checking before resolving loses to `..` and symlinks.** A path can be textually under the
root and not under it on disk. Both sides are resolved before comparison.

**3. Guarding after `mkdir` still creates directories in a protected location.** Validation is
the first statement in the write path, and a test snapshots the directory tree before and
after a refused write to assert nothing was created.

There is also a root-level guard: any vault root inside an iCloud-synced Obsidian location is
refused outright, so a misconfiguration cannot point the pipeline at real notes.

**Atomic writes:** content is staged to a temp file in the *destination directory* — not
`/tmp`, because `os.replace` is only atomic within a single filesystem, and a cross-device
move degrades to copy-then-delete, which is exactly the half-written file staging was meant to
prevent. The temp file has a non-`.md` suffix so Obsidian's watcher never indexes a partial
note.

**Note and transcript are written as a pair, with both paths guarded up front**, so a wikilink
can never dangle and a bad transcript path cannot orphan a note.

### Filenames

The filename is the title slug, with a short hash of the URL appended on collision. The hash
comes from the URL rather than a counter so that reprocessing one video lands on the same
filename instead of accumulating `-2`, `-3` copies.

Those two requirements pull against each other, and the tension is the design:

| situation | correct behaviour |
|---|---|
| a different video wants this name | suffix it, keep both notes |
| this video already owns it | reuse it, overwrite in place |

From the filename alone the two are indistinguishable, so the index maps slug → owning
`source_url` and is **rebuilt from the vault at the start of every run**. Accumulating it in
memory only ever knew what the current run had written — and since runs are separate processes
days apart, two videos that generated the same title on different days silently resolved to
one file, with the store still recording the lost video as done.

A note with no readable `source_url` — anything hand-written — still reserves its slug. An
unknown owner is treated as "not ours", because the cost of guessing wrong is destroying
something the user wrote.

**Names are compared the way the filesystem compares them, not the way Python does.** macOS
ships APFS case-insensitive, so `weekly-review-notes.md` and `Weekly-Review-Notes.md` are one
file. A case-sensitive index believes the lowercase name is free, `os.replace` disagrees, and
the existing note is destroyed with no error and no suffix. Slugs are always lowercase, so the
only files this can collide with are ones the pipeline did not write — hand-made notes, or its
own notes renamed in Obsidian. Exactly the files with no second copy.

**Folding the key space needs a conflict rule.** Once names are compared case-insensitively,
pairs that used to be distinct collide, and letting the last one win turns the safety
mechanism into the bug it was added to prevent. Two names that fold together but disagree
about their owner are marked **contested** — occupied, attribution unknown — so nobody gets
the fast path to that name and every claimant is suffixed away. Losing a filename is
recoverable; losing the note under it is not.

Frontmatter is parsed as a block and then searched, rather than scanned for from the top of
the file. A regex reaching for `source_url:` looks anchored but walks straight past the
closing `---`, so an unindented `source_url:` line in a note's *body* reads as that note's
owner. Quotes are stripped, because Obsidian's Properties editor re-saves the value quoted and
a note would otherwise stop recognising itself the moment it was opened.

---

## Consolidation

A second pass over the registry and note frontmatter, idempotent and re-runnable.

**Merging** collapses near-duplicates onto a canonical tag — the most-used variant, tie-broken
by length then alphabetically, so the result is deterministic.

Matching uses **derivational suffixes** (`-ing`, `-ed`, `-er`, `-s`, with consonant doubling
and dropped trailing `e`) in addition to a similarity ratio. This is deliberate: `garden` and
`gardening` score 0.80 on a sequence matcher, below a sensible threshold, and lowering the
threshold starts merging genuinely distinct tags. A plain prefix rule would merge `post` into
`postgres`; a suffix rule does not.

**Both rules build equivalence groups; a canonical is elected once per group.** They are not
applied in sequence, because that lets the first rule consume the token the second one needs:

```
{garden: 5, gardens: 2, gardening: 9}  ->  all three collapse            ✅
{garden: 1, gardens: 2, gardening: 9}  ->  `gardening` is stranded       ❌
```

Same words, different tallies, different answer — when the plural outnumbers the singular,
stem-grouping elects `gardens`, and `gardens`/`gardening` is not a derivation, so the bridge
disappears. Whether two tags merge must depend on the words, not on how often each was used.

That failure was also permanent: once notes had been rewritten from `garden` to `gardens`,
the bridging form no longer existed in the vault for a later pass to find — which directly
contradicts this pass being safe to re-run as the vault grows.

**Counts are recomputed from the notes; descriptions are carried across.** The notes are the
truth about counts, and a stale count is worse than none. Descriptions cannot be derived from
frontmatter at all, so rebuilding the registry from scratch silently empties them. A canonical
tag with no description of its own inherits from the most-used tag it absorbed; splits inherit
nothing, since a nested child is narrower than its parent and would otherwise be handed a
claim nobody wrote about it.

**Splitting** promotes an oversized tag into Obsidian nested tags using **co-occurrence**: if a
meaningful share of `coding` notes are also tagged `databases`, that is the sub-cluster, and it
is visible without another model call.

---

## Pipeline

Collaborators are injected, which is what makes an offline end-to-end test possible: swapping
the downloader for a fixture exercises every branch with zero platform traffic.

**Failure isolation is the constraint that keeps a new source from breaking the batch
pipeline.** Granularity matters: "the video failed" is not a diagnosis. A 404, an ffmpeg
crash, an out-of-memory transcription and a malformed model response landing in the same
bucket means the failure log cannot identify which stage is broken — and at a few hundred
videos that is the difference between fixing one thing and re-running everything.

**One exception is absolute:** a vault guard error is re-raised, never swallowed. It means a
write was about to land outside the vault. That is a stop, not an item failure, and no
isolation policy may downgrade it to a logged warning.

---

## Run journal

An append-only JSONL record under `data/logs/`, one object per event: run started, each video,
each pacing sleep, run finished. `doomnotes journal` reads it back.

**It records; it does not react.** That constraint is load-bearing rather than minimalist. A
journal that counted consecutive same-stage failures and stopped the run would be making the
failure-isolation decision — and making it inside a module named "logging", where nobody would
look for it. What to *do* about what it records belongs to the isolation policy.

Why it is needed at all: the run summary reports totals, so 29 failures look like 29 problems,
and the store keeps only the latest state per URL, overwritten on each attempt. Neither can
say that videos 6 through 14 failed at the same stage in an unbroken run — which is what a
dead model server looks like from outside, and is one problem rather than nine.

```
run 20260809T0812   14 video(s)
  summarize         9     0.0-0.0s   videos 6-14  <- unbroken run
  caption_only      5     0.0-0.0s   videos 1-5   <- unbroken run
```

JSONL rather than a table or a single document, because a run can be killed at any moment and
a torn write should cost one line rather than the file. Reading skips unparseable lines for
the same reason. A run with no `run_finished` record is one whose process did not reach the
end, which is a different thing from a run that completed with failures.

Every write is failure-tolerant: an unwritable path degrades to no journal, never to a raised
exception. An observability feature that can abort a rate-limited batch is worse than none.

The journal contains saved-video URLs, exactly as the store does. It lives under `data/`,
gitignored and blocked by path in the pre-commit hook, and never inside the vault.
