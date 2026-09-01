# DoomNotes

Turn a backlog of saved Instagram and TikTok videos into structured, searchable markdown
notes in an Obsidian vault.

Saved-video collections are effectively write-only: hundreds of items accumulate and none of
them are searchable. DoomNotes makes them queryable.

```
export → yt-dlp download → faster-whisper transcript → local LLM summary → markdown note
```

Everything runs locally. No API key, no third-party service, no cloud.

---

## Install

```bash
git clone git@github.com:d-q222/DoomNotes.git && cd DoomNotes
make install          # uv venv, dependencies, editable install (puts `doomnotes` on PATH)
make install-hooks    # pre-commit secret guard — hooks do not travel with git
ollama pull qwen3.5:9b
make serve            # start the Ollama daemon
```

Requires Python 3.12+, `ffmpeg`, `yt-dlp`, and [Ollama](https://ollama.com).

## Use

```bash
doomnotes parse                  # read both exports, print a reconciliation
doomnotes parse --register       # record URLs in the store
doomnotes run --limit 30         # process a batch, newest-first
doomnotes status                 # what the store considers processed
doomnotes consolidate --dry-run  # tag pass 2: merge duplicates, split sub-clusters
doomnotes check-auth             # print the cookie-auth probe (run it manually)
```

Configuration is `config.toml`: vault path, batch cap, sleep range, model, per-platform auth.

---

## Architecture

Five stages, each a pure function over the previous stage's output, each independently
runnable. Stage isolation is structural rather than aspirational: a new input source arrives
as a module in `sources/` and touches nothing else.

```
  sources/                    manual · ig_export · tiktok_export
      │                       → Iterable[VideoRef]
      ▼
  store.filter_unprocessed()  authoritative, keyed on URL, never consults the vault
      │                       newest-first queue + batch cap
      ▼
  download.py                 yt-dlp + per-platform auth + failure classification
      │        └──────────────┐
      ▼                       │ caption-only path — Instagram only
  transcribe.py               │ faster-whisper
      │        ┌──────────────┘
      ▼
  summarize.py                Ollama, JSON-schema-constrained, tag registry injected
      │
      ▼
  vault.py                    write guard → atomic temp+replace
      │
      ▼
  consolidate.py              tag merge + sub-cluster split
```

Design rationale for each seam is in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

### The asymmetry that shapes everything

The two exports have **opposite** metadata gaps, and this drives most of the design:

|  | caption in export | author | save timestamp |
|---|---|---|---|
| Instagram | ✅ | ✅ | ❌ (file order only) |
| TikTok | ❌ | ❌ | ✅ exact |

Consequence: **a failed download is a missing note on TikTok, but not on Instagram.** The
Instagram export ships the caption, so a deleted or private video still yields a caption-only
note with `has_transcript: false`. TikTok's export contains only `Date` and `Link`, so a
failed TikTok download is unrecoverable. Manual-list sources are in TikTok's position — their
caption arrives with the media.

---

## Layout

```
config.toml              vault path, batch cap, sleep range, model, per-platform auth
schema/note.json         structure of the model's output
src/doomnotes/
  models.py              VideoRef, Media, Note
  sources/               one adapter per input
  store.py               sqlite: url, state, error, attempts, timestamps
  download.py            yt-dlp wrapper, failure classification, pacing
  transcribe.py          transcribe(path) -> str | None
  summarize.py           Ollama client and the system prompt
  tags.py                registry read/write, prompt injection
  consolidate.py         merge + sub-cluster split
  render.py              Note -> markdown, slugs, wikilinks
  journal.py             append-only JSONL record of every run
  vault.py               write guard + atomic write
  pipeline.py            orchestration, stage isolation
  cli.py                 doomnotes {parse,run,status,consolidate,journal,check-auth}
scripts/                 smoke tests, spine test, benchmarks, secret guards
data/                    gitignored: audio/, logs/, state.db, run traces
private/                 gitignored: personal working notes
```

The package directory is `doomnotes` for historical reasons; the project is DoomNotes.

---

## Verification

```bash
make test           # unit tests
make spine          # full pipeline end-to-end on fixtures, zero platform traffic
make isolation      # prove the write guard refuses a protected vault
make smoke-ollama   # prove schema-constrained generation conforms
make smoke-whisper  # prove faster-whisper loads and transcribes
make audit-leaks    # cross-check tracked files AND git history against the real exports
```

`make spine` exercises every branch — caption-only salvage, unrecoverable TikTok failure,
malformed URLs dropped at the source, checkpoint halting, and re-run semantics — without
making a single request to either platform.

Some tests are marked `xfail` on purpose. They assert behaviour that deliberately-naive
baselines do not yet have, so the gap is documented rather than hidden.

---

## Security and privacy

A platform session cookie is a **bearer credential equivalent to a logged-in session**, and
2FA does not protect it — 2FA is a login-time challenge, and the `sessionid` cookie exists to
prove it was already cleared.

- Cookies live in `~/.config/doomnotes/`, mode 600, **outside the repository entirely** — not
  gitignored-but-present.
- They are passed to yt-dlp as a *path* or browser name, never as a value, so nothing lands in
  the process table or shell history. There is a test asserting this.
- `.gitignore` plus a pre-commit guard block credential-shaped strings, `state.db`, platform
  exports, `private/`, and any file containing more than five saved-video URLs.
- `make audit-leaks` goes further: it reads the real exports and searches both the working
  tree and full git history for genuine shortcodes, video ids and creator handles.

Saved-video data — URLs, creator handles, transcripts, notes — is not part of this repository.

### Operating constraints

- Read-only. The tool never likes, follows, comments or messages.
- Runs on a personal machine from a residential IP; never a cloud server.
- Gentle cadence: a capped batch per run with randomised delays between downloads.
- On a captcha, checkpoint, challenge or dead session the run **halts and reports**. It does not
  retry, because retrying is what escalates a challenge.
- Errors that are as likely to describe one video as the session — a bare `HTTP 401`, a single
  consent redirect — are retried rather than halting. A halt leaves the video unprocessed and
  first in the next queue, so halting on a per-video error stops the whole backlog for good.
- An auth failure **costs the video none of its retry attempts**. A stale cookie is not a
  property of the video, and three runs against a dead session would otherwise write off the
  backlog while the actual fault sat in a file on disk. The run summary reports them.
