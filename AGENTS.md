# Repository Instructions — DoomNotes

## 1. Product boundary

A local pipeline turning saved Instagram/TikTok videos into structured markdown notes in an
Obsidian vault: `export → download → transcribe → summarize → note`.

Everything runs on one machine. No API key, no third-party service, no cloud. Summarization is a
local Ollama model.

Out of scope unless explicitly asked: OCR of on-screen text, the platform "likes" lists, video
retention beyond transcription, prompt caching, batch APIs.

## 2. Read only the context needed

Search for symbols and call sites before opening whole files. Read targeted line ranges after a
search locates the code. Reuse facts already established in the current task; do not re-derive the
export formats, the parser's nesting behaviour, or the model's schema semantics — they are recorded
in `docs/ARCHITECTURE.md`.

Do not ask for information determinable from the repository.

## 3. Architecture and ownership

Each module owns one responsibility. Put a change in the module that already owns it.

| module | owns |
|---|---|
| `models.py` | the data shapes crossing stage boundaries |
| `sources/` | one adapter per input; nothing downstream knows which source produced a ref |
| `store.py` | what counts as processed; the queue |
| `download.py` | yt-dlp invocation, auth, failure classification, pacing |
| `transcribe.py` | audio → text, or `None` for no usable speech |
| `summarize.py` | the model call and the system prompt |
| `tags.py` / `consolidate.py` | tag vocabulary; pass 1 injection, pass 2 repair |
| `render.py` | `Note` → markdown, slugs, wikilinks |
| `journal.py` | append-only JSONL record of every run — records, never reacts |
| `vault.py` | **every** filesystem write under the vault |
| `pipeline.py` | orchestration and failure isolation only |

A new input source is a module in `sources/` and nothing else. If adding one requires editing
`pipeline.py`, the seam is in the wrong place — fix the seam, don't thread the special case.

## 4. Non-negotiable invariants

These are never what gets minimized. §6.1 trims scope, not correctness.

### Vault isolation

`vault.py` is the single enforcement point. There is deliberately no other write path.

- Validate before creating anything. A refused write must leave zero trace — not even a directory.
- Resolve both the root and the candidate, then compare with `Path.relative_to`. Never
  `str.startswith`: it accepts sibling directories (`…-backup`).
- Any vault root inside an iCloud-synced location is refused outright.
- `VaultGuardError` is re-raised, never swallowed. No isolation policy may downgrade it to a warning.
- Stage to a temp file **in the destination directory** (`os.replace` is only atomic within one
  filesystem) with a non-`.md` suffix (Obsidian watches the directory).

### Secrets

- Session cookies live outside the repository, mode 600. A `sessionid` cookie is a bearer credential
  equivalent to a logged-in session; 2FA does not protect it.
- Cookies reach yt-dlp as a **path or browser name, never a value** — nothing in the process table
  or shell history.
- Never log, print, or interpolate cookie contents.

### Personal data never enters the repository

Saved-video URLs, shortcodes, TikTok video ids, creator handles, transcripts, notes, `state.db`, and
platform export paths containing an account name or archive token. Code and documentation only.

- `.gitignore` + `scripts/check_no_secrets.sh` (shape check) + `make audit-leaks` (identity check
  against the real exports, working tree **and** history).
- Personal working notes belong in `private/`, which is gitignored.
- Do not hardcode an export path containing an account name; take it from config or an argument.

### Platform behaviour

Read-only always — never like, follow, comment or message. Capped batch per run with randomised
delays. On any captcha, checkpoint or unexpected redirect: **halt and report, do not retry.**
Retrying is what escalates a challenge.

### Store authority

The store is authoritative and keys on canonical URL. It must never consult the vault to decide what
is processed — if it did, deleting a note would silently re-download and re-summarise it.

## 5. Deliberately naive code — do not "fix" silently

Several functions are intentionally simplistic, each marked with a `DELIBERATELY NAIVE` /
`UNTUNED` / `DELIBERATELY PERMISSIVE` block stating what it does, why that is insufficient, and what
is intended. They are decisions awaiting a judgement call, not bugs.

Currently: pacing (`download.py`), failure-state handling (`store.py`),
isolation granularity (`pipeline.py`), registry injection (`tags.py`), Whisper model size
(`config.toml`).

Their gaps are pinned by `xfail` tests that assert the missing behaviour. Do not delete an `xfail`
marker to make a suite green — implementing the behaviour is what removes it.

If asked to improve one, implement it properly and remove the marker and the block together. Do not
leave the block describing behaviour that no longer exists.

## 6. Work process

1. Restate the acceptance criteria internally.
2. Inspect the smallest relevant execution path.
3. Check existing tests and local conventions before designing new abstractions.
4. Produce a short plan when multiple modules or the note schema are affected.
5. Implement the smallest coherent change.
6. Run focused tests first; the full suite once stable.
7. Review the diff for vault escapes, leaked personal data, swallowed guard errors, and scope creep.
8. Update documentation only where behaviour, setup, or the note contract changed.

Do not perform unrelated refactors, dependency upgrades, or formatting churn while implementing
something else.

### 6.1 Before writing new code (necessity ladder)

Read the relevant execution path first — be lazy about the solution, never about reading. Walk this
ladder in order and stop at the first step that satisfies the requirement:

1. Does this need to exist at all? If it is speculative or "for later", skip it.
2. Does the codebase already do this? Reuse the existing module.
3. Does the standard library or an installed dependency cover it? Use it; add nothing.
4. Is it a one-line or single-function change? Write that; do not scaffold around it.
5. Only then write the minimum new code, in the module that already owns that responsibility (§3).

Do not add a class, module, config layer, dependency or "future override" hook until a second
concrete caller exists.

### 6.2 When structure has earned its place (add it now)

The counterweight. When one of these fires, the abstraction is justified by a second real instance,
not a hypothetical:

- Third copy of the same logic → extract the helper now, not before.
- A previously-skipped hook gains an actual second caller → build it.
- One logical change forces edits in 3+ spots → a seam is missing.
- A function stops fitting on a screen or mixes concerns → split by responsibility per §3.
- A test is hard to write because logic is tangled with I/O → separate the pure part.
- The same 4+ args are threaded everywhere → a small data object is now cheaper.

## 7. Delegation

The main agent owns requirement interpretation, architecture, security decisions, and final
verification. Delegate only work that is bounded, independently verifiable, and cheaper than
carrying its raw output in the main context.

Prefer zero subagents for small tasks, one for ordinary ones, at most three unless the work is
genuinely independent. Do not parallelize coupled writes. Do not let a subagent spawn another. Do
not ask several agents the same question to manufacture a council.

Anything touching the vault guard, cookie handling, or what reaches the repository warrants an
independent privacy review of the finished diff.

Require concise findings with file paths and unresolved uncertainties — not raw logs. Verify
consequential worker conclusions yourself; a confident review can still be wrong.

## 8. Token and tool efficiency

- Use deterministic tools before model reasoning: `rg`, `find`, `pytest`, `git`.
- Search for symbols before opening large files; read targeted ranges.
- Run the narrowest relevant test while iterating; not the full suite after every edit.
- Summarize failures; do not paste complete logs unless the exact text diagnoses the problem.
- Do not generate boilerplate an existing helper already covers.

## 9. Coding standards

- Target Python 3.12+. Preserve `from __future__ import annotations`.
- Plain functions and modules. No classes or frameworks without a clear need.
- Type hints on public functions; short docstrings stating inputs, outputs and assumptions.
- Keep network and subprocess calls behind narrow, injectable functions — that is what makes the
  offline spine test possible. Every stage collaborator must be substitutable.
- Never let a failed download crash a run; classify and continue.
- Prefer clear code over compressed clever code.

## 10. Testing policy

```bash
make test          # unit tests
make spine         # full pipeline on fixtures, zero platform traffic
make isolation     # the write guard refuses a protected vault
make audit-leaks   # working tree AND git history vs the real exports
```

- **No test may make a request to Instagram or TikTok.** Inject a fixture downloader.
- Tests must not write to a real vault. Use `tmp_path`.
- `xfail` markers are specifications (§5), not failures to suppress. `strict=True` where the naive
  behaviour is well-defined — an `xpass` means a test passed for the wrong reason.
- A parser test asserts a *reconciliation*, not a hardcoded count: the exports' own label counts
  disagree with each other.

## 11. Documentation

`README.md` and `docs/ARCHITECTURE.md` are outward-facing: they describe the project, never address
a particular reader. Personal notes, checklists and session handoffs go in `private/`.

When a change invalidates a documented claim, correct the claim in the same change. Several existing
notes record findings that contradicted earlier assumptions — that pattern is deliberate and worth
continuing. State what was measured, not what was assumed.
