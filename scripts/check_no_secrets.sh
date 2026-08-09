#!/usr/bin/env bash
# Pre-commit guard: refuse to stage anything that would publish secrets or
# personal saved-video data.
#
# .gitignore prevents the accidental case. This catches the deliberate-looking
# one: a `git add -f`, a file written to an unexpected path, or a doc that grew
# a pasted corpus. It inspects STAGED CONTENT, not the working tree.
#
# Install:  make install-hooks       (or: ln -s ../../scripts/check_no_secrets.sh .git/hooks/pre-commit)
# Bypass:   git commit --no-verify   (don't)

set -uo pipefail

fail=0
red()  { printf '\033[31m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }

staged=$(git diff --cached --name-only --diff-filter=ACM)
[ -z "$staged" ] && exit 0

# ── 1. Forbidden paths ────────────────────────────────────────────────────
while IFS= read -r f; do
  case "$f" in
    data/*|*/data/*|*.db|*.sqlite*|*cookies*|.env|.env.*|*.pem|*.key)
      red "BLOCKED  $f  — this path is never publishable"
      fail=1 ;;
    *saved_posts.html|*user_data_tiktok.json|*saved_collections.html)
      red "BLOCKED  $f  — platform export (personal saved-video data)"
      fail=1 ;;
    private/*|*/private/*)
      red "BLOCKED  $f  — private/ holds personal working docs, not repo docs"
      fail=1 ;;
  esac
done <<< "$staged"

# ── 2. Credential-shaped content ──────────────────────────────────────────
# Session cookies are the critical one: an Instagram sessionid is a bearer
# credential equivalent to a logged-in session, and 2FA does not protect it.
# NOTE on format: yt-dlp consumes Netscape cookie jars, which are TAB-separated
# with no '=' at all:
#     .instagram.com<TAB>TRUE<TAB>/<TAB>TRUE<TAB>1799999999<TAB>sessionid<TAB>73621459812%3A...
# An '=' -anchored pattern misses a live credential entirely, so each cookie name
# is matched both as `name=value` and as a bare field followed by whitespace.
patterns=(
  'sessionid[=[:space:]]+[A-Za-z0-9%_.-]{10,}'
  'csrftoken[=[:space:]]+[A-Za-z0-9%_.-]{10,}'
  'ds_user_id[=[:space:]]+[0-9]{5,}'
  '# Netscape HTTP Cookie File'  # doomnotes:allow-fixture (self-match: this is the pattern, not a cookie)
  '^\.?(instagram|tiktok)\.com[[:space:]]+(TRUE|FALSE)[[:space:]]'
  'BEGIN [A-Z ]*PRIVATE KEY'
  'gh[pousr]_[A-Za-z0-9]{30,}'
  'sk-ant-[A-Za-z0-9_-]{20,}'
  'AKIA[0-9A-Z]{16}'
  'xox[baprs]-[A-Za-z0-9-]{10,}'
)
# Escape hatch, deliberately narrow: a line carrying the literal pragma
#   doomnotes:allow-fixture
# is exempt. It has to be written on the same line as the match, so it shows up
# in review right next to the thing it excuses. Used by tests that assert a
# credential does NOT leak — they necessarily contain a credential-shaped string.
for pat in "${patterns[@]}"; do
  if hits=$(git diff --cached -U0 | grep -Ea "^\+.*${pat}" 2>/dev/null \
            | grep -v 'doomnotes:allow-fixture'); then
    red "BLOCKED  credential-shaped string matching /${pat}/:"
    printf '%s\n' "$hits" | head -5 | sed 's/^/         /'
    red "         If this is a test fixture, append:  # doomnotes:allow-fixture"
    fail=1
  fi
done

# ── 3. Bulk personal data ─────────────────────────────────────────────────
# A doc may legitimately cite an example URL. A pasted corpus of saved videos is
# a different thing. Threshold, not a ban.
THRESHOLD=5
while IFS= read -r f; do
  [ -f "$f" ] || continue
  # File-level pragma for files that are ALL synthetic fixtures. Must be a
  # deliberate edit to the file itself, so it can't happen by accident.
  if git show ":$f" 2>/dev/null | grep -q 'doomnotes:synthetic-urls'; then
    warn "note     $f  — declared synthetic fixtures, bulk-URL check skipped"
    continue
  fi
  n=$(git show ":$f" 2>/dev/null \
      | grep -oEa 'instagram\.com/(reel|reels|p|tv)/[A-Za-z0-9_-]{5,}|tiktokv?\.com/(share/)?video/[0-9]{6,}' \
      | sort -u | wc -l | tr -d ' ')
  if [ "${n:-0}" -gt "$THRESHOLD" ]; then
    red "BLOCKED  $f  — contains $n distinct saved-video URLs (limit $THRESHOLD)"
    red "         That is a saved-video corpus. Keep it in data/."
    fail=1
  elif [ "${n:-0}" -gt 0 ]; then
    warn "note     $f  — cites $n saved-video URL(s); fine as an example"
  fi
done <<< "$staged"

if [ "$fail" -ne 0 ]; then
  red ""
  red "Commit refused. Nothing was committed."
  exit 1
fi
exit 0
