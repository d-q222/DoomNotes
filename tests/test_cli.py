"""CLI tests.

Every URL in this file is invented (doomnotes:synthetic-urls).

These run against a throwaway config pointed at a temp vault, so no test can
touch the real vault, the real store, or the real exports. The one command not
exercised end to end is `parse`, which by definition reads the exports in
~/Downloads — `test_sources.py` covers the parsers themselves against a
synthetic fixture instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doomnotes.cli import main

CONFIG = """
[vault]
path = "{vault}"
transcripts_dir = "_transcripts"

[paths]
state_db = "{data}/state.db"
audio_dir = "{data}/audio"

[pacing]
batch_cap = 35
sleep_min_s = 20
sleep_max_s = 90
queue_order = ["instagram", "tiktok"]

[auth.instagram]
mode = "none"
browser = "chrome"
cookies_file = "~/.config/doomnotes/cookies.txt"

[auth.tiktok]
mode = "none"
browser = "chrome"
cookies_file = "~/.config/doomnotes/cookies_tiktok.txt"

[download]
timeout_s = 180
audio_format = "m4a"

[transcribe]
model = "base.en"
device = "cpu"
compute_type = "int8"
beam_size = 1
min_transcript_chars = 40

[summarize]
model = "qwen3.5:9b"
host = "http://127.0.0.1:11434"
temperature = 0.2
num_ctx = 8192
max_caption_chars = 4000
max_transcript_chars = 12000
think = false

[tags]
registry_file = "_meta/tags.json"
max_tags_per_note = 5
split_threshold = 15
merge_similarity = 0.85
"""


@pytest.fixture()
def env(tmp_path: Path):
    vault = tmp_path / "ai-notes-vault"
    (vault / "_meta").mkdir(parents=True)
    (vault / "_transcripts").mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text(CONFIG.format(vault=vault, data=data), encoding="utf-8")
    return {"config": str(cfg), "vault": vault, "data": data, "tmp": tmp_path}


def run_cli(env, *argv: str) -> int:
    return main(["-c", env["config"], *argv])


# ── dispatch ─────────────────────────────────────────────────────────────


def test_no_subcommand_is_an_error_not_a_default_action(capsys) -> None:
    """`doomnotes` alone must not start doing something to an account."""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


@pytest.mark.parametrize("cmd", ["parse", "status", "run", "consolidate", "check-auth"])
def test_every_subcommand_has_help(cmd: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main([cmd, "--help"])
    assert exc.value.code == 0


# ── status ───────────────────────────────────────────────────────────────


def test_status_on_an_empty_store_says_so(env, capsys) -> None:
    assert run_cli(env, "status") == 0
    assert "empty" in capsys.readouterr().out


def test_status_reports_registered_urls(env, capsys) -> None:
    from doomnotes.models import VideoRef
    from doomnotes.store import Store

    with Store(env["data"] / "state.db") as store:
        store.register([VideoRef("https://www.instagram.com/reel/AAAAAAAAAAA/", "instagram")])

    assert run_cli(env, "status") == 0
    out = capsys.readouterr().out
    assert "instagram" in out and "pending" in out


# ── check-auth ───────────────────────────────────────────────────────────


def test_check_auth_makes_no_network_call_and_prints_no_real_url(env, capsys) -> None:
    """It prints commands for a human to run. It must not run them.

    An earlier version of this command shipped a real saved-post shortcode
    inherited from the plan, which is how it ended up in git history. The
    placeholder is load-bearing, so it is asserted.
    """
    import re

    assert run_cli(env, "check-auth") == 0
    out = capsys.readouterr().out
    assert re.search(r"instagram\.com/(reel|reels|p|tv)/[A-Za-z0-9_-]{5,}", out) is None
    assert re.search(r"tiktokv?\.com/(share/)?video/[0-9]{6,}", out) is None


def test_check_auth_never_prints_a_cookie_value(env, capsys) -> None:
    run_cli(env, "check-auth")
    out = capsys.readouterr().out
    for marker in ("sessionid", "csrftoken", "ds_user_id", "sid_tt"):
        assert marker not in out


def test_check_auth_states_what_the_cookie_actually_is(env, capsys) -> None:
    """The revocation path has to be in front of him at the moment he creates it."""
    run_cli(env, "check-auth")
    out = capsys.readouterr().out.lower()
    assert "bearer credential" in out
    assert "2fa does not protect it" in out
    assert "login activity" in out


# ── consolidate ──────────────────────────────────────────────────────────


def test_consolidate_dry_run_writes_nothing(env, capsys) -> None:
    vault: Path = env["vault"]
    (vault / "a.md").write_text(
        "---\ntitle: \"a\"\ntopic: gardens\ntags: [gardens]\n---\n\nbody\n", encoding="utf-8"
    )
    (vault / "b.md").write_text(
        "---\ntitle: \"b\"\ntopic: garden\ntags: [garden]\n---\n\nbody\n", encoding="utf-8"
    )
    snapshot = {p: p.read_bytes() for p in sorted(vault.rglob("*")) if p.is_file()}

    assert run_cli(env, "consolidate", "--dry-run") == 0
    assert "dry run" in capsys.readouterr().out

    after = {p: p.read_bytes() for p in sorted(vault.rglob("*")) if p.is_file()}
    assert after == snapshot


def test_consolidate_writes_the_registry_into_the_vault(env) -> None:
    vault: Path = env["vault"]
    (vault / "a.md").write_text(
        "---\ntitle: \"a\"\ntopic: gardens\ntags: [gardens]\n---\n\nbody\n", encoding="utf-8"
    )
    assert run_cli(env, "consolidate") == 0
    registry = json.loads((vault / "_meta" / "tags.json").read_text(encoding="utf-8"))
    assert "gardens" in registry["tags"]


# ── run ──────────────────────────────────────────────────────────────────


def test_run_refuses_a_guarded_vault_before_doing_anything(env, tmp_path, capsys) -> None:
    """Point the config at a protected root; it must fail at startup, exit 2.

    This is the isolation test at the CLI level: the guard has to fire before
    the store is opened or a single directory is created.
    """
    bad = tmp_path / "config-bad.toml"
    bad.write_text(
        CONFIG.format(
            vault=Path("~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Starting Vault")
            .expanduser(),
            data=env["data"],
        ),
        encoding="utf-8",
    )
    assert main(["-c", str(bad), "run", "--no-sleep"]) == 2
    assert "vault guard refused" in capsys.readouterr().err


def test_run_with_a_zero_limit_touches_no_platform(env, capsys) -> None:
    """Proves the wiring runs end to end without a daemon or a network call.

    `--limit 0` drains an empty queue, so no downloader, transcriber or model
    is ever constructed — but everything around them is exercised for real.
    """
    urls = env["tmp"] / "urls.txt"
    urls.write_text("https://www.instagram.com/reel/AAAAAAAAAAA/\n", encoding="utf-8")

    assert run_cli(env, "run", "--urls", str(urls), "--limit", "0", "--no-sleep") == 0
    assert "Run summary" in capsys.readouterr().out


def test_the_tag_registry_survives_an_interrupted_run(env, monkeypatch) -> None:
    """A paced batch sleeps ~30 minutes in total, so Ctrl-C partway is normal.

    The notes an interrupted run already wrote are on disk regardless. If their
    vocabulary is not persisted with them, the next run's pass-1 prompt gets a
    registry that disagrees with the vault and mints duplicates of tags that
    already exist.
    """
    from doomnotes import cli as cli_mod

    def die_after_contributing_tags(refs, **kwargs):
        kwargs["registry"].observe(["coding", "databases"])
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod, "run_batch", die_after_contributing_tags)

    urls = env["tmp"] / "urls.txt"
    urls.write_text("https://www.instagram.com/reel/AAAAAAAAAAA/\n", encoding="utf-8")
    with pytest.raises(KeyboardInterrupt):
        run_cli(env, "run", "--urls", str(urls), "--no-sleep")

    registry = json.loads((env["vault"] / "_meta" / "tags.json").read_text(encoding="utf-8"))
    assert set(registry["tags"]) == {"coding", "databases"}


def test_check_auth_shows_the_tiktok_url_form_a_real_run_will_request(env, capsys) -> None:
    """The probe has to test what the pipeline does, or it proves nothing.

    yt-dlp has no extractor for the tiktokv.com/share/ host in the export, so
    the pipeline rewrites TikTok refs to yt-dlp's own canonical form before
    requesting them. If check-auth still showed the export's URL, a passing
    probe would say nothing about whether a batch will work.
    """
    run_cli(env, "check-auth")
    out = capsys.readouterr().out
    assert "https://www.tiktok.com/@_/video/<ID>" in out
    assert "fetch_url" in out, "and how to revert it if the probe says otherwise"


def test_check_auth_gives_a_pasteable_url_template_per_platform(env, capsys) -> None:
    """The probe is copy-pasted at 8am. Ambiguity there costs real time."""
    run_cli(env, "check-auth")
    out = capsys.readouterr().out
    assert "https://www.instagram.com/reel/<SHORTCODE>/" in out
    assert "https://www.tiktok.com/@_/video/<ID>" in out


def test_run_reports_dropped_manual_urls_rather_than_failing_them(env, capsys) -> None:
    """A malformed line is a parse problem, not a download problem."""
    urls = env["tmp"] / "urls.txt"
    urls.write_text(
        "https://www.instagram.com/reel/AAAAAAAAAAA/\nnot-a-url\n", encoding="utf-8"
    )
    assert run_cli(env, "run", "--urls", str(urls), "--limit", "0", "--no-sleep") == 0
    assert "dropped" in capsys.readouterr().out


def test_a_failed_registry_write_does_not_hide_why_the_run_ended(env, monkeypatch, caplog) -> None:
    """An exception raised inside a `finally` REPLACES the one propagating.

    A disk-full error while persisting the registry would otherwise swap the
    checkpoint or guard error that actually ended the run for a symptom — at
    exactly the moment the diagnosis is needed.
    """
    from doomnotes import cli as cli_mod
    from doomnotes.vault import VaultWriter

    def die(refs, **kwargs):
        raise RuntimeError("the real reason the run ended")

    def cannot_write(self, relative, text):
        raise OSError("No space left on device")

    monkeypatch.setattr(cli_mod, "run_batch", die)
    monkeypatch.setattr(VaultWriter, "write_text", cannot_write)

    urls = env["tmp"] / "urls.txt"
    urls.write_text("https://www.instagram.com/reel/AAAAAAAAAAA/\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="the real reason the run ended"):
        run_cli(env, "run", "--urls", str(urls), "--no-sleep")
