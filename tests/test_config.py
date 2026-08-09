"""Config loading, and the shipped config.toml itself.

The second half matters more than the first. `config.toml` is documentation
that the code reads, so a key renamed in one place and not the other fails
silently: `cfg.get(...)` returns its default and the run proceeds with a value
nobody chose. These tests pin the real file against the real call sites.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from doomnotes import config as config_mod
from doomnotes.config import REPO_ROOT, load
from doomnotes.summarize import settings_from_config as summarize_settings
from doomnotes.transcribe import settings_from_config as transcribe_settings


@pytest.fixture()
def cfg():
    return load()


def write_config(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    return p


# ── the loader ───────────────────────────────────────────────────────────


def test_get_returns_default_for_a_missing_key(tmp_path: Path) -> None:
    cfg = load(write_config(tmp_path, "[a]\nb = 1\n"))
    assert cfg.get("a", "b") == 1
    assert cfg.get("a", "nope", default="fallback") == "fallback"
    assert cfg.get("nope", "nope", default=None) is None


def test_relative_paths_resolve_against_the_repo_not_the_cwd(tmp_path: Path) -> None:
    """`doomnotes run` from anywhere must use the same state.db.

    Resolving against the cwd would silently create a second, empty store the
    first time the command is run from another directory — and an empty store
    means every video looks unprocessed.
    """
    cfg = load(write_config(tmp_path, '[paths]\nstate_db = "data/state.db"\n'))
    before = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert cfg.path("paths", "state_db") == REPO_ROOT / "data" / "state.db"
    finally:
        os.chdir(before)


def test_path_expands_a_home_relative_value(tmp_path: Path) -> None:
    cfg = load(write_config(tmp_path, '[vault]\npath = "~/Documents/ai-notes-vault"\n'))
    assert cfg.vault_root == Path("~/Documents/ai-notes-vault").expanduser()


def test_absolute_paths_are_left_alone(tmp_path: Path) -> None:
    cfg = load(write_config(tmp_path, f'[paths]\naudio_dir = "{tmp_path}/audio"\n'))
    assert cfg.path("paths", "audio_dir") == tmp_path / "audio"


def test_a_missing_required_path_raises_rather_than_defaulting(tmp_path: Path) -> None:
    """A typo'd path key must be loud. Silently defaulting writes somewhere else."""
    cfg = load(write_config(tmp_path, "[paths]\n"))
    with pytest.raises(KeyError):
        cfg.path("paths", "state_db")


# ── the shipped config.toml ──────────────────────────────────────────────


def test_the_repo_config_loads(cfg) -> None:
    assert cfg.source_path == config_mod.DEFAULT_CONFIG


@pytest.mark.parametrize(
    "keys",
    [
        ("vault", "path"),
        ("vault", "transcripts_dir"),
        ("paths", "state_db"),
        ("paths", "audio_dir"),
        ("pacing", "batch_cap"),
        ("pacing", "sleep_min_s"),
        ("pacing", "sleep_max_s"),
        ("pacing", "queue_order"),
        ("download", "timeout_s"),
        ("download", "audio_format"),
        ("transcribe", "model"),
        ("transcribe", "min_transcript_chars"),
        ("summarize", "model"),
        ("summarize", "host"),
        ("summarize", "think"),
        ("tags", "registry_file"),
        ("tags", "max_tags_per_note"),
        ("tags", "split_threshold"),
        ("tags", "merge_similarity"),
    ],
)
def test_every_key_the_code_reads_exists_in_the_shipped_config(cfg, keys) -> None:
    """A default that silently fires is a value nobody chose."""
    assert cfg.get(*keys) is not None, f"config.toml has no {'.'.join(keys)}"


def _config_reads_in_source() -> set[tuple[str, ...]]:
    """Every `cfg.get("a", "b", ...)` / `cfg.path("a", "b")` in the package.

    Reads are all literal string keys today, so a static scan is exact rather
    than approximate. If a dynamic read is ever introduced this will start
    under-reporting, which fails loudly rather than silently.
    """
    import ast

    found: set[tuple[str, ...]] = set()
    for py in sorted((REPO_ROOT / "src" / "doomnotes").rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"get", "path"}:
                continue
            keys = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if len(keys) >= 2:
                found.add(tuple(keys))
    return found


def _config_keys_on_disk(cfg) -> set[tuple[str, str]]:
    return {
        (section, key)
        for section, body in cfg.raw.items()
        if isinstance(body, dict)
        for key, value in body.items()
        if not isinstance(value, dict)  # [auth.instagram] is read as a whole table
    }


# Keys deliberately present but not yet read. Each needs a reason, and the
# reason has to be a decision that has not been made — not "we'll get to it".
# Anything not listed here that nothing reads fails the check below.
PENDING_CONFIG_KEYS: dict[tuple[str, str], str] = {
    ("download", "max_attempts"): (
        "no retry loop exists to configure. Arrives with the failure taxonomy "
        "(3.2) and the store's retry policy (2.1), which interlock."
    ),
}


def test_no_config_key_is_read_by_nothing(cfg) -> None:
    """The drift check that matters, in the direction that catches it.

    Asserting "the code's keys exist in config.toml" catches a rename. It
    cannot catch a key that is *only* in config.toml, which is the worse
    failure: the value looks configured, is edited with intent, and is inert.
    `audio_format = "m4a"` sat there doing nothing, because the pipeline called
    download() positionally and its function defaults won.

    The escape hatch is an explicit list with a reason per key, so switching
    this check off requires saying why in writing.
    """
    on_disk = _config_keys_on_disk(cfg)
    read = {k[:2] for k in _config_reads_in_source()}
    dead = sorted(on_disk - read - set(PENDING_CONFIG_KEYS))
    assert not dead, f"config.toml declares keys nothing reads: {dead}"


def test_pending_config_keys_are_still_actually_pending(cfg) -> None:
    """The allowlist must not outlive the reason for it.

    Once a key IS read, leaving it listed here silently exempts it from the
    check forever.
    """
    read = {k[:2] for k in _config_reads_in_source()}
    stale = sorted(set(PENDING_CONFIG_KEYS) & read)
    assert not stale, f"now read, so remove from PENDING_CONFIG_KEYS: {stale}"

    on_disk = _config_keys_on_disk(cfg)
    missing = sorted(set(PENDING_CONFIG_KEYS) - on_disk)
    assert not missing, f"listed as pending but gone from config.toml: {missing}"


def test_settings_builders_agree_with_the_shipped_config(cfg) -> None:
    s = summarize_settings(cfg)
    assert s.model == cfg.get("summarize", "model")
    assert s.max_tags == cfg.get("tags", "max_tags_per_note")

    t = transcribe_settings(cfg)
    assert t.model == cfg.get("transcribe", "model")
    assert t.min_chars == cfg.get("transcribe", "min_transcript_chars")


def test_the_configured_vault_is_not_the_main_vault(cfg) -> None:
    """Belt and braces. The guard enforces this too, but a config that even
    points at iCloud is a mistake worth catching in the suite."""
    root = str(cfg.vault_root)
    assert "Mobile Documents" not in root
    assert "CloudStorage" not in root


def test_no_cookie_value_is_stored_in_the_config(cfg) -> None:
    """Tier-1 #1: cookies are referenced by PATH, never by value.

    A `sessionid` in config.toml would be a bearer credential in a git repo.
    """
    text = config_mod.DEFAULT_CONFIG.read_text(encoding="utf-8")
    for marker in ("sessionid", "csrftoken", "ds_user_id", "sid_tt"):
        assert marker not in text

    for platform in ("instagram", "tiktok"):
        auth = cfg.get("auth", platform, default={})
        assert set(auth) <= {"mode", "browser", "cookies_file"}
        assert str(auth.get("cookies_file", "")).startswith("~/.config/")


def test_cookie_paths_live_outside_the_repo_and_outside_synced_dirs(cfg) -> None:
    for platform in ("instagram", "tiktok"):
        p = Path(str(cfg.get("auth", platform, "cookies_file"))).expanduser()
        assert not str(p).startswith(str(REPO_ROOT)), "a cookie inside the repo is one commit away"
        for synced in ("Mobile Documents", "CloudStorage", "Dropbox", "Google Drive"):
            assert synced not in str(p)


def test_pacing_window_is_ordered_and_non_instant(cfg) -> None:
    """Not a judgement on the numbers — #3.3 is yours. Just that they are sane."""
    lo = float(cfg.get("pacing", "sleep_min_s"))
    hi = float(cfg.get("pacing", "sleep_max_s"))
    assert 0 < lo <= hi


def test_queue_order_names_only_known_platforms(cfg) -> None:
    """A typo here drops a whole platform from every run, silently.

    Tolerant of the key disappearing: HANDS-ON #3.3 may replace a flat drain
    order with an interleave, and this test must not be the reason that change
    fails.
    """
    order = cfg.get("pacing", "queue_order", default=None)
    if order is None:
        pytest.skip("no queue_order — pacing no longer drains queues in a fixed order")
    assert set(order) <= {"instagram", "tiktok"}
