"""Config loading. Thin on purpose — config.toml is the documentation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.toml"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    source_path: Path

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, *keys: str, default: Any = None) -> Path:
        value = self.get(*keys, default=default)
        if value is None:
            raise KeyError("/".join(keys))
        p = Path(str(value)).expanduser()
        # Relative paths resolve against the repo, not the caller's cwd, so the
        # CLI behaves the same regardless of where it is invoked from.
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def vault_root(self) -> Path:
        return self.path("vault", "path")


def load(path: str | Path | None = None) -> Config:
    p = Path(path).expanduser() if path else DEFAULT_CONFIG
    with open(p, "rb") as fh:
        return Config(raw=tomllib.load(fh), source_path=p)
