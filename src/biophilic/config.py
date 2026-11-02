"""Load YAML configuration and resolve repository paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


def repo_root() -> Path:
    """Repository root: $BIOPHILIC_ROOT, else the directory that contains config/pipeline.yaml."""
    env = os.environ.get("BIOPHILIC_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "pipeline.yaml").exists():
            return parent
    return Path.cwd()


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """Load config/<name>.yaml."""
    with open(repo_root() / "config" / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(key: str) -> Path:
    """Absolute path for a key under `paths:` in config/pipeline.yaml."""
    return repo_root() / load("pipeline")["paths"][key]


@dataclass(frozen=True)
class Item:
    code: str
    legacy_code: str
    dimension: str
    element: str
    definition: str
    visual_cues: str
    not_this: str


def bdia_items() -> list[Item]:
    return [Item(**it) for it in load("bdia_items")["items"]]


def bdia_dimensions() -> dict[str, str]:
    return dict(load("bdia_items")["dimensions"])


def spaces() -> list[str]:
    return list(load("bdia_items")["spaces"].keys())
