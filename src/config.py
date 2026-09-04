"""Configuration loading and shared paths."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

HF_CACHE_DIR = ROOT / "models" / "hf_cache"


def load_config(path: Path | str = ROOT / "config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)