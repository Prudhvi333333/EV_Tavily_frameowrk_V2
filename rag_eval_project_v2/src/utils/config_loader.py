from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    cfg_path = Path(config_path).resolve()
    project_root = cfg_path.parent.parent
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    best_cfg_path = project_root / "config" / "best_config.yaml"
    if best_cfg_path.exists():
        with best_cfg_path.open("r", encoding="utf-8") as f:
            best_cfg = yaml.safe_load(f) or {}
        config = _deep_merge(config, best_cfg)

    config["_meta"] = {
        "config_path": str(cfg_path),
        "project_root": str(project_root),
    }
    _ensure_directories(config)
    return config


def resolve_path(config: dict[str, Any], relative_or_abs: str) -> Path:
    p = Path(relative_or_abs)
    if p.is_absolute():
        return p
    root = Path(config["_meta"]["project_root"])
    return (root / p).resolve()


def _ensure_directories(config: dict[str, Any]) -> None:
    path_keys = [
        "web_cache_dir",
        "chroma_dir",
        "reports_dir",
        "progress_dir",
        "logs_dir",
    ]
    for key in path_keys:
        p = resolve_path(config, config["paths"][key])
        p.mkdir(parents=True, exist_ok=True)

