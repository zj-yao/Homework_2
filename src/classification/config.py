from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return _expand_paths(config)


def merge_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must use key=value syntax: {override}")
        dotted_key, raw_value = override.split("=", 1)
        target = merged
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Cannot set nested override below non-dict key: {dotted_key}")
        target[parts[-1]] = yaml.safe_load(raw_value)
    return _expand_paths(merged)


def _expand_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_paths(item) for item in value]
    if isinstance(value, str) and (value.startswith("~/") or value == "~"):
        return str(Path(value).expanduser())
    return value
