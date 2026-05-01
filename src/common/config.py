from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _expand_paths(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("~"):
        return str(Path(value).expanduser())
    if isinstance(value, dict):
        return {key: _expand_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_paths(item) for item in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and expand user-home paths."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return _expand_paths(data)


def merge_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Return a copy of config with `key.path=value` command-line overrides."""
    merged = deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must use key=value format: {override}")
        dotted_key, raw_value = override.split("=", 1)
        keys = [part for part in dotted_key.split(".") if part]
        if not keys:
            raise ValueError(f"Override key is empty: {override}")

        cursor: dict[str, Any] = merged
        for key in keys[:-1]:
            next_value = cursor.setdefault(key, {})
            if not isinstance(next_value, dict):
                raise ValueError(f"Cannot set nested key below non-mapping value: {key}")
            cursor = next_value
        cursor[keys[-1]] = yaml.safe_load(raw_value)
    return _expand_paths(merged)
