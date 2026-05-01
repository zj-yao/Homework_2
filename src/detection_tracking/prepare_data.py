from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class YoloDatasetConfig:
    """Validated Ultralytics YOLO dataset metadata."""

    train: Path
    val: Path
    names: list[str]
    test: Path | None = None
    path: Path | None = None

    @property
    def num_classes(self) -> int:
        return len(self.names)

    def to_ultralytics_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "train": str(self.train),
            "val": str(self.val),
            "nc": self.num_classes,
            "names": self.names,
        }
        if self.test is not None:
            payload["test"] = str(self.test)
        if self.path is not None:
            payload["path"] = str(self.path)
        return payload


def _resolve_path(value: str | Path, root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if root is not None and not path.is_absolute():
        path = root / path
    return path


def _normalise_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, Mapping):
        items = sorted(raw_names.items(), key=lambda item: int(item[0]))
        names = [str(value).strip() for _, value in items]
    elif isinstance(raw_names, (list, tuple)):
        names = [str(value).strip() for value in raw_names]
    else:
        raise ValueError("names must be a non-empty list or index-to-name mapping")

    if not names or any(not name for name in names):
        raise ValueError("names must contain at least one non-empty class name")
    if len(names) != len(set(names)):
        raise ValueError("names must be unique")
    return names


def _require_existing_path(key: str, value: Path) -> Path:
    if not value.exists():
        raise ValueError(f"{key} path does not exist: {value}")
    return value


def validate_yolo_dataset(config: Mapping[str, Any] | YoloDatasetConfig) -> YoloDatasetConfig:
    """Validate and normalise an Ultralytics dataset YAML mapping."""

    if isinstance(config, YoloDatasetConfig):
        data = config.to_ultralytics_dict()
    else:
        data = dict(config)

    for required_key in ("train", "val", "names"):
        if required_key not in data:
            raise ValueError(f"{required_key} is required in YOLO dataset config")

    root = _resolve_path(data["path"]) if data.get("path") else None
    train = _require_existing_path("train", _resolve_path(data["train"], root))
    val = _require_existing_path("val", _resolve_path(data["val"], root))
    test = _resolve_path(data["test"], root) if data.get("test") else None
    if test is not None:
        _require_existing_path("test", test)

    names = _normalise_names(data.get("names"))
    if "nc" in data and int(data["nc"]) != len(names):
        raise ValueError(f"nc={data['nc']} does not match number of names={len(names)}")

    return YoloDatasetConfig(train=train, val=val, test=test, names=names, path=root)


def load_yolo_data_yaml(path: str | Path) -> YoloDatasetConfig:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected mapping in YOLO data YAML: {path}")
    return validate_yolo_dataset(data)


def write_yolo_data_yaml(config: YoloDatasetConfig, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config.to_ultralytics_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or rewrite a YOLO dataset YAML.")
    parser.add_argument("--input", required=True, help="Path to an existing YOLO data YAML")
    parser.add_argument("--output", help="Optional path for a normalized YAML copy")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_yolo_data_yaml(args.input)
    if args.output:
        write_yolo_data_yaml(config, args.output)
    print(f"Validated YOLO dataset: {config.num_classes} classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
