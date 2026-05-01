from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .prepare_data import load_yolo_data_yaml


def _optional_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser())


def build_train_kwargs(
    data_yaml: str | Path,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    project: str | Path = "outputs/detection_tracking/yolov8_vehicle",
    name: str = "train",
    device: str | None = None,
    workers: int = 8,
    seed: int = 42,
    patience: int | None = 20,
    cache: bool = False,
    amp: bool = True,
    exist_ok: bool = True,
) -> dict[str, Any]:
    """Build Ultralytics train kwargs without importing or starting YOLO."""

    kwargs: dict[str, Any] = {
        "data": str(Path(data_yaml).expanduser()),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "project": str(Path(project).expanduser()),
        "name": name,
        "workers": int(workers),
        "seed": int(seed),
        "cache": bool(cache),
        "amp": bool(amp),
        "exist_ok": bool(exist_ok),
    }
    if device is not None:
        kwargs["device"] = device
    if patience is not None:
        kwargs["patience"] = int(patience)
    return kwargs


def train_yolo(model: str | Path, train_kwargs: dict[str, Any]) -> Any:
    """Run YOLOv8 fine-tuning. Imports Ultralytics only when called."""

    from ultralytics import YOLO

    detector = YOLO(str(model))
    return detector.train(**train_kwargs)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on road vehicle data.")
    parser.add_argument("--config", default="configs/detection/yolov8_vehicle.yaml")
    parser.add_argument("--model", help="YOLO model checkpoint, e.g. yolov8n.pt")
    parser.add_argument("--data-yaml", help="Ultralytics data YAML")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-data-validation", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _load_config(args.config)

    model = args.model or config.get("model", "yolov8n.pt")
    data_yaml = args.data_yaml or config.get("data_yaml")
    if not data_yaml:
        raise ValueError("data_yaml must be provided in config or --data-yaml")

    if not args.skip_data_validation:
        load_yolo_data_yaml(data_yaml)

    train_config = dict(config.get("train", {}))
    for key in ["epochs", "imgsz", "batch", "project", "name", "device", "workers", "seed", "patience"]:
        value = getattr(args, key)
        if value is not None:
            train_config[key] = value
    if args.cache:
        train_config["cache"] = True
    if args.no_amp:
        train_config["amp"] = False

    train_kwargs = build_train_kwargs(data_yaml=data_yaml, **train_config)
    train_yolo(model=model, train_kwargs=train_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
