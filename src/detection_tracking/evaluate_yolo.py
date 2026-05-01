from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_yolo(
    model_path: str | Path,
    data_yaml: str | Path,
    imgsz: int = 640,
    batch: int = 8,
    device: str | None = None,
    split: str = "val",
) -> Any:
    """Run Ultralytics validation for a trained detector."""

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    kwargs: dict[str, Any] = {
        "data": str(Path(data_yaml).expanduser()),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "split": split,
    }
    if device is not None:
        kwargs["device"] = device
    return model.val(**kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 vehicle detector.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--split", default="val")
    return parser


def summarize_metrics(metrics: Any) -> dict[str, float]:
    """Extract the compact validation metrics that matter for reports."""

    results = getattr(metrics, "results_dict", {})
    return {str(key): float(value) for key, value in results.items()}


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metrics = evaluate_yolo(
        model_path=args.model,
        data_yaml=args.data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
    )
    print(json.dumps(summarize_metrics(metrics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
