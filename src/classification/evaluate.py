from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.classification.config import load_config, merge_overrides
from src.classification.dataset import build_flower_datasets, make_dataloaders
from src.classification.models import build_model
from src.classification.train import resolve_device, run_one_epoch, seed_everything


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Flower102 classification checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON path.")
    parser.add_argument("--override", action="append", default=[], help="Override config values, e.g. data.image_size=224")
    args = parser.parse_args(argv)

    config = merge_overrides(load_config(args.config), args.override)
    evaluate(config, checkpoint_path=args.checkpoint, split=args.split, output_path=args.output)
    return 0


def evaluate(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str = "test",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    seed_everything(int(config.get("seed", 42)))
    train_config = config.get("train", {})
    data_config = config.get("data", {})
    device = resolve_device(train_config.get("device", "auto"))

    bundle = build_flower_datasets(
        data_dir=data_config.get("data_dir", "data/flower102"),
        source=data_config.get("source", "folder"),
        image_size=int(data_config.get("image_size", 224)),
        val_ratio=float(data_config.get("val_ratio", 0.1)),
        test_ratio=float(data_config.get("test_ratio", 0.1)),
        seed=int(config.get("seed", 42)),
        split_file=data_config.get("split_file"),
        download=bool(data_config.get("download", False)),
    )
    loaders = make_dataloaders(
        bundle,
        batch_size=int(train_config.get("batch_size", 32)),
        num_workers=int(data_config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=device, weights_only=False)
    model_name = checkpoint.get("model_name", config.get("model", {}).get("name", "resnet18"))
    num_classes = int(checkpoint.get("num_classes", data_config.get("num_classes") or len(bundle.class_names)))
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    metrics = run_one_epoch(
        model,
        loaders[split],
        nn.CrossEntropyLoss(),
        device,
        optimizer=None,
        max_batches=train_config.get("max_eval_batches"),
        desc=f"eval {split}",
    )
    result = {
        "split": split,
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "num_samples": metrics["num_samples"],
        "checkpoint": str(Path(checkpoint_path).expanduser()),
    }

    metrics_path = Path(output_path) if output_path else Path(config.get("output_dir", "outputs/classification/run")) / f"eval_{split}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
