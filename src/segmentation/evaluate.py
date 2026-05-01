from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .losses import build_loss
from .train import (
    _ignore_index,
    build_dataloader,
    build_model,
    load_config,
    merge_overrides,
    resolve_device,
    save_json,
    set_seed,
    evaluate_epoch,
)


def evaluate_from_config(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str = "val",
) -> dict[str, Any]:
    set_seed(int(config.get("experiment", {}).get("seed", 42)))
    device = resolve_device(config)
    num_classes = int(config.get("data", {}).get("num_classes", config.get("model", {}).get("num_classes", 8)))
    ignore_index = _ignore_index(config)
    dataloader = build_dataloader(config, split, shuffle=False)

    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)

    criterion = build_loss(
        name=str(config.get("train", {}).get("loss", "ce")),
        num_classes=num_classes,
        ignore_index=ignore_index,
        ce_weight=float(config.get("train", {}).get("ce_weight", 1.0)),
        dice_weight=float(config.get("train", {}).get("dice_weight", 1.0)),
    )
    metrics = evaluate_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    output_dir = Path(config.get("experiment", {}).get("output_dir", "outputs/segmentation/unet"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output_dir / "evaluation_metrics.json")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a U-Net segmentation checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values, e.g. data.root=/path/to/data. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_overrides(load_config(args.config), args.override)
    metrics = evaluate_from_config(config, args.checkpoint, split=args.split)
    print(f"{args.split}_miou={metrics['miou']:.4f}, {args.split}_loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
