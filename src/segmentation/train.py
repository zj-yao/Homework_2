from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .dataset import StanfordBackgroundDataset
from .losses import build_loss
from .metrics import SegmentationMetricTracker
from .unet import UNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_history_csv(history: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in history for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def merge_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must use key=value format: {override}")
        dotted_key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = merged
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot set nested override through non-dict key: {key}")
        cursor[keys[-1]] = value
    return merged


class NullLogger:
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbLogger:
    def __init__(self, config: dict[str, Any]):
        import wandb

        experiment = config.get("experiment", {})
        mode = experiment.get("mode") or experiment.get("log_mode")
        self.wandb = wandb
        self.run = wandb.init(
            project=experiment.get("project", "homework2-segmentation"),
            name=experiment.get("name", "unet"),
            config=config,
            **({"mode": mode} if mode not in {None, "", "none"} else {}),
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.wandb.log(metrics, step=step)

    def finish(self) -> None:
        self.wandb.finish()


class SwanlabLogger:
    def __init__(self, config: dict[str, Any]):
        import swanlab

        experiment = config.get("experiment", {})
        mode = experiment.get("mode") or experiment.get("log_mode")
        self.swanlab = swanlab
        swanlab.init(
            project=experiment.get("project", "homework2-segmentation"),
            experiment_name=experiment.get("name", "unet"),
            config=config,
            **({"mode": mode} if mode not in {None, "", "none"} else {}),
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.swanlab.log(metrics, step=step)

    def finish(self) -> None:
        finish = getattr(self.swanlab, "finish", None)
        if finish is not None:
            finish()


def build_logger(config: dict[str, Any]):
    logger_name = str(config.get("experiment", {}).get("logger", "none")).lower()
    if logger_name in {"", "none", "disabled", "false"}:
        return NullLogger()
    if logger_name == "wandb":
        return WandbLogger(config)
    if logger_name == "swanlab":
        return SwanlabLogger(config)
    raise ValueError(f"Unsupported logger: {logger_name}")


def _ignore_index(config: dict[str, Any]) -> int | None:
    value = config.get("data", {}).get("ignore_index")
    if value in {None, "none", "None"}:
        return None
    return int(value)


def _image_size(config: dict[str, Any]) -> int | list[int] | None:
    return config.get("data", {}).get("image_size")


def build_dataset(config: dict[str, Any], split: str) -> StanfordBackgroundDataset:
    data_config = config.get("data", {})
    split_file = data_config.get(f"{split}_split")
    return StanfordBackgroundDataset(
        root=data_config["root"],
        split=split,
        split_file=split_file,
        image_dir=data_config.get("image_dir", "images"),
        mask_dir=data_config.get("mask_dir"),
        image_size=_image_size(config),
        normalize=bool(data_config.get("normalize", False)),
        mean=data_config.get("mean", (0.485, 0.456, 0.406)),
        std=data_config.get("std", (0.229, 0.224, 0.225)),
    )


def build_dataloader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    train_config = config.get("train", {})
    dataset = build_dataset(config, split)
    return DataLoader(
        dataset,
        batch_size=int(train_config.get("batch_size", 4)),
        shuffle=shuffle,
        num_workers=int(train_config.get("num_workers", 4)),
        pin_memory=bool(train_config.get("pin_memory", torch.cuda.is_available())),
    )


def build_model(config: dict[str, Any]) -> UNet:
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    return UNet(
        in_channels=int(model_config.get("in_channels", 3)),
        num_classes=int(model_config.get("num_classes", data_config.get("num_classes", 8))),
        base_channels=int(model_config.get("base_channels", 32)),
        bilinear=bool(model_config.get("bilinear", False)),
    )


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_config = config.get("train", {})
    optimizer_name = str(train_config.get("optimizer", "adamw")).lower()
    lr = float(train_config.get("lr", 1e-3))
    weight_decay = float(train_config.get("weight_decay", 1e-4))
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("train", {}).get("device", "cuda")).lower()
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    tracker = SegmentationMetricTracker(num_classes=num_classes, ignore_index=ignore_index)
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        tracker.update(logits.cpu(), masks.cpu())
    metrics = tracker.compute()
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "metrics": metrics,
    }


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list history in {path}")
    return [dict(row) for row in payload]


def train_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)
    output_dir = Path(config.get("experiment", {}).get("output_dir", "outputs/segmentation/unet"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config.json")

    device = resolve_device(config)
    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)
    num_classes = int(config.get("data", {}).get("num_classes", config.get("model", {}).get("num_classes", 8)))
    ignore_index = _ignore_index(config)

    model = build_model(config).to(device)
    criterion = build_loss(
        name=str(config.get("train", {}).get("loss", "ce")),
        num_classes=num_classes,
        ignore_index=ignore_index,
        ce_weight=float(config.get("train", {}).get("ce_weight", 1.0)),
        dice_weight=float(config.get("train", {}).get("dice_weight", 1.0)),
    )
    optimizer = build_optimizer(model, config)
    logger = build_logger(config)

    history = _load_history(output_dir / "history.json")
    best_miou = -1.0
    epochs = int(config.get("train", {}).get("epochs", 50))
    start_epoch = 1
    resume_from = config.get("train", {}).get("resume_from")
    if resume_from not in {None, "", "none", "None"}:
        checkpoint_path = Path(str(resume_from)).expanduser()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if not history:
            history = _load_history(checkpoint_path.parent / "history.json")
        history = [row for row in history if int(row.get("epoch", 0)) < start_epoch]

    if history:
        best_miou = max(float(row["val_miou"]) for row in history)
    try:
        for epoch in range(start_epoch, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_miou": val_metrics["miou"],
            }
            history.append(row)
            logger.log(row, step=epoch)

            payload = _checkpoint_payload(model, optimizer, epoch, config, row)
            torch.save(payload, output_dir / "latest.pt")
            if row["val_miou"] >= best_miou:
                best_miou = float(row["val_miou"])
                torch.save(payload, output_dir / "best.pt")

            save_json(history, output_dir / "history.json")
            write_history_csv(history, output_dir / "history.csv")
    finally:
        logger.finish()
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch U-Net for segmentation.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values, e.g. train.epochs=10. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_overrides(load_config(args.config), args.override)
    history = train_from_config(config)
    if history:
        last = history[-1]
        print(
            f"Finished {len(history)} epochs. "
            f"val_miou={last['val_miou']:.4f}, val_loss={last['val_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
