from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from src.classification.config import load_config, merge_overrides
from src.classification.dataset import build_flower_datasets, make_dataloaders, save_split_indices
from src.classification.models import build_model, create_param_groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a Flower102 classification model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--override", action="append", default=[], help="Override config values, e.g. train.epochs=5")
    args = parser.parse_args(argv)

    config = merge_overrides(load_config(args.config), args.override)
    train(config)
    return 0


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    output_dir = Path(config.get("output_dir", "outputs/classification/run")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = config.get("data", {})
    bundle = build_flower_datasets(
        data_dir=data_config.get("data_dir", "data/flower102"),
        source=data_config.get("source", "folder"),
        image_size=int(data_config.get("image_size", 224)),
        val_ratio=float(data_config.get("val_ratio", 0.1)),
        test_ratio=float(data_config.get("test_ratio", 0.1)),
        seed=seed,
        split_file=data_config.get("split_file"),
        download=bool(data_config.get("download", False)),
    )

    train_config = config.get("train", {})
    device = resolve_device(train_config.get("device", "auto"))
    loaders = make_dataloaders(
        bundle,
        batch_size=int(train_config.get("batch_size", 32)),
        num_workers=int(data_config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    num_classes = int(data_config.get("num_classes") or len(bundle.class_names))
    model_config = config.get("model", {})
    model_name = model_config.get("name", "resnet18")
    model = build_model(model_name, num_classes=num_classes, pretrained=bool(model_config.get("pretrained", True)))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        create_param_groups(
            model,
            backbone_lr=float(train_config.get("backbone_lr", 1e-4)),
            head_lr=float(train_config.get("head_lr", 1e-3)),
        ),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )

    logger = OptionalExperimentLogger(config)
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = 0
    epochs = int(train_config.get("epochs", 10))

    _write_json(output_dir / "class_names.json", bundle.class_names)
    _write_json(output_dir / "config_resolved.json", config)
    if bundle.split_indices:
        save_split_indices(bundle.split_indices, output_dir / "split_indices.json")

    for epoch in range(1, epochs + 1):
        train_metrics = run_one_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            max_batches=train_config.get("max_train_batches"),
            desc=f"train {epoch}/{epochs}",
        )
        val_metrics = run_one_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            optimizer=None,
            max_batches=train_config.get("max_val_batches"),
            desc=f"val {epoch}/{epochs}",
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        logger.log(row, step=epoch)

        checkpoint = {
            "epoch": epoch,
            "model_name": model_name,
            "num_classes": num_classes,
            "class_names": bundle.class_names,
            "config": config,
            "model_state": model.state_dict(),
            "val_accuracy": val_metrics["accuracy"],
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if val_metrics["accuracy"] >= best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(checkpoint, output_dir / "best.pt")

        _write_json(output_dir / "history.json", history)
        _write_history_csv(history, output_dir / "history.csv")

    summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "final_val_accuracy": history[-1]["val_accuracy"] if history else None,
        "epochs": epochs,
    }
    _write_json(output_dir / "metrics.json", summary)
    logger.finish()
    return summary


def run_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | str | None = None,
    desc: str | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    max_batches_int = None if max_batches in (None, "") else int(max_batches)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress = tqdm(loader, desc=desc, leave=False, disable=max_batches_int == 1)
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (images, labels) in enumerate(progress):
            if max_batches_int is not None and batch_idx >= max_batches_int:
                break
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_samples += batch_size

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "num_samples": 0.0}
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "num_samples": float(total_samples),
    }


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class OptionalExperimentLogger:
    def __init__(self, config: dict[str, Any]) -> None:
        logging_config = config.get("logging", {})
        self.backend = str(logging_config.get("backend", "none")).lower()
        self.run = None
        if self.backend in {"none", "disabled", "off"}:
            return
        mode = logging_config.get("mode")
        init_options = {"mode": mode} if mode not in {None, "", "none"} else {}
        if self.backend == "wandb":
            import wandb

            self.run = wandb.init(
                project=logging_config.get("project", "homework2-classification"),
                name=logging_config.get("run_name"),
                config=config,
                **init_options,
            )
        elif self.backend == "swanlab":
            import swanlab

            self.run = swanlab.init(
                project=logging_config.get("project", "homework2-classification"),
                experiment_name=logging_config.get("run_name"),
                config=config,
                **init_options,
            )
        else:
            raise ValueError(f"Unsupported logging backend: {self.backend}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self.run is None:
            return
        if self.backend == "wandb":
            self.run.log(metrics, step=step)
        else:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is None:
            return
        if self.backend == "wandb":
            self.run.finish()
        else:
            self.run.finish()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in history for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    raise SystemExit(main())
