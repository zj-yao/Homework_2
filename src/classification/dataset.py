from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FlowerDatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    class_names: list[str]
    split_indices: dict[str, list[int]] | None = None


def make_deterministic_splits(
    num_items: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("val_ratio and test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_items, generator=generator).tolist()
    val_count = int(round(num_items * val_ratio))
    test_count = int(round(num_items * test_ratio))
    train_count = num_items - val_count - test_count
    if train_count <= 0:
        raise ValueError("Split ratios leave no training samples")

    return {
        "train": indices[:train_count],
        "val": indices[train_count : train_count + val_count],
        "test": indices[train_count + val_count :],
    }


def build_flower_datasets(
    data_dir: str | Path,
    source: Literal["folder", "torchvision"] = "folder",
    image_size: int = 224,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    split_file: str | Path | None = None,
    download: bool = False,
) -> FlowerDatasetBundle:
    data_dir = Path(data_dir).expanduser()
    if source == "folder":
        return _build_folder_datasets(data_dir, image_size, val_ratio, test_ratio, seed, split_file)
    if source == "torchvision":
        return _build_torchvision_flowers102(data_dir, image_size, download)
    raise ValueError(f"Unsupported classification data source: {source}")


def make_dataloaders(
    bundle: FlowerDatasetBundle,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    return {
        "train": DataLoader(
            bundle.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            bundle.val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            bundle.test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


def save_split_indices(split_indices: dict[str, list[int]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split_indices, indent=2), encoding="utf-8")


def _build_folder_datasets(
    data_dir: Path,
    image_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_file: str | Path | None,
) -> FlowerDatasetBundle:
    if not data_dir.exists():
        raise FileNotFoundError(f"Folder dataset not found: {data_dir}")

    train_base = datasets.ImageFolder(data_dir, transform=_train_transform(image_size))
    val_base = datasets.ImageFolder(data_dir, transform=_eval_transform(image_size))
    test_base = datasets.ImageFolder(data_dir, transform=_eval_transform(image_size))

    split_indices = _load_or_create_splits(len(train_base), val_ratio, test_ratio, seed, split_file)
    return FlowerDatasetBundle(
        train=Subset(train_base, split_indices["train"]),
        val=Subset(val_base, split_indices["val"]),
        test=Subset(test_base, split_indices["test"]),
        class_names=list(train_base.classes),
        split_indices=split_indices,
    )


def _build_torchvision_flowers102(data_dir: Path, image_size: int, download: bool) -> FlowerDatasetBundle:
    train = datasets.Flowers102(data_dir, split="train", transform=_train_transform(image_size), download=download)
    val = datasets.Flowers102(data_dir, split="val", transform=_eval_transform(image_size), download=download)
    test = datasets.Flowers102(data_dir, split="test", transform=_eval_transform(image_size), download=download)
    class_names = [f"class_{idx:03d}" for idx in range(102)]
    return FlowerDatasetBundle(train=train, val=val, test=test, class_names=class_names)


def _load_or_create_splits(
    num_items: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_file: str | Path | None,
) -> dict[str, list[int]]:
    if split_file is None:
        return make_deterministic_splits(num_items, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    split_path = Path(split_file).expanduser()
    if split_path.exists():
        return json.loads(split_path.read_text(encoding="utf-8"))

    splits = make_deterministic_splits(num_items, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    save_split_indices(splits, split_path)
    return splits


def _train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _eval_transform(image_size: int) -> transforms.Compose:
    resize_size = max(image_size, int(round(image_size * 1.14)))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
