import json

import numpy as np
import torch
from PIL import Image

from src.segmentation.evaluate import evaluate_from_config
from src.segmentation.train import train_from_config
from src.segmentation.unet import UNet


def _write_tiny_dataset(root):
    image_dir = root / "images"
    mask_dir = root / "labels"
    split_dir = root / "splits"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()
    split_dir.mkdir()

    names = ["sample_0.png", "sample_1.png"]
    for index, name in enumerate(names):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., index % 3] = 180
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[:, 16:] = 1
        Image.fromarray(image).save(image_dir / name)
        Image.fromarray(mask).save(mask_dir / name)

    (split_dir / "train.txt").write_text("\n".join(names), encoding="utf-8")
    (split_dir / "val.txt").write_text(names[-1], encoding="utf-8")


def _config(data_root, output_dir):
    return {
        "experiment": {
            "name": "pytest_unet",
            "output_dir": str(output_dir),
            "seed": 7,
            "logger": "none",
        },
        "data": {
            "root": str(data_root),
            "image_dir": "images",
            "mask_dir": "labels",
            "train_split": "splits/train.txt",
            "val_split": "splits/val.txt",
            "num_classes": 2,
            "image_size": [32, 32],
            "normalize": False,
        },
        "model": {"in_channels": 3, "num_classes": 2, "base_channels": 4},
        "train": {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "lr": 0.001,
            "loss": "ce_dice",
            "device": "cpu",
        },
    }


def test_train_from_config_writes_history_and_best_checkpoint(tmp_path):
    data_root = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    _write_tiny_dataset(data_root)

    history = train_from_config(_config(data_root, output_dir))

    assert len(history) == 1
    assert (output_dir / "history.json").exists()
    assert (output_dir / "history.csv").exists()
    assert (output_dir / "best.pt").exists()
    saved_history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    assert saved_history[0]["epoch"] == 1
    assert "val_miou" in saved_history[0]


def test_train_from_config_can_resume_from_latest_checkpoint(tmp_path):
    data_root = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    _write_tiny_dataset(data_root)
    config = _config(data_root, output_dir)

    train_from_config(config)
    config["train"]["epochs"] = 2
    config["train"]["resume_from"] = str(output_dir / "latest.pt")

    history = train_from_config(config)

    assert [row["epoch"] for row in history] == [1, 2]
    checkpoint = torch.load(output_dir / "latest.pt", map_location="cpu")
    assert checkpoint["epoch"] == 2


def test_evaluate_from_config_loads_checkpoint_and_writes_metrics(tmp_path):
    data_root = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    _write_tiny_dataset(data_root)
    output_dir.mkdir()
    model = UNet(in_channels=3, num_classes=2, base_channels=4)
    checkpoint_path = output_dir / "model.pt"
    torch.save({"model": model.state_dict(), "epoch": 0}, checkpoint_path)

    metrics = evaluate_from_config(_config(data_root, output_dir), checkpoint_path)

    assert "miou" in metrics
    assert 0.0 <= metrics["miou"] <= 1.0
    assert (output_dir / "evaluation_metrics.json").exists()
