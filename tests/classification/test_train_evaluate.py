import json
import subprocess
import sys
from pathlib import Path

import yaml

from .conftest import write_synthetic_image_folder


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_train_and_evaluate_entrypoints_write_smoke_artifacts(tmp_path):
    data_dir = write_synthetic_image_folder(tmp_path, num_classes=3, images_per_class=3)
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 19,
                "data": {
                    "source": "folder",
                    "data_dir": str(data_dir),
                    "num_classes": 3,
                    "image_size": 32,
                    "val_ratio": 0.2,
                    "test_ratio": 0.2,
                    "num_workers": 0,
                },
                "model": {"name": "resnet18", "pretrained": False},
                "train": {
                    "epochs": 1,
                    "batch_size": 2,
                    "backbone_lr": 1e-4,
                    "head_lr": 1e-3,
                    "weight_decay": 0.0,
                    "max_train_batches": 1,
                    "max_val_batches": 1,
                    "device": "cpu",
                },
                "logging": {"backend": "none"},
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    train_result = subprocess.run(
        [sys.executable, "-m", "src.classification.train", "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert train_result.returncode == 0, train_result.stderr
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "latest.pt").exists()
    assert (output_dir / "history.json").exists()
    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    assert history[0]["epoch"] == 1
    assert "val_accuracy" in history[0]

    eval_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.classification.evaluate",
            "--config",
            str(config_path),
            "--checkpoint",
            str(output_dir / "best.pt"),
            "--split",
            "test",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert eval_result.returncode == 0, eval_result.stderr
    metrics = json.loads((output_dir / "eval_test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["split"] == "test"
    assert 0.0 <= metrics["accuracy"] <= 1.0
