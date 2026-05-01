from pathlib import Path

import pytest
import torch
import yaml

from src.common.config import load_config, merge_overrides
from src.common.io import ensure_dir, save_json, write_history_csv
from src.common.seed import seed_everything


def test_seed_everything_makes_torch_random_reproducible():
    seed_everything(123)
    first = torch.randn(4)

    seed_everything(123)
    second = torch.randn(4)

    assert torch.equal(first, second)


def test_load_config_reads_yaml_and_expands_user(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"data_dir": "~/dataset", "train": {"epochs": 3}}),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["data_dir"] == str(Path("~/dataset").expanduser())
    assert config["train"]["epochs"] == 3


def test_merge_overrides_sets_nested_values_without_mutating_original():
    original = {"train": {"lr": 0.1}, "name": "base"}

    merged = merge_overrides(original, ["train.lr=0.001", "train.epochs=5", "name=run"])

    assert original == {"train": {"lr": 0.1}, "name": "base"}
    assert merged["train"]["lr"] == pytest.approx(0.001)
    assert merged["train"]["epochs"] == 5
    assert merged["name"] == "run"


def test_io_helpers_create_parent_dirs_and_save_outputs(tmp_path):
    json_path = tmp_path / "nested" / "metrics.json"
    csv_path = tmp_path / "nested" / "history.csv"

    ensure_dir(json_path.parent)
    save_json({"accuracy": 0.9}, json_path)
    write_history_csv(
        [
            {"epoch": 1, "loss": 2.0, "accuracy": 0.4},
            {"epoch": 2, "loss": 1.2, "accuracy": 0.7},
        ],
        csv_path,
    )

    assert json_path.exists()
    assert '"accuracy": 0.9' in json_path.read_text(encoding="utf-8")
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "accuracy,epoch,loss"
