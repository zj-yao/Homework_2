from pathlib import Path

import pytest
import yaml

from src.detection_tracking.prepare_data import (
    YoloDatasetConfig,
    validate_yolo_dataset,
    write_yolo_data_yaml,
)


def test_validate_yolo_dataset_accepts_existing_paths_and_names(tmp_path):
    train_dir = tmp_path / "images" / "train"
    val_dir = tmp_path / "images" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    config = validate_yolo_dataset(
        {"train": train_dir, "val": val_dir, "names": ["car", "bus", "truck"]}
    )

    assert isinstance(config, YoloDatasetConfig)
    assert config.train == train_dir
    assert config.val == val_dir
    assert config.num_classes == 3
    assert config.to_ultralytics_dict()["nc"] == 3


def test_validate_yolo_dataset_rejects_missing_paths_and_empty_names(tmp_path):
    train_dir = tmp_path / "images" / "train"
    train_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="train"):
        validate_yolo_dataset({"val": train_dir, "names": ["car"]})

    with pytest.raises(ValueError, match="val"):
        validate_yolo_dataset({"train": train_dir, "val": tmp_path / "missing", "names": ["car"]})

    with pytest.raises(ValueError, match="names"):
        validate_yolo_dataset({"train": train_dir, "val": train_dir})

    with pytest.raises(ValueError, match="names"):
        validate_yolo_dataset({"train": train_dir, "val": train_dir, "names": []})


def test_write_yolo_data_yaml_writes_ultralytics_compatible_file(tmp_path):
    train_dir = tmp_path / "images" / "train"
    val_dir = tmp_path / "images" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    config = validate_yolo_dataset({"train": train_dir, "val": val_dir, "names": ["car", "bus"]})

    output_path = write_yolo_data_yaml(config, tmp_path / "road_vehicle.yaml")

    loaded = yaml.safe_load(Path(output_path).read_text(encoding="utf-8"))
    assert loaded == {
        "train": str(train_dir),
        "val": str(val_dir),
        "nc": 2,
        "names": ["car", "bus"],
    }
