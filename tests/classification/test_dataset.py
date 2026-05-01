import torch

from src.classification.dataset import build_flower_datasets, make_deterministic_splits

from .conftest import write_synthetic_image_folder


def test_make_deterministic_splits_is_reproducible_and_exhaustive():
    first = make_deterministic_splits(20, val_ratio=0.2, test_ratio=0.2, seed=123)
    second = make_deterministic_splits(20, val_ratio=0.2, test_ratio=0.2, seed=123)
    different_seed = make_deterministic_splits(20, val_ratio=0.2, test_ratio=0.2, seed=456)

    assert first == second
    assert first != different_seed
    assert {name: len(values) for name, values in first.items()} == {"train": 12, "val": 4, "test": 4}

    all_indices = first["train"] + first["val"] + first["test"]
    assert sorted(all_indices) == list(range(20))
    assert len(set(all_indices)) == 20


def test_build_flower_datasets_loads_folder_data_with_transforms(tmp_path):
    data_dir = write_synthetic_image_folder(tmp_path, num_classes=3, images_per_class=4)

    bundle = build_flower_datasets(
        data_dir=data_dir,
        source="folder",
        image_size=32,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=7,
    )

    assert bundle.class_names == ["class_00", "class_01", "class_02"]
    assert len(bundle.train) == 6
    assert len(bundle.val) == 3
    assert len(bundle.test) == 3

    image, label = bundle.train[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)
    assert 0 <= label < 3
