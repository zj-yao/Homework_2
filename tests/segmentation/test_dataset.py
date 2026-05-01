import numpy as np
import torch
from PIL import Image

from src.segmentation.dataset import StanfordBackgroundDataset


def test_stanford_background_dataset_loads_image_mask_pair(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "labels"
    image_dir.mkdir()
    mask_dir.mkdir()
    Image.fromarray(np.full((4, 5, 3), 127, dtype=np.uint8)).save(image_dir / "scene.png")
    Image.fromarray(np.array([[0, 1, 1, 2, 2]] * 4, dtype=np.uint8)).save(
        mask_dir / "scene.png"
    )

    dataset = StanfordBackgroundDataset(
        root=tmp_path,
        split="train",
        image_size=(6, 8),
    )

    image, mask = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 6, 8)
    assert image.dtype == torch.float32
    assert mask.shape == (6, 8)
    assert mask.dtype == torch.long
    assert set(mask.unique().tolist()) == {0, 1, 2}


def test_stanford_background_dataset_split_file_accepts_image_stems(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "labels"
    split_dir = tmp_path / "splits"
    image_dir.mkdir()
    mask_dir.mkdir()
    split_dir.mkdir()
    Image.fromarray(np.full((4, 5, 3), 127, dtype=np.uint8)).save(image_dir / "scene.png")
    Image.fromarray(np.array([[0, 1, 1, 2, 2]] * 4, dtype=np.uint8)).save(
        mask_dir / "scene.png"
    )
    (split_dir / "train.txt").write_text("scene\n", encoding="utf-8")

    dataset = StanfordBackgroundDataset(root=tmp_path, split="train")

    assert len(dataset) == 1
    image, mask = dataset[0]
    assert image.shape == (3, 4, 5)
    assert mask.shape == (4, 5)


def test_stanford_background_dataset_finds_raw_regions_txt_masks(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "labels"
    split_dir = tmp_path / "splits"
    image_dir.mkdir()
    mask_dir.mkdir()
    split_dir.mkdir()
    Image.fromarray(np.full((3, 4, 3), 64, dtype=np.uint8)).save(image_dir / "9005294.jpg")
    np.savetxt(
        mask_dir / "9005294.regions.txt",
        np.array([[0, 1, -1, 2]] * 3, dtype=np.int64),
        fmt="%d",
    )
    (split_dir / "train.txt").write_text("9005294\n", encoding="utf-8")

    dataset = StanfordBackgroundDataset(root=tmp_path, split="train")

    _, mask = dataset[0]
    assert mask.shape == (3, 4)
    assert set(mask.unique().tolist()) == {-1, 0, 1, 2}


def test_stanford_background_dataset_accepts_relative_root_with_default_split(tmp_path, monkeypatch):
    root = tmp_path / "stanford"
    image_dir = root / "images"
    mask_dir = root / "labels"
    split_dir = root / "splits"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()
    split_dir.mkdir()
    Image.fromarray(np.full((3, 4, 3), 64, dtype=np.uint8)).save(image_dir / "scene.jpg")
    np.savetxt(mask_dir / "scene.regions.txt", np.array([[0, 1, 2, -1]] * 3), fmt="%d")
    (split_dir / "train.txt").write_text("scene\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    dataset = StanfordBackgroundDataset(root="stanford", split="train")

    assert len(dataset) == 1
