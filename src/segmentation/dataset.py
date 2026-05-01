from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXTENSIONS = (
    ".regions.txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".txt",
)
DEFAULT_MASK_DIRS = ("labels", "masks", "annotations", "semantic_labels")


def _pil_resampling(name: str) -> int:
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def _as_hw(image_size: int | Iterable[int] | None) -> tuple[int, int] | None:
    if image_size is None:
        return None
    if isinstance(image_size, int):
        return (image_size, image_size)
    size = tuple(int(value) for value in image_size)
    if len(size) != 2:
        raise ValueError("image_size must be an int or a (height, width) pair")
    return size


class StanfordBackgroundDataset(Dataset):
    """Image/mask pair loader for Stanford Background-style segmentation data.

    The loader supports common layouts such as ``images/<name>.jpg`` with
    ``labels/<name>.png`` or a split file containing either one image name/stem
    per line or two whitespace/comma-separated paths: ``image mask``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        split_file: str | Path | None = None,
        image_dir: str = "images",
        mask_dir: str | None = None,
        image_size: int | Iterable[int] | None = None,
        normalize: bool = False,
        mean: Iterable[float] = (0.485, 0.456, 0.406),
        std: Iterable[float] = (0.229, 0.224, 0.225),
    ):
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = _as_hw(image_size)
        self.normalize = normalize
        self.mean = torch.tensor(tuple(mean), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(tuple(std), dtype=torch.float32).view(3, 1, 1)

        if not self.root.exists():
            raise FileNotFoundError(f"Segmentation data root does not exist: {self.root}")

        self.images_root = self.root / image_dir
        if not self.images_root.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.images_root}")

        self.masks_root = self._resolve_masks_root(mask_dir)
        if split_file is None:
            candidate = self.root / "splits" / f"{split}.txt"
            split_file = candidate if candidate.exists() else None
        self.samples = (
            self._read_split_file(Path(split_file))
            if split_file is not None
            else self._scan_samples()
        )
        if not self.samples:
            raise FileNotFoundError(
                f"No image/mask pairs found under {self.root} for split '{split}'"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = self._read_mask(mask_path)

        if self.image_size is not None:
            height, width = self.image_size
            image = image.resize((width, height), _pil_resampling("BILINEAR"))
            mask_image = Image.fromarray(mask.astype(np.int32), mode="I")
            mask = np.asarray(
                mask_image.resize((width, height), _pil_resampling("NEAREST")),
                dtype=np.int64,
            )

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        if self.normalize:
            image_tensor = (image_tensor - self.mean) / self.std
        mask_tensor = torch.from_numpy(mask.astype(np.int64)).long()
        return image_tensor, mask_tensor

    def _resolve_masks_root(self, mask_dir: str | None) -> Path:
        if mask_dir is not None:
            masks_root = self.root / mask_dir
            if not masks_root.exists():
                raise FileNotFoundError(f"Mask directory does not exist: {masks_root}")
            return masks_root
        for candidate in DEFAULT_MASK_DIRS:
            masks_root = self.root / candidate
            if masks_root.exists():
                return masks_root
        raise FileNotFoundError(
            "Could not find a mask directory. Tried: "
            + ", ".join(str(self.root / name) for name in DEFAULT_MASK_DIRS)
        )

    def _read_split_file(self, split_file: Path) -> list[tuple[Path, Path]]:
        if not split_file.is_absolute():
            split_file = self.root / split_file
        if not split_file.exists():
            raise FileNotFoundError(f"Split file does not exist: {split_file}")

        samples = []
        for raw_line in split_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part for part in line.replace(",", " ").split() if part]
            if len(parts) == 1:
                image_path = self._resolve_image(parts[0])
                mask_path = self._find_mask_for_image(image_path)
            elif len(parts) == 2:
                image_path = self._resolve_path(parts[0], self.images_root)
                mask_path = self._resolve_path(parts[1], self.masks_root)
            else:
                raise ValueError(f"Invalid split line in {split_file}: {raw_line}")
            samples.append((image_path, mask_path))
        return samples

    def _scan_samples(self) -> list[tuple[Path, Path]]:
        samples = []
        for image_path in sorted(self.images_root.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = self._find_mask_for_image(image_path)
            samples.append((image_path, mask_path))
        return samples

    def _resolve_path(self, raw_path: str, default_root: Path) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            if path.exists():
                path = path
            elif (self.root / path).exists():
                path = self.root / path
            else:
                path = default_root / path
        if not path.exists():
            raise FileNotFoundError(f"Referenced path does not exist: {path}")
        return path

    def _resolve_image(self, raw_name: str) -> Path:
        raw_path = Path(raw_name)
        if not raw_path.suffix:
            search_bases = [raw_path]
            if not raw_path.is_absolute():
                search_bases.extend([self.root / raw_path, self.images_root / raw_path])
            for base in search_bases:
                for extension in IMAGE_EXTENSIONS:
                    candidate = base.with_suffix(extension)
                    if candidate.exists():
                        return candidate

        path = self._resolve_path(raw_name, self.images_root)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Image file does not exist: {path}")

    def _find_mask_for_image(self, image_path: Path) -> Path:
        rel_stem = image_path.relative_to(self.images_root).with_suffix("")
        for extension in MASK_EXTENSIONS:
            candidate = self.masks_root / rel_stem.with_suffix(extension)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find mask for image: {image_path}")

    def _read_mask(self, mask_path: Path) -> np.ndarray:
        if mask_path.suffix.lower() == ".txt":
            mask = np.loadtxt(mask_path, dtype=np.int64)
            return np.asarray(mask, dtype=np.int64)
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return np.asarray(mask, dtype=np.int64)
