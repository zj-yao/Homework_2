from pathlib import Path

from PIL import Image


def write_synthetic_image_folder(root: Path, num_classes: int = 3, images_per_class: int = 4) -> Path:
    """Create a tiny ImageFolder-style dataset with deterministic RGB images."""
    data_dir = root / "flowers"
    for class_idx in range(num_classes):
        class_dir = data_dir / f"class_{class_idx:02d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_idx in range(images_per_class):
            color = (
                (class_idx * 53 + image_idx * 7) % 255,
                (class_idx * 29 + image_idx * 17) % 255,
                (class_idx * 11 + image_idx * 31) % 255,
            )
            Image.new("RGB", (36, 36), color=color).save(class_dir / f"sample_{image_idx:02d}.jpg")
    return data_dir
