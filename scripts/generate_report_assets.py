from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "tex_report" / "figures"


def read_history(path: str | Path) -> list[dict[str, float]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def read_yolo_results(path: str | Path) -> list[dict[str, float]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            normalized = {key.strip(): float(value) for key, value in row.items()}
            rows.append(normalized)
        return rows


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_classification() -> None:
    experiments = {
        "ResNet18 pretrained": ROOT / "outputs/classification/resnet18_pretrained_baseline/history.csv",
        "ResNet18 random init": ROOT / "outputs/classification/resnet18_random_init/history.csv",
        "ResNet18 + SE": ROOT / "outputs/classification/resnet18_se_block/history.csv",
        "ResNet34 pretrained": ROOT / "outputs/enhanced/classification/resnet34_pretrained_60ep/history.csv",
    }

    plt.figure(figsize=(8.5, 5.0))
    for label, path in experiments.items():
        rows = read_history(path)
        plt.plot([r["epoch"] for r in rows], [r["val_accuracy"] for r in rows], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Flower102 Classification Validation Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(FIG_DIR / "classification_val_accuracy.png")

    plt.figure(figsize=(8.5, 5.0))
    for label, path in experiments.items():
        rows = read_history(path)
        plt.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Flower102 Classification Validation Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(FIG_DIR / "classification_val_loss.png")


def plot_segmentation() -> None:
    experiments = {
        "CE": ROOT / "outputs/segmentation/unet_ce/history.csv",
        "Dice": ROOT / "outputs/segmentation/unet_dice/history.csv",
        "CE + Dice": ROOT / "outputs/segmentation/unet_ce_dice/history.csv",
        "U-Net64 CE + Dice": ROOT / "outputs/enhanced/segmentation/unet64_ce_dice_100ep/history.csv",
    }

    plt.figure(figsize=(8.5, 5.0))
    for label, path in experiments.items():
        rows = read_history(path)
        plt.plot([r["epoch"] for r in rows], [r["val_miou"] for r in rows], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation mIoU")
    plt.title("Stanford Background Segmentation mIoU")
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(FIG_DIR / "segmentation_val_miou.png")

    plt.figure(figsize=(8.5, 5.0))
    for label, path in experiments.items():
        rows = read_history(path)
        plt.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Stanford Background Segmentation Validation Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(FIG_DIR / "segmentation_val_loss.png")


def plot_detection() -> None:
    experiments = {
        "YOLOv8n": ROOT / "outputs/detection_tracking/yolov8_vehicle/yolov8n_vehicle/results.csv",
        "YOLOv8s": ROOT / "outputs/enhanced/detection_tracking/yolov8s_vehicle/yolov8s_vehicle_100ep/results.csv",
    }

    plt.figure(figsize=(8.5, 5.0))
    for label, path in experiments.items():
        rows = read_yolo_results(path)
        plt.plot([r["epoch"] for r in rows], [r["metrics/mAP50(B)"] for r in rows], label=f"{label} mAP50")
    plt.xlabel("Epoch")
    plt.ylabel("mAP50")
    plt.title("Road Vehicle Detection mAP50")
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(FIG_DIR / "detection_map50.png")

    plt.figure(figsize=(8.5, 5.0))
    for label, path in experiments.items():
        rows = read_yolo_results(path)
        plt.plot([r["epoch"] for r in rows], [r["metrics/mAP50-95(B)"] for r in rows], label=f"{label} mAP50-95")
    plt.xlabel("Epoch")
    plt.ylabel("mAP50-95")
    plt.title("Road Vehicle Detection mAP50-95")
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(FIG_DIR / "detection_map50_95.png")


def copy_figures() -> None:
    copies = {
        ROOT / "outputs/enhanced/detection_tracking/yolov8s_vehicle/yolov8s_vehicle_100ep/results.png": FIG_DIR / "yolov8s_results.png",
        ROOT / "outputs/detection_tracking/tracked_video/campus_road_01_yolov8s_tracking_contact_sheet.jpg": FIG_DIR / "tracking_contact_sheet.jpg",
        ROOT / "outputs/detection_tracking/occlusion_frames/campus_road_01/occlusion_contact_sheet.jpg": FIG_DIR / "occlusion_contact_sheet.jpg",
        ROOT / "outputs/detection_tracking/occlusion_frames/campus_road_01/occlusion_000536.jpg": FIG_DIR / "occlusion_frame_536.jpg",
        ROOT / "outputs/detection_tracking/tracked_video/campus_road_01_yolov8s_summary.json": FIG_DIR / "tracking_summary.json",
    }
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for source, target in copies.items():
        if source.exists():
            shutil.copy2(source, target)


def write_result_tables_json() -> None:
    payload = {
        "classification": {
            "resnet18_pretrained": json.loads((ROOT / "outputs/classification/resnet18_pretrained_baseline/eval_test_metrics.json").read_text()),
            "resnet18_random": json.loads((ROOT / "outputs/classification/resnet18_random_init/eval_test_metrics.json").read_text()),
            "resnet18_se": json.loads((ROOT / "outputs/classification/resnet18_se_block/eval_test_metrics.json").read_text()),
            "resnet34_pretrained": json.loads((ROOT / "outputs/enhanced/classification/resnet34_pretrained_60ep/eval_test_metrics.json").read_text()),
        },
        "segmentation": {
            "unet_ce": json.loads((ROOT / "outputs/segmentation/unet_ce/evaluation_metrics.json").read_text()),
            "unet_dice": json.loads((ROOT / "outputs/segmentation/unet_dice/evaluation_metrics.json").read_text()),
            "unet_ce_dice": json.loads((ROOT / "outputs/segmentation/unet_ce_dice/evaluation_metrics.json").read_text()),
            "unet64_ce_dice": json.loads((ROOT / "outputs/enhanced/segmentation/unet64_ce_dice_100ep/evaluation_metrics.json").read_text()),
        },
        "tracking": json.loads((ROOT / "outputs/detection_tracking/tracked_video/campus_road_01_yolov8s_summary.json").read_text()),
    }
    (FIG_DIR / "result_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    })
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_classification()
    plot_segmentation()
    plot_detection()
    copy_figures()
    write_result_tables_json()


if __name__ == "__main__":
    main()
