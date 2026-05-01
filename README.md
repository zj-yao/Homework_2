# Deep Learning Homework 2

This repository contains the code scaffold for Homework 2: image classification, road-vehicle detection/tracking, and semantic segmentation.

Public GitHub repository: <https://github.com/zj-yao/Homework_2>

Model weights: <https://drive.google.com/drive/folders/1GpObjJP_6nYOdDF8HlNkBnlWKpdh7mBl?usp=sharing>

Read `AGENTS.md` first for the full project breakdown, environment notes, GPU constraints, and required deliverables.

## Team

| Name | Student ID | Division of labor |
| --- | --- | --- |
| 姚宗骏 | 25110980027 | Flower classification experiments, overall code organization, report integration |
| 付思维 | 25210980037 | Vehicle detection/tracking, video processing, line-counting analysis |
| 姜涵霖 | 25110980011 | U-Net semantic segmentation, loss comparison, result analysis |

## Environment

Use the prepared conda environment:

```bash
conda activate dl_hw2
```

Or run commands without activating:

```bash
conda run -n dl_hw2 python -m pytest
```

The environment has PyTorch CUDA, torchvision, Ultralytics YOLO, OpenCV, wandb, swanlab, and pytest installed.

## Project Layout

```text
src/
  common/              # shared config, seeding, and I/O helpers
  classification/      # Task 1: Flower102 transfer learning
  detection_tracking/  # Task 2: YOLOv8 detection, tracking, counting
  segmentation/        # Task 3: U-Net semantic segmentation

configs/
  classification/
  detection/
  segmentation/

tests/
  common/
  classification/
  detection_tracking/
  segmentation/

scripts/
```

## Current Status

All three required tasks have been implemented and trained. The final report draft is in `tex_report/main.pdf`, and generated report figures are in `tex_report/figures/`.

Datasets used:

- `102 Category Flower Dataset`: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- `Road Vehicle Images Dataset`: https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset
- `Stanford Background Dataset`: http://dags.stanford.edu/projects/scenedataset.html
- a 19.53 second road/campus video for tracking and line counting

More detailed dataset preparation notes are in `scripts/download_or_prepare_data.md`.

Large datasets are stored outside the repo at:

```text
/data/yzj/homework2_data/
```

The repo-local `data/` entries are symlinks to that directory, so existing configs can still use paths such as `data/flower102` and `data/stanford_background`.

## Smoke Tests

```bash
conda run -n dl_hw2 python -m pytest tests -q
```

Task-specific smoke tests:

```bash
conda run -n dl_hw2 python -m pytest tests/classification -q
conda run -n dl_hw2 python -m pytest tests/detection_tracking -q
conda run -n dl_hw2 python -m pytest tests/segmentation -q
```

## Task 1: Flower102 Classification

The default classification configs use `torchvision.datasets.Flowers102` with `download: true`, so the first run can download and use the official Oxford files automatically under `data/flower102/`.

Run the pretrained ResNet-18 baseline:

```bash
bash scripts/run_classification.sh configs/classification/baseline_resnet18.yaml
```

Other provided configs:

```bash
bash scripts/run_classification.sh configs/classification/resnet18_random_init.yaml
bash scripts/run_classification.sh configs/classification/resnet18_se.yaml
```

Evaluate a checkpoint manually:

```bash
conda run -n dl_hw2 python -m src.classification.evaluate \
  --config configs/classification/baseline_resnet18.yaml \
  --checkpoint outputs/classification/resnet18_pretrained_baseline/best.pt \
  --split test
```

## Task 2: Vehicle Detection, Tracking, And Counting

Prepare a YOLO-format vehicle dataset and update:

```text
configs/detection/road_vehicle_data.yaml
```

The dataset is hosted on Kaggle, so command-line download usually requires a configured Kaggle API token.

Train YOLOv8:

```bash
scripts/run_detection_tracking.sh train --config configs/detection/yolov8_vehicle.yaml
```

Evaluate the trained detector:

```bash
scripts/run_detection_tracking.sh eval \
  --model outputs/detection_tracking/yolov8_vehicle/yolov8n_vehicle/weights/best.pt \
  --data-yaml configs/detection/road_vehicle_data.yaml
```

Track a 10-30 second video and draw a counting line:

```bash
scripts/run_detection_tracking.sh track \
  --model outputs/detection_tracking/yolov8_vehicle/yolov8n_vehicle/weights/best.pt \
  --video data/videos/input.mp4 \
  --output outputs/detection_tracking/tracked_video/output.mp4 \
  --line 200 400 900 400
```

Extract 3-4 consecutive frames for occlusion analysis:

```bash
scripts/run_detection_tracking.sh occlusion \
  --video data/videos/input.mp4 \
  --output-dir outputs/detection_tracking/occlusion_frames \
  --start-frame 120 \
  --frame-count 4
```

## Task 3: U-Net Segmentation

Prepare Stanford Background data under `data/stanford_background/` or edit the segmentation configs. The default configs expect:

```text
data/stanford_background/
  images/
  labels/              # supports raw *.regions.txt masks
  splits/train.txt
  splits/val.txt
```

The raw Stanford Background semantic labels may contain negative values for unknown pixels. The provided segmentation configs set `ignore_index: -1` so CE, Dice, CE+Dice, and mIoU ignore those pixels.

Run the three required loss experiments:

```bash
bash scripts/run_segmentation.sh configs/segmentation/unet_ce.yaml
bash scripts/run_segmentation.sh configs/segmentation/unet_dice.yaml
bash scripts/run_segmentation.sh configs/segmentation/unet_ce_dice.yaml
```

Evaluate a checkpoint:

```bash
conda run -n dl_hw2 python -m src.segmentation.evaluate \
  --config configs/segmentation/unet_ce.yaml \
  --checkpoint outputs/segmentation/unet_ce/best.pt \
  --split val
```

## Submission Reminders

Final submission needs:

- PDF experiment report: `tex_report/main.pdf`;
- public GitHub repository link: <https://github.com/zj-yao/Homework_2>;
- model weights uploaded to cloud storage: <https://drive.google.com/drive/folders/1GpObjJP_6nYOdDF8HlNkBnlWKpdh7mBl?usp=sharing>;
- wandb or swanlab screenshots for loss and validation metrics;
- task results tables for Accuracy, mAP, and mIoU.
