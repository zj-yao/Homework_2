# Homework 2 Project Guide

本文件用于让新的 Codex/Agent 对话快速理解本项目。每次开始本作业相关的新对话时，先阅读本文件，再阅读 `PROGRESS.md` 和 `HW2_深度学习与空间智能.pdf`，然后再执行具体任务。

## Project Context

- 课程作业：深度学习与空间智能期中作业 / Homework 2。
- 原始要求文件：`HW2_深度学习与空间智能.pdf`。
- 当前目录已建立代码脚手架、配置、数据软链接、smoke training 输出和正式训练队列脚本；不要再按“空目录”处理。
- 该目录当前可能不是 git 仓库；如果需要提交代码到 GitHub，后续应初始化 git 或将本目录迁移到公开仓库。
- 本项目使用的 Python 环境是 conda 环境 `dl_hw2`：

```bash
conda activate dl_hw2
```

环境路径：

```text
/home/yzj/.conda/envs/dl_hw2
```

之前上级 README 提到的 `/home/yzj/.venv-gpu` 和 `dlgpu` 不作为本项目的默认环境；本机检查时没有找到对应环境。后续训练、测试、生成报告图表时，默认都在 `dl_hw2` 中运行。

## Current State For Future Agents

Last updated: 2026-04-28.

当前项目已经从规划阶段进入正式训练与增强实验阶段：

- 数据已经下载到 `/data/yzj/homework2_data/`，repo 内 `data/*` 是软链接。
- 已完成 smoke training：
  - classification smoke: `outputs/smoke/classification_baseline/`
  - segmentation smoke: `outputs/smoke/segmentation_unet_ce/`
  - detection smoke: `runs/detect/outputs/smoke/detection_yolo/yolov8n_smoke/`
- Smoke 后验证：`conda run -n dl_hw2 python -m pytest -q` 通过，结果是 `36 passed`。
- SwanLab offline logging 已验证可用；正式训练使用 `swanlab` 的 offline mode，不需要现在登录账号。日志会落在 `swanlog/`，后续可用 `swanlab sync <run_dir>` 同步到云端，或直接截本地可视化/曲线。
- 第一轮正式训练队列已完成，脚本：`scripts/run_formal_training_queue.sh`。
- 第一轮正式训练进度文件：
  - `outputs/formal_training/status.tsv`
  - `outputs/formal_training/current_step.txt`
  - `outputs/formal_training/summary.txt`
  - 每一步详细日志：`outputs/formal_training/logs/*.log`
- 增强实验队列脚本：`scripts/run_enhanced_training_queue.sh`。
- 增强实验队列曾于 2026-04-28 11:27:18 UTC 按用户要求暂停以释放 GPU，队列终止并写入 `PAUSED_BY_USER`。
- 2026-04-29 用户要求继续运行；已给 `src/segmentation/train.py` 增加 `train.resume_from` 支持，可从 U-Net64 的 `latest.pt` 接着训练。
- 继续运行脚本：`scripts/run_enhanced_training_resume.sh`。它不会重跑已完成的 ResNet34，只会从 U-Net64 epoch 36 checkpoint 续训到 100，然后继续 YOLOv8s。
- 2026-04-29 续跑时 U-Net64 已完成并评估；YOLOv8s 权重准备曾因代理连接失败中断。随后已手动下载 `/data/yzj/homework2_data/pretrained/yolov8s.pt`，并创建 `scripts/run_enhanced_yolov8s_remaining.sh` 用于只跑剩余 YOLOv8s train/eval。
- 暂停时已完成：
  - `classification_resnet34_pretrained_60ep_train`
  - `classification_resnet34_pretrained_60ep_eval_test`
- 暂停时正在运行但被终止：
  - `segmentation_unet64_ce_dice_100ep_train`
  - 已保存到 epoch 36/100 的 `history.csv`、`latest.pt`、`best.pt`，位置：`outputs/enhanced/segmentation/unet64_ce_dice_100ep/`
- 暂停时尚未运行：
  - `segmentation_unet64_ce_dice_100ep_eval_val`
  - `prepare_yolov8s_pretrained`
  - `detection_yolov8s_vehicle_100ep_train`
  - `detection_yolov8s_vehicle_100ep_eval_val`
- 注意：如果继续运行脚本正在跑或跑完，应优先查看 `outputs/enhanced_training/status.tsv` 和 `outputs/enhanced_training/current_step.txt`。
- 增强实验队列进度文件：
  - `outputs/enhanced_training/status.tsv`
  - `outputs/enhanced_training/current_step.txt`
  - `outputs/enhanced_training/summary.txt`
  - `outputs/enhanced_training/pause_note.txt`
  - 每一步详细日志：`outputs/enhanced_training/logs/*.log`

如果用户问“现在跑到哪了”，先执行：

```bash
pgrep -af 'run_formal_training_queue.sh|run_enhanced_training_queue.sh' || true
cat outputs/enhanced_training/current_step.txt 2>/dev/null || cat outputs/formal_training/current_step.txt 2>/dev/null || true
tail -n 30 outputs/enhanced_training/status.tsv 2>/dev/null || tail -n 30 outputs/formal_training/status.tsv 2>/dev/null || true
```

再查看当前步骤日志，例如：

```bash
tail -n 80 outputs/enhanced_training/logs/<current_step>.log
```

第一轮正式训练队列按顺序运行，不并行抢 GPU：

1. Classification:
   - ResNet18 pretrained baseline, 30 epochs.
   - ResNet18 pretrained baseline, 15 epochs.
   - ResNet18 pretrained low LR, 15 epochs.
   - ResNet18 pretrained high LR, 15 epochs.
   - ResNet18 random initialization, 30 epochs.
   - ResNet18 + SE block, 30 epochs.
   - 每个分类实验训练后会跑 test evaluation。
2. Segmentation:
   - U-Net + CE Loss, 50 epochs.
   - U-Net + Dice Loss, 50 epochs.
   - U-Net + CE + Dice Loss, 50 epochs.
   - 每个分割实验训练后会跑 val evaluation。
3. Detection:
   - YOLOv8n vehicle detector, 50 epochs, `imgsz=640`, `batch=8`.
   - 训练后跑 YOLO validation。

增强实验队列用于提高结果质量，不是 PDF 的最低要求，但可用于报告中的 stronger experiments：

1. Classification:
   - ResNet34 pretrained, 60 epochs.
   - 训练后跑 test evaluation。
   - 配置：`configs/classification/resnet34_pretrained_60ep.yaml`
   - 输出：`outputs/enhanced/classification/resnet34_pretrained_60ep/`
2. Segmentation:
   - U-Net `base_channels=64` + CE + Dice, 100 epochs.
   - 训练后跑 val evaluation。
   - 配置：`configs/segmentation/unet64_ce_dice_100ep.yaml`
   - 输出：`outputs/enhanced/segmentation/unet64_ce_dice_100ep/`
3. Detection:
   - YOLOv8s vehicle detector, 100 epochs, `imgsz=640`, `batch=8`.
   - 训练前会下载/确认 `/data/yzj/homework2_data/pretrained/yolov8s.pt`。
   - 训练后跑 YOLO validation。
   - 配置：`configs/detection/yolov8s_vehicle.yaml`
   - 输出：`outputs/enhanced/detection_tracking/yolov8s_vehicle/yolov8s_vehicle_100ep/`

视频已准备：`data/videos/campus_road_01.mp4`，约 19.53 秒、30 FPS、1280x720。Task 2 的 final tracking、遮挡 ID 分析、越线计数已完成；产物在 `outputs/detection_tracking/tracked_video/` 和 `outputs/detection_tracking/occlusion_frames/campus_road_01/`。

## Local Environment And GPU Notes

本机已检查到 `dl_hw2` 中有本项目需要的主要依赖：

- Python `3.10.20`
- PyTorch `2.5.1+cu121`
- Torchvision `0.20.1+cu121`
- Ultralytics YOLO `8.4.41`
- OpenCV `4.13.0`
- Kaggle API `1.7.4.5`
- kagglehub `1.0.0`
- wandb `0.26.1`
- swanlab `0.7.16`
- pytest `9.0.3`
- NumPy、Matplotlib、Pandas、Scikit-learn、Albumentations 等常用实验包

GPU 情况：

- GPU: NVIDIA GeForce RTX 3060
- VRAM: 12GB
- Driver: `535.216.03`
- CUDA reported by driver: `12.2`
- CUDA used by PyTorch: `12.1`
- PyTorch CUDA test passed.

This GPU is sufficient for the assignment if training settings are kept practical:

- Task 1 classification: ResNet-18/ResNet-34 fine-tuning is fine. Start with batch size `32` or `64`.
- Task 2 detection/tracking: use YOLOv8n or YOLOv8s first, input size `640`, batch size `8` or `16`. Avoid YOLOv8m/l unless there is a clear need.
- Task 3 segmentation: start U-Net with image size `256x256` or `320x320`, batch size `4` or `8`.
- If trying ViT-Tiny or Swin-T for Task 1, watch VRAM and training time. If time is limited, SE-block or CBAM on ResNet is the safer attention experiment.

## High-Level Goal

完成三个深度学习实验任务，并最终提交一份 PDF 实验报告、公开 GitHub 代码仓库、训练好的模型权重下载链接。

三个任务分别是：

1. 使用 ImageNet 预训练 CNN 在 `102 Category Flower Dataset` 上做图像分类微调。
2. 使用 `Road Vehicle Images Dataset` 微调 YOLOv8 或类似单阶段检测器，并完成视频多目标跟踪、遮挡分析、越线计数。
3. 从零手写 U-Net，在 `Stanford Background Dataset` 上做语义分割，并比较 CE Loss、Dice Loss、CE + Dice 的 mIoU。

注意：PDF 第一项标题写了“宠物识别”，但正文明确要求使用 `102 Category Flower Dataset`。实现和报告应以正文的数据集要求为准，可在报告中简单说明按正文执行花卉分类任务。

## Dataset Source Links

- 102 Category Flower Dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- Road Vehicle Images Dataset: https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset
- Stanford Background Dataset: http://dags.stanford.edu/projects/scenedataset.html

Dataset preparation details live in `scripts/download_or_prepare_data.md`. Notes:

- Flower102 configs default to `torchvision.datasets.Flowers102` with `download: true`, because the Oxford source is not an ImageFolder layout.
- The Road Vehicle dataset is on Kaggle and may require Kaggle login/API credentials.
- Stanford Background download is linked from the Stanford page as `iccv09Data.tar.gz`; semantic labels use `labels/*.regions.txt`.
- Large datasets should live under `/data/yzj/homework2_data/`; repo-local `data/*` paths should be symlinks to that external directory.

## Required Final Deliverables

最终交付必须包括：

- PDF 格式实验报告。
- Public GitHub repo，包含完整代码与清晰 README。
- README 中写明：
  - 环境配置方法；
  - 数据集准备方式；
  - 三个任务的训练命令；
  - 测试/推理/可视化命令；
  - 主要结果和产物位置。
- 训练好的模型权重上传到百度云、Google Drive 或其他网盘。
- 实验报告中必须包含：
  - GitHub repo 链接；
  - 模型权重网盘下载地址；
  - 模型结构、数据集、实验结果介绍；
  - 训练/验证/测试集划分；
  - 网络结构；
  - batch size；
  - learning rate；
  - optimizer；
  - iteration / epoch；
  - loss function；
  - evaluation metrics；
  - wandb 或 swanlab 的训练可视化截图；
  - 训练集和验证集 loss 曲线；
  - 验证集 Accuracy / mAP 曲线；
  - 小组成员姓名、学号、具体分工。

## Recommended Repository Layout

建议后续按下面结构组织项目。不要把所有任务写进一个超大的脚本。

```text
Homework_2/
  AGENTS.md
  README.md
  requirements.txt
  HW2_深度学习与空间智能.pdf

  configs/
    classification/
    detection/
    segmentation/

  data/
    flower102/
    road_vehicle/
    stanford_background/
    videos/

  src/
    common/
      seed.py
      logging.py
      metrics.py
      paths.py

    classification/
      dataset.py
      models.py
      train.py
      evaluate.py
      search.py
      visualize.py

    detection_tracking/
      prepare_data.py
      train_yolo.py
      evaluate_yolo.py
      track_video.py
      line_counter.py
      occlusion_analysis.py

    segmentation/
      dataset.py
      unet.py
      losses.py
      metrics.py
      train.py
      evaluate.py
      visualize.py

  scripts/
    download_or_prepare_data.md
    run_classification.sh
    run_detection_tracking.sh
    run_segmentation.sh

  outputs/
    classification/
    detection_tracking/
    segmentation/

  report/
    main.tex or main.md
    figures/
    final_report.pdf
```

If time is short, the structure can be simplified, but keep the three task areas separated.

## Task 1: Flower Classification With Transfer Learning

### Objective

在 `102 Category Flower Dataset` 上训练花卉分类模型，对比预训练、随机初始化、超参数变化、注意力机制带来的影响。

### Required Dataset

- Dataset: `102 Category Flower Dataset`
- Number of classes: 102
- Need train/val/test split.
- Record split strategy in report. If using an official split, document it. If manually splitting, use a fixed random seed and save the split files.

### Baseline Requirement

Use ResNet-18 or ResNet-34:

- Load ImageNet-pretrained backbone.
- Replace final classification layer with a 102-class output layer.
- Train the new output layer from scratch.
- Fine-tune the remaining layers with a smaller learning rate.

Recommended baseline:

- `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`
- Final layer: `nn.Linear(in_features, 102)`
- Optimizer with parameter groups:
  - backbone learning rate: small, for example `1e-4`
  - classifier learning rate: larger, for example `1e-3`
- Use standard image augmentations:
  - random resized crop;
  - random horizontal flip;
  - normalization with ImageNet mean/std.

### Experiments To Run

At minimum, run these experiments:

1. **Pretrained ResNet baseline**
   - ImageNet initialization.
   - Replace final layer.
   - Different learning rates for backbone and head.

2. **Hyperparameter analysis**
   - Compare at least 2-3 learning rates or learning-rate combinations.
   - Compare at least 2 training lengths, such as fewer vs more epochs.
   - Keep a table of validation accuracy and final test accuracy.

3. **Pretraining ablation**
   - Same architecture, but random initialization.
   - Compare with pretrained baseline.
   - Discuss convergence speed and final accuracy.

4. **Attention or lightweight transformer comparison**
   - Preferred practical path: add SE block or CBAM to ResNet.
   - Alternative: train a lightweight ViT-Tiny or Swin-T if dependencies and GPU time allow.
   - Compare Accuracy against baseline.

### Metrics And Artifacts

Save:

- best checkpoint;
- final checkpoint if useful;
- `history.json` or `history.csv`;
- train/val loss curves;
- validation accuracy curve;
- final test accuracy;
- confusion matrix or sample predictions if time allows.

Report:

- model architecture;
- training details;
- hyperparameter table;
- pretrained vs random initialization comparison;
- attention model vs baseline comparison.

## Task 2: Road Vehicle Detection, Tracking, Occlusion Analysis, And Line Counting

### Objective

训练一个车辆检测模型，并在 10-30 秒测试视频上完成多目标跟踪、遮挡/ID 跳变分析和越线计数。

### Required Dataset

- Dataset: `Road Vehicle Images Dataset`
- Need convert or prepare labels into YOLO-compatible format if using YOLOv8.
- Need train/val split.
- Document class names and dataset format in README and report.

### Detection Model Requirement

Use YOLOv8 or another modern one-stage detector.

Recommended practical path:

- Use Ultralytics YOLOv8.
- Start from a small model such as `yolov8n.pt` or `yolov8s.pt`.
- Fine-tune on Road Vehicle Images Dataset.
- Export/save best checkpoint.

Expected artifacts:

- trained model weights, such as `best.pt`;
- validation metrics;
- mAP curve or wandb/swanlab screenshot;
- sample detection images.

### Video Tracking Requirement

Prepare a 10-30 second video:

- Can be filmed by phone on campus, road, or intersection.
- Put raw video under `data/videos/`.
- Keep one processed output video with bounding boxes, class names, confidence, and stable Tracking ID.

Use YOLOv8 tracking or similar:

- Each target must display a stable `Tracking ID`.
- Output must include bounding box and class.
- Save annotated video.

Recommended implementation:

- `track_video.py` loads trained YOLO weights.
- Use YOLO tracking backend such as ByteTrack or BoT-SORT.
- For each frame, draw:
  - bounding box;
  - class;
  - confidence;
  - track ID.

### Occlusion And ID Switch Analysis

Find a video segment where vehicles overlap, cross, or become dense.

Required:

- Extract 3-4 consecutive frames from this segment.
- Visualize each frame with boxes and IDs.
- In the report, analyze whether:
  - the tracker maintained the same ID;
  - the object was temporarily lost;
  - an ID switch occurred.

Analysis should mention possible causes:

- detector confidence drop during occlusion;
- bounding boxes overlap heavily;
- motion prediction uncertainty;
- association based on IoU, appearance, or trajectory;
- frame rate and video blur.

### Line Crossing Count

Implement a virtual counting line in the video frame.

Required logic:

- Define two points for the line.
- For each tracked object, compute detection-box center.
- Store previous and current center for each Tracking ID.
- Count an object once when its center crosses the line.
- Use Tracking ID to avoid double counting.
- Draw current total count on video.

Recommended output:

- annotated tracking video with line and count;
- final count value in console/log;
- small explanation in report.

## Task 3: U-Net From Scratch For Semantic Segmentation

### Objective

手写一个经典 U-Net，在 `Stanford Background Dataset` 上做语义分割，并比较三种损失函数对验证集 mIoU 的影响。

### Required Dataset

- Dataset: `Stanford Background Dataset`
- Need image/mask loading.
- Need train/val split.
- Record number of semantic classes.
- Make sure mask labels are converted to integer class IDs.

### Model Requirement

Do not use pretrained weights.

Must manually implement a U-Net using basic deep learning framework APIs:

- encoder/downsampling path;
- decoder/upsampling path;
- skip connections;
- final pixel-wise classifier.

Recommended PyTorch modules:

- `nn.Conv2d`
- `nn.BatchNorm2d` if desired
- `nn.ReLU`
- `nn.MaxPool2d`
- `nn.ConvTranspose2d` or bilinear upsampling + convolution

Do not use a library-provided pretrained segmentation model.

### Loss Function Requirement

Manually implement Dice Loss.

Train and compare:

1. Cross-Entropy Loss only.
2. Dice Loss only.
3. Cross-Entropy Loss + Dice Loss.

Keep all non-loss settings as consistent as possible:

- same train/val split;
- same image size;
- same optimizer;
- same batch size;
- same epoch count;
- same random seed.

### Metrics And Artifacts

Primary metric:

- validation mIoU.

Also save:

- train/val loss curves;
- validation mIoU curve;
- qualitative segmentation predictions;
- best checkpoint for each loss configuration;
- metrics table comparing CE, Dice, and CE + Dice.

Report should discuss:

- why Dice Loss helps with pixel imbalance;
- whether Dice alone is stable;
- whether the combined loss improves mIoU.

## Common Experiment Standards

Use consistent experiment hygiene across all three tasks.

### Reproducibility

- Set random seeds where possible.
- Save config files for each run.
- Save exact train/val/test splits if manually generated.
- Record hardware/GPU and major package versions.

### Logging

Use wandb or swanlab as required by the assignment.

For each task, log:

- train loss;
- validation loss;
- validation metric:
  - classification: Accuracy;
  - detection: mAP;
  - segmentation: mIoU.

Screenshots from wandb/swanlab must be included in the final PDF report.

### Checkpoints

Each task should save:

- best model by validation metric;
- latest model if useful;
- config and metrics beside the checkpoint.

Suggested output naming:

```text
outputs/
  classification/
    resnet18_pretrained_baseline/
    resnet18_random_init/
    resnet18_se_block/

  detection_tracking/
    yolov8_vehicle/
    tracked_video/
    occlusion_frames/

  segmentation/
    unet_ce/
    unet_dice/
    unet_ce_dice/
```

### Evaluation Tables

The final report should include at least these tables:

1. Classification experiment table:
   - model;
   - pretrained or random initialization;
   - attention module;
   - learning rate;
   - epochs;
   - validation accuracy;
   - test accuracy.

2. Detection experiment table:
   - model;
   - input size;
   - epochs;
   - mAP;
   - precision;
   - recall.

3. Segmentation experiment table:
   - loss function;
   - epochs;
   - validation mIoU;
   - validation loss.

## Recommended Work Order

Recommended implementation order:

1. Create repo skeleton:
   - `README.md`
   - `requirements.txt`
   - `src/`
   - `configs/`
   - `outputs/`
   - `report/`

2. Prepare shared utilities:
   - random seed setup;
   - config loading;
   - metric saving;
   - plotting helper;
   - wandb/swanlab setup.

3. Finish Task 1 first:
   - easiest to get a stable baseline quickly;
   - establishes logging and checkpoint conventions;
   - provides early report material.

4. Finish Task 3 second:
   - from-scratch U-Net needs careful debugging;
   - loss comparison is self-contained.

5. Finish Task 2 third:
   - detection data conversion and video tracking have more external friction;
   - final visual artifacts are important for the report.

6. Upload weights:
   - classification best models;
   - YOLO best detector;
   - U-Net best models for each loss configuration.

7. Write final report:
   - fill results tables;
   - insert wandb/swanlab screenshots;
   - insert visual examples;
   - add repo and weight links.

## Minimum Viable Completion Plan

If time becomes limited, prioritize the following:

1. Task 1:
   - pretrained ResNet baseline;
   - random initialization ablation;
   - one attention/SE model;
   - one small hyperparameter comparison.

2. Task 2:
   - train YOLOv8 small model;
   - run tracking on a short video;
   - implement line counting;
   - extract 3-4 occlusion frames and analyze.

3. Task 3:
   - implement working U-Net;
   - train CE, Dice, CE + Dice;
   - report validation mIoU table.

This satisfies the assignment more reliably than over-optimizing one task while leaving another incomplete.

## Report Outline

Suggested PDF report structure:

1. Cover page:
   - course;
   - assignment title;
   - group members;
   - student IDs;
   - division of labor;
   - GitHub repo link;
   - model weight link.

2. Introduction:
   - brief overview of three tasks.

3. Task 1: Flower Classification:
   - dataset;
   - model architecture;
   - training settings;
   - hyperparameter analysis;
   - pretrained ablation;
   - attention comparison;
   - curves and results table.

4. Task 2: Vehicle Detection And Tracking:
   - dataset;
   - YOLOv8 training setup;
   - mAP results;
   - tracking pipeline;
   - occlusion frames and ID analysis;
   - line-counting logic and result.

5. Task 3: Semantic Segmentation:
   - dataset;
   - U-Net architecture;
   - Dice Loss implementation idea;
   - CE vs Dice vs CE + Dice comparison;
   - mIoU table and qualitative examples.

6. Conclusion:
   - summarize main findings;
   - compare transfer learning, detection/tracking, and segmentation observations;
   - mention limitations and possible improvements.

## Notes For Future Agents

- Do not assume datasets are already downloaded.
- Do not overwrite user-created experiment outputs without asking.
- Keep task-specific code separated.
- Prefer clear, reproducible scripts over notebook-only workflows.
- When writing code, preserve exact commands in README so the report can cite them.
- When changing experiment settings, update both config and report notes.
- Before claiming a result, verify that the metric file, checkpoint, or plot exists.
- If a run is only a smoke test, label it clearly and do not present it as a final result.
- For final submission, make sure report links are real public links, not local paths.
