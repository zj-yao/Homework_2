# Homework 2 Progress

Last updated: 2026-05-01.

本文件记录当前作业进度，方便之后新开对话时快速接上。新对话建议先读 `AGENTS.md`，再读本文件。

## Current Status

- 代码脚手架、配置文件、训练脚本、评估脚本已建立。
- 三个数据集已下载并通过 `data/*` 软链接接入项目。
- Smoke training 已完成并验证过。
- 第一轮正式训练已全部完成。
- 增强实验已全部完成：
  - ResNet34 pretrained 60 epochs。
  - U-Net64 CE+Dice 100 epochs。
  - YOLOv8s 100 epochs。
- 当前没有本项目训练进程在跑。
- 用户视频已上传并重命名为 `data/videos/campus_road_01.mp4`。视频已检查合格：19.53 秒，30 FPS，1280x720。
- Task 2 视频 tracking、越线计数、遮挡连续帧提取已完成。
- 报告草稿已开始写在 `tex_report/main.tex`，报告图表素材已生成在 `tex_report/figures/`。
- 本机 TeX Live 可用路径：`/data/yzj/texlive/2025/bin/x86_64-linux/`。
- `tex_report/main.tex` 已用该 TeX Live 编译生成 `tex_report/main.pdf`。
- 本地 Git 仓库已初始化，当前分支 `main`，首个 commit 为 `1072a1c Initial homework 2 project`。
- 远端 GitHub 仓库已创建并推送：`https://github.com/zj-yao/Homework_2`。
- Google Drive 模型权重链接已提供：`https://drive.google.com/drive/folders/1GpObjJP_6nYOdDF8HlNkBnlWKpdh7mBl?usp=sharing`。
- 模型权重已实际上传到 Google Drive：
  - `homework2_model_weights.zip`，直接链接 `https://drive.google.com/open?id=178zInJjSr-Ue0lw0Fc0YVSLJvls1ZG4E`；
  - `WEIGHTS_MANIFEST.txt`；
  - `SHA256SUMS.txt`。
  - Drive 端 zip MD5：`ce483e8cbe79de93c3a4e773e44bb0af`，与本地一致。
- 小组成员信息已提供并写入报告/README：
  - 姚宗骏，25110980027；
  - 付思维，25210980037；
  - 姜涵霖，25110980011。

## Data

数据实际存放在 `/data/yzj/homework2_data/`，项目内 `data/*` 是软链接。

| Task | Dataset | Project path | Real path |
| --- | --- | --- | --- |
| Task 1 Classification | 102 Category Flower Dataset | `data/flower102` | `/data/yzj/homework2_data/flower102` |
| Task 2 Detection/Tracking | Road Vehicle Images Dataset | `data/road_vehicle` | `/data/yzj/homework2_data/road_vehicle` |
| Task 3 Segmentation | Stanford Background Dataset | `data/stanford_background` | `/data/yzj/homework2_data/stanford_background` |
| Task 2 Video | User-recorded video | `data/videos` | `/data/yzj/homework2_data/videos` |

Uploaded video: `data/videos/campus_road_01.mp4`.

## Training Queues

First formal training queue:

- Script: `scripts/run_formal_training_queue.sh`
- Status file: `outputs/formal_training/status.tsv`
- Status: `ALL_DONE`
- Completed at: 2026-04-28 10:09:07 UTC

Enhanced training:

- Main script: `scripts/run_enhanced_training_queue.sh`
- Resume script: `scripts/run_enhanced_training_resume.sh`
- YOLOv8s remaining script: `scripts/run_enhanced_yolov8s_remaining.sh`
- Status file: `outputs/enhanced_training/status.tsv`
- Status: `ALL_DONE`
- Completed at: 2026-04-29 05:12:42 UTC

Note: enhanced training was paused once at U-Net64 epoch 36 to free the GPU, then resumed from `latest.pt`.

## Task 1: Flower Classification

Dataset: `102 Category Flower Dataset`.

All classification experiments use Flower102 with train/val/test evaluation. Test set size is 6149 images.

| Experiment | Epochs | Best Val Acc | Test Acc | Output |
| --- | ---: | ---: | ---: | --- |
| ResNet18 pretrained baseline | 30 | 0.9176 | 0.8853 | `outputs/classification/resnet18_pretrained_baseline/` |
| ResNet18 pretrained baseline | 15 | 0.9059 | 0.8785 | `outputs/classification/resnet18_pretrained_baseline_15ep/` |
| ResNet18 pretrained low LR | 15 | 0.8961 | 0.8666 | `outputs/classification/resnet18_pretrained_lr_low_15ep/` |
| ResNet18 pretrained high LR | 15 | 0.9147 | 0.8862 | `outputs/classification/resnet18_pretrained_lr_high_15ep/` |
| ResNet18 random init | 30 | 0.4255 | 0.3627 | `outputs/classification/resnet18_random_init/` |
| ResNet18 + SE block | 30 | 0.8892 | 0.8580 | `outputs/classification/resnet18_se_block/` |
| ResNet34 pretrained enhanced | 60 | 0.9196 | 0.8863 | `outputs/enhanced/classification/resnet34_pretrained_60ep/` |

Current best classification test accuracy:

- ResNet34 pretrained 60ep: `0.8863`
- ResNet18 pretrained high LR 15ep is essentially tied: `0.8862`

Useful report points:

- ImageNet pretraining strongly outperforms random initialization.
- ResNet34 slightly improves validation accuracy, but test accuracy is only marginally higher than the best ResNet18 setting.
- SE-block did not improve this run, so report it honestly as an attention experiment that underperformed the baseline.

## Task 2: Vehicle Detection And Future Tracking

Dataset: `Road Vehicle Images Dataset`.

Training data path: `data/road_vehicle/trafic_data`.

| Experiment | Epochs | Precision | Recall | mAP50 | mAP50-95 | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| YOLOv8n formal | 50 | 0.5952 | 0.3833 | 0.4315 | 0.2648 | `outputs/detection_tracking/yolov8_vehicle/yolov8n_vehicle/` |
| YOLOv8s enhanced | 100 | 0.7359 | 0.4597 | 0.5220 | 0.3108 | `outputs/enhanced/detection_tracking/yolov8s_vehicle/yolov8s_vehicle_100ep/` |

Current best detection model:

- `outputs/enhanced/detection_tracking/yolov8s_vehicle/yolov8s_vehicle_100ep/weights/best.pt`

YOLOv8s is better than YOLOv8n and should be used for final video tracking unless inference speed becomes a problem.

Video tracking outputs:

- Input video: `data/videos/campus_road_01.mp4`
- Model: `outputs/enhanced/detection_tracking/yolov8s_vehicle/yolov8s_vehicle_100ep/weights/best.pt`
- Final tracked/counting video: `outputs/detection_tracking/tracked_video/campus_road_01_yolov8s_tracked_counted.mp4`
- Tracking contact sheet: `outputs/detection_tracking/tracked_video/campus_road_01_yolov8s_tracking_contact_sheet.jpg`
- Counting summary: `outputs/detection_tracking/tracked_video/campus_road_01_yolov8s_summary.json`
- Counting line: `(640, 180)` to `(640, 720)`
- Final count: `5`
- Crossed IDs: `1, 24, 26, 67, 90`

Occlusion outputs:

- Consecutive frames: `outputs/detection_tracking/occlusion_frames/campus_road_01/occlusion_000536.jpg` to `occlusion_000539.jpg`
- Occlusion contact sheet: `outputs/detection_tracking/occlusion_frames/campus_road_01/occlusion_contact_sheet.jpg`
- Draft analysis text: `outputs/detection_tracking/occlusion_frames/campus_road_01/occlusion_analysis.md`
- Selected time range: about `17.87s-17.97s`
- Observation: foreground white car partially occludes right-side vehicles; no obvious ID switch is visible in the selected consecutive frames.

## Task 3: Semantic Segmentation

Dataset: `Stanford Background Dataset`.

Model: handwritten U-Net from scratch, no pretrained weights.

| Experiment | Epochs | Val mIoU | Val Loss | Output |
| --- | ---: | ---: | ---: | --- |
| U-Net32 + CE | 50 | 0.5776 | 0.5997 | `outputs/segmentation/unet_ce/` |
| U-Net32 + Dice | 50 | 0.5566 | 0.3818 | `outputs/segmentation/unet_dice/` |
| U-Net32 + CE+Dice | 50 | 0.6006 | 1.0844 | `outputs/segmentation/unet_ce_dice/` |
| U-Net64 + CE+Dice enhanced | 100 | 0.5874 | 1.1101 | `outputs/enhanced/segmentation/unet64_ce_dice_100ep/` |

Current best segmentation result:

- U-Net32 + CE+Dice, val mIoU `0.6006`.

Useful report points:

- CE+Dice gave the best result among the three required loss functions.
- Dice alone was lower than CE+Dice.
- Increasing U-Net capacity to base channels 64 and training longer did not improve over U-Net32 CE+Dice in this run, likely due to overfitting or optimization instability.

## Logs And Visualization

SwanLab offline logging was enabled for formal/enhanced training. Local logs are under:

- `swanlog/`

CSV/plot artifacts are also available:

- Classification histories: `outputs/classification/*/history.csv`, `outputs/enhanced/classification/*/history.csv`
- Segmentation histories: `outputs/segmentation/*/history.csv`, `outputs/enhanced/segmentation/*/history.csv`
- YOLO curves:
  - `outputs/detection_tracking/yolov8_vehicle/yolov8n_vehicle/results.png`
  - `outputs/enhanced/detection_tracking/yolov8s_vehicle/yolov8s_vehicle_100ep/results.png`

For the final report, use SwanLab screenshots or generated plots from these CSV files.

## Report Draft

Report folder requested by the user:

- `tex_report/main.tex`: current TeX report draft.
- `tex_report/main.pdf`: compiled PDF from the current draft.
- `tex_report/README.md`: compile notes.
- `tex_report/figures/`: generated curves, tracking contact sheet, occlusion contact sheet, YOLO results figure, and `result_summary.json`.

The report draft already includes:

- dataset splits and environment;
- Task 1 classification method, settings, curves, result table, and analysis;
- Task 2 detection metrics, tracking output, line-crossing count, occlusion frames, and analysis;
- Task 3 handwritten U-Net setup, loss comparison, curves, result table, and analysis.

Local compile command:

```bash
cd tex_report
PATH=/data/yzj/texlive/2025/bin/x86_64-linux:$PATH latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Items already filled in the report:

- public GitHub repository URL: `https://github.com/zj-yao/Homework_2`;
- cloud-drive URL for final model weights: `https://drive.google.com/drive/folders/1GpObjJP_6nYOdDF8HlNkBnlWKpdh7mBl?usp=sharing`;
- direct uploaded weights archive: `https://drive.google.com/open?id=178zInJjSr-Ue0lw0Fc0YVSLJvls1ZG4E`;
- group member names, student IDs, and division of labor;

Items still worth checking before final submission:

- optional SwanLab cloud screenshots if the teacher requires platform screenshots rather than exported local curves.

## Remaining Work

1. Recompile `tex_report/main.tex` whenever report text changes.
2. Keep pushing final commits to public GitHub repository `https://github.com/zj-yao/Homework_2`.
3. Optional: replace generated local curves with SwanLab cloud screenshots if the teacher strictly requires platform screenshots.
4. Optional: revoke local rclone Google Drive credentials after submission if this is a shared machine. Config path: `~/.config/rclone/rclone.conf`.

## Important Notes

- The Kaggle token was used for dataset download and stored under `.secrets/`; do not commit secrets.
- Before publishing the repo, verify `.secrets/`, `data/`, `outputs/`, `runs/`, `swanlog/`, and model weights are not committed accidentally.
- Since the token appeared in chat, regenerate/revoke it before final public submission if possible.
