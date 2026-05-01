#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/outputs/formal_training/logs"
STATUS_FILE="$ROOT/outputs/formal_training/status.tsv"
CURRENT_FILE="$ROOT/outputs/formal_training/current_step.txt"
SUMMARY_FILE="$ROOT/outputs/formal_training/summary.txt"
DETECTION_PROJECT="$ROOT/outputs/detection_tracking/yolov8_vehicle"

mkdir -p "$LOG_DIR"
: > "$STATUS_FILE"

timestamp() {
  date -Is
}

run_step() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${name}.log"

  printf '%s\tSTART\t%s\n' "$(timestamp)" "$name" | tee -a "$STATUS_FILE"
  printf '%s\n' "$name" > "$CURRENT_FILE"
  {
    printf '[%s] START %s\n' "$(timestamp)" "$name"
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n\n'
  } > "$log_file"

  "$@" >> "$log_file" 2>&1
  local code=$?

  if [ "$code" -eq 0 ]; then
    printf '%s\tDONE\t%s\n' "$(timestamp)" "$name" | tee -a "$STATUS_FILE"
  else
    printf '%s\tFAILED(%s)\t%s\n' "$(timestamp)" "$code" "$name" | tee -a "$STATUS_FILE"
    printf 'FAILED: %s, see %s\n' "$name" "$log_file" > "$SUMMARY_FILE"
    exit "$code"
  fi
}

cd "$ROOT" || exit 1
printf 'Formal training queue started at %s\n' "$(timestamp)" > "$SUMMARY_FILE"
printf 'Project root: %s\n' "$ROOT" >> "$SUMMARY_FILE"

run_step classification_resnet18_pretrained_baseline_train \
  conda run -n dl_hw2 python -m src.classification.train \
  --config configs/classification/baseline_resnet18.yaml \
  --override logging.backend=swanlab \
  --override logging.mode=offline \
  --override logging.run_name=resnet18-pretrained-baseline-formal

run_step classification_resnet18_pretrained_baseline_eval_test \
  conda run -n dl_hw2 python -m src.classification.evaluate \
  --config configs/classification/baseline_resnet18.yaml \
  --checkpoint outputs/classification/resnet18_pretrained_baseline/best.pt \
  --split test \
  --override train.device=cuda

run_step classification_resnet18_pretrained_baseline_15ep_train \
  conda run -n dl_hw2 python -m src.classification.train \
  --config configs/classification/baseline_resnet18.yaml \
  --override output_dir=outputs/classification/resnet18_pretrained_baseline_15ep \
  --override train.epochs=15 \
  --override logging.backend=swanlab \
  --override logging.mode=offline \
  --override logging.run_name=resnet18-pretrained-baseline-15ep-formal

run_step classification_resnet18_pretrained_baseline_15ep_eval_test \
  conda run -n dl_hw2 python -m src.classification.evaluate \
  --config configs/classification/baseline_resnet18.yaml \
  --checkpoint outputs/classification/resnet18_pretrained_baseline_15ep/best.pt \
  --split test \
  --override output_dir=outputs/classification/resnet18_pretrained_baseline_15ep \
  --override train.device=cuda

run_step classification_resnet18_pretrained_lr_low_15ep_train \
  conda run -n dl_hw2 python -m src.classification.train \
  --config configs/classification/baseline_resnet18.yaml \
  --override output_dir=outputs/classification/resnet18_pretrained_lr_low_15ep \
  --override train.epochs=15 \
  --override train.backbone_lr=0.00005 \
  --override train.head_lr=0.0005 \
  --override logging.backend=swanlab \
  --override logging.mode=offline \
  --override logging.run_name=resnet18-pretrained-lr-low-15ep-formal

run_step classification_resnet18_pretrained_lr_low_15ep_eval_test \
  conda run -n dl_hw2 python -m src.classification.evaluate \
  --config configs/classification/baseline_resnet18.yaml \
  --checkpoint outputs/classification/resnet18_pretrained_lr_low_15ep/best.pt \
  --split test \
  --override output_dir=outputs/classification/resnet18_pretrained_lr_low_15ep \
  --override train.device=cuda

run_step classification_resnet18_pretrained_lr_high_15ep_train \
  conda run -n dl_hw2 python -m src.classification.train \
  --config configs/classification/baseline_resnet18.yaml \
  --override output_dir=outputs/classification/resnet18_pretrained_lr_high_15ep \
  --override train.epochs=15 \
  --override train.backbone_lr=0.0002 \
  --override train.head_lr=0.002 \
  --override logging.backend=swanlab \
  --override logging.mode=offline \
  --override logging.run_name=resnet18-pretrained-lr-high-15ep-formal

run_step classification_resnet18_pretrained_lr_high_15ep_eval_test \
  conda run -n dl_hw2 python -m src.classification.evaluate \
  --config configs/classification/baseline_resnet18.yaml \
  --checkpoint outputs/classification/resnet18_pretrained_lr_high_15ep/best.pt \
  --split test \
  --override output_dir=outputs/classification/resnet18_pretrained_lr_high_15ep \
  --override train.device=cuda

run_step classification_resnet18_random_init_train \
  conda run -n dl_hw2 python -m src.classification.train \
  --config configs/classification/resnet18_random_init.yaml \
  --override logging.backend=swanlab \
  --override logging.mode=offline \
  --override logging.run_name=resnet18-random-init-formal

run_step classification_resnet18_random_init_eval_test \
  conda run -n dl_hw2 python -m src.classification.evaluate \
  --config configs/classification/resnet18_random_init.yaml \
  --checkpoint outputs/classification/resnet18_random_init/best.pt \
  --split test \
  --override train.device=cuda

run_step classification_resnet18_se_block_train \
  conda run -n dl_hw2 python -m src.classification.train \
  --config configs/classification/resnet18_se.yaml \
  --override logging.backend=swanlab \
  --override logging.mode=offline \
  --override logging.run_name=resnet18-se-block-formal

run_step classification_resnet18_se_block_eval_test \
  conda run -n dl_hw2 python -m src.classification.evaluate \
  --config configs/classification/resnet18_se.yaml \
  --checkpoint outputs/classification/resnet18_se_block/best.pt \
  --split test \
  --override train.device=cuda

run_step segmentation_unet_ce_train \
  conda run -n dl_hw2 python -m src.segmentation.train \
  --config configs/segmentation/unet_ce.yaml \
  --override experiment.logger=swanlab \
  --override experiment.mode=offline

run_step segmentation_unet_ce_eval_val \
  conda run -n dl_hw2 python -m src.segmentation.evaluate \
  --config configs/segmentation/unet_ce.yaml \
  --checkpoint outputs/segmentation/unet_ce/best.pt \
  --split val \
  --override train.device=cuda

run_step segmentation_unet_dice_train \
  conda run -n dl_hw2 python -m src.segmentation.train \
  --config configs/segmentation/unet_dice.yaml \
  --override experiment.logger=swanlab \
  --override experiment.mode=offline

run_step segmentation_unet_dice_eval_val \
  conda run -n dl_hw2 python -m src.segmentation.evaluate \
  --config configs/segmentation/unet_dice.yaml \
  --checkpoint outputs/segmentation/unet_dice/best.pt \
  --split val \
  --override train.device=cuda

run_step segmentation_unet_ce_dice_train \
  conda run -n dl_hw2 python -m src.segmentation.train \
  --config configs/segmentation/unet_ce_dice.yaml \
  --override experiment.logger=swanlab \
  --override experiment.mode=offline

run_step segmentation_unet_ce_dice_eval_val \
  conda run -n dl_hw2 python -m src.segmentation.evaluate \
  --config configs/segmentation/unet_ce_dice.yaml \
  --checkpoint outputs/segmentation/unet_ce_dice/best.pt \
  --split val \
  --override train.device=cuda

run_step detection_yolov8n_vehicle_train \
  conda run -n dl_hw2 python -m src.detection_tracking.train_yolo \
  --config configs/detection/yolov8_vehicle.yaml \
  --project "$DETECTION_PROJECT" \
  --name yolov8n_vehicle \
  --device 0 \
  --workers 4

run_step detection_yolov8n_vehicle_eval_val \
  conda run -n dl_hw2 python -m src.detection_tracking.evaluate_yolo \
  --model "$DETECTION_PROJECT/yolov8n_vehicle/weights/best.pt" \
  --data-yaml configs/detection/road_vehicle_data.yaml \
  --imgsz 640 \
  --batch 8 \
  --device 0

printf 'ALL_DONE\n' > "$CURRENT_FILE"
printf 'Formal training queue completed at %s\n' "$(timestamp)" >> "$SUMMARY_FILE"
printf '%s\tALL_DONE\tformal_training_queue\n' "$(timestamp)" | tee -a "$STATUS_FILE"
