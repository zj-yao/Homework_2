#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/outputs/enhanced_training/logs"
STATUS_FILE="$ROOT/outputs/enhanced_training/status.tsv"
CURRENT_FILE="$ROOT/outputs/enhanced_training/current_step.txt"
SUMMARY_FILE="$ROOT/outputs/enhanced_training/summary.txt"
YOLOV8S_MODEL="/data/yzj/homework2_data/pretrained/yolov8s.pt"
YOLOV8S_PROJECT="$ROOT/outputs/enhanced/detection_tracking/yolov8s_vehicle"

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
printf 'Enhanced training queue started at %s\n' "$(timestamp)" > "$SUMMARY_FILE"
printf 'Project root: %s\n' "$ROOT" >> "$SUMMARY_FILE"

run_step classification_resnet34_pretrained_60ep_train \
  conda run -n dl_hw2 python -m src.classification.train \
  --config configs/classification/resnet34_pretrained_60ep.yaml \
  --override logging.backend=swanlab \
  --override logging.mode=offline \
  --override logging.run_name=resnet34-pretrained-60ep-enhanced

run_step classification_resnet34_pretrained_60ep_eval_test \
  conda run -n dl_hw2 python -m src.classification.evaluate \
  --config configs/classification/resnet34_pretrained_60ep.yaml \
  --checkpoint outputs/enhanced/classification/resnet34_pretrained_60ep/best.pt \
  --split test \
  --override train.device=cuda

run_step segmentation_unet64_ce_dice_100ep_train \
  conda run -n dl_hw2 python -m src.segmentation.train \
  --config configs/segmentation/unet64_ce_dice_100ep.yaml \
  --override experiment.logger=swanlab \
  --override experiment.mode=offline

run_step segmentation_unet64_ce_dice_100ep_eval_val \
  conda run -n dl_hw2 python -m src.segmentation.evaluate \
  --config configs/segmentation/unet64_ce_dice_100ep.yaml \
  --checkpoint outputs/enhanced/segmentation/unet64_ce_dice_100ep/best.pt \
  --split val \
  --override train.device=cuda

run_step prepare_yolov8s_pretrained \
  bash -lc "mkdir -p /data/yzj/homework2_data/pretrained && if [ ! -f '$YOLOV8S_MODEL' ]; then cd /data/yzj/homework2_data/pretrained && conda run -n dl_hw2 python -c 'from ultralytics import YOLO; YOLO(\"yolov8s.pt\")'; fi && test -f '$YOLOV8S_MODEL'"

run_step detection_yolov8s_vehicle_100ep_train \
  conda run -n dl_hw2 python -m src.detection_tracking.train_yolo \
  --config configs/detection/yolov8s_vehicle.yaml \
  --project "$YOLOV8S_PROJECT" \
  --name yolov8s_vehicle_100ep \
  --device 0 \
  --workers 4

run_step detection_yolov8s_vehicle_100ep_eval_val \
  conda run -n dl_hw2 python -m src.detection_tracking.evaluate_yolo \
  --model "$YOLOV8S_PROJECT/yolov8s_vehicle_100ep/weights/best.pt" \
  --data-yaml configs/detection/road_vehicle_data.yaml \
  --imgsz 640 \
  --batch 8 \
  --device 0

printf 'ALL_DONE\n' > "$CURRENT_FILE"
printf 'Enhanced training queue completed at %s\n' "$(timestamp)" >> "$SUMMARY_FILE"
printf '%s\tALL_DONE\tenhanced_training_queue\n' "$(timestamp)" | tee -a "$STATUS_FILE"
