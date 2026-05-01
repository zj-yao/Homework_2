#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/outputs/enhanced_training/logs"
STATUS_FILE="$ROOT/outputs/enhanced_training/status.tsv"
CURRENT_FILE="$ROOT/outputs/enhanced_training/current_step.txt"
SUMMARY_FILE="$ROOT/outputs/enhanced_training/summary.txt"
YOLOV8S_PROJECT="$ROOT/outputs/enhanced/detection_tracking/yolov8s_vehicle"

mkdir -p "$LOG_DIR"

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
printf 'Enhanced YOLOv8s remaining queue started at %s\n' "$(timestamp)" >> "$SUMMARY_FILE"
printf '%s\tRESUME_START\tenhanced_yolov8s_remaining\n' "$(timestamp)" | tee -a "$STATUS_FILE"

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
printf 'Enhanced YOLOv8s remaining queue completed at %s\n' "$(timestamp)" >> "$SUMMARY_FILE"
printf '%s\tALL_DONE\tenhanced_yolov8s_remaining\n' "$(timestamp)" | tee -a "$STATUS_FILE"
