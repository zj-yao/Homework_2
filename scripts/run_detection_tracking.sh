#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-help}"
shift || true

case "${MODE}" in
  train)
    conda run -n dl_hw2 python -m src.detection_tracking.train_yolo "$@"
    ;;
  eval)
    conda run -n dl_hw2 python -m src.detection_tracking.evaluate_yolo "$@"
    ;;
  track)
    conda run -n dl_hw2 python -m src.detection_tracking.track_video "$@"
    ;;
  occlusion)
    conda run -n dl_hw2 python -m src.detection_tracking.occlusion_analysis "$@"
    ;;
  test)
    conda run -n dl_hw2 pytest tests/detection_tracking -q "$@"
    ;;
  help|--help|-h)
    cat <<'USAGE'
Usage:
  scripts/run_detection_tracking.sh train [train_yolo args]
  scripts/run_detection_tracking.sh eval --model outputs/.../best.pt --data-yaml configs/detection/road_vehicle_data.yaml
  scripts/run_detection_tracking.sh track --model outputs/.../best.pt --video data/videos/input.mp4 --output outputs/detection_tracking/tracked_video/output.mp4 --line x1 y1 x2 y2
  scripts/run_detection_tracking.sh occlusion --video data/videos/input.mp4 --output-dir outputs/detection_tracking/occlusion_frames --start-frame 120
  scripts/run_detection_tracking.sh test
USAGE
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
