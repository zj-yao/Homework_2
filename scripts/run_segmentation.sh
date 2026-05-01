#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/segmentation/unet_ce.yaml}
if [[ $# -gt 0 ]]; then
  shift
fi

conda run -n dl_hw2 python -m src.segmentation.train --config "$CONFIG" "$@"
