#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/classification/baseline_resnet18.yaml}"
CHECKPOINT_PATH="${2:-}"

conda run -n dl_hw2 python -m src.classification.train --config "${CONFIG_PATH}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(
    CONFIG_PATH="${CONFIG_PATH}" conda run -n dl_hw2 python -c \
      'import os, yaml; from pathlib import Path; config = yaml.safe_load(open(os.environ["CONFIG_PATH"], encoding="utf-8")); print(Path(config.get("output_dir", "outputs/classification/run")) / "best.pt")'
  )"
fi

conda run -n dl_hw2 python -m src.classification.evaluate \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --split test
