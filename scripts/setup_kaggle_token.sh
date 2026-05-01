#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_TOKEN="${PROJECT_ROOT}/.secrets/kaggle/access_token"
TARGET_DIR="${HOME}/.kaggle"
TARGET_TOKEN="${TARGET_DIR}/access_token"

if [[ ! -f "${SOURCE_TOKEN}" ]]; then
  echo "Missing local Kaggle token: ${SOURCE_TOKEN}" >&2
  echo "Create it first, then rerun this script." >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"
install -m 600 "${SOURCE_TOKEN}" "${TARGET_TOKEN}"
echo "Kaggle access token installed at ${TARGET_TOKEN}"
