from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path

def write_history_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    row_list = list(rows)
    fieldnames = sorted({key for row in row_list for key in row})

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)
    return output_path
