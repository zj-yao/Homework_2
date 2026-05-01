from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .line_counter import LineCrossingCounter, LineSegment


def box_center(xyxy: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _tensor_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _draw_label(frame: Any, xyxy: list[float], label: str, color: tuple[int, int, int]) -> None:
    import cv2

    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def annotate_result(frame: Any, result: Any, counter: LineCrossingCounter | None = None) -> Any:
    """Annotate one Ultralytics result frame with boxes, labels, IDs, and optional count."""

    import cv2

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return frame

    xyxys = _tensor_to_list(getattr(boxes, "xyxy", []))
    confs = _tensor_to_list(getattr(boxes, "conf", []))
    classes = _tensor_to_list(getattr(boxes, "cls", []))
    ids = _tensor_to_list(getattr(boxes, "id", None))
    if not ids:
        ids = [None] * len(xyxys)

    names = getattr(result, "names", {}) or {}
    for xyxy, conf, cls_idx, track_id in zip(xyxys, confs, classes, ids):
        class_id = int(cls_idx)
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
        label = f"{class_name} {float(conf):.2f}"
        if track_id is not None:
            label = f"ID {int(track_id)} {label}"
        _draw_label(frame, [float(value) for value in xyxy], label, (40, 220, 40))

        if counter is not None and track_id is not None:
            counter.update(int(track_id), box_center(xyxy))

    if counter is not None:
        cv2.line(
            frame,
            (int(counter.line.start[0]), int(counter.line.start[1])),
            (int(counter.line.end[0]), int(counter.line.end[1])),
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Count: {counter.total}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    return frame


def track_video(
    model_path: str | Path,
    video_path: str | Path,
    output_path: str | Path,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    device: str | None = None,
    line: LineSegment | None = None,
    summary_path: str | Path | None = None,
) -> int:
    """Run YOLO tracking on a video and save an annotated output video."""

    import cv2
    from ultralytics import YOLO

    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(f"Video does not exist: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise OSError(f"Could not create output video: {output}")

    model = YOLO(str(model_path))
    counter = LineCrossingCounter(line) if line is not None else None
    frame_count = 0
    try:
        results = model.track(
            source=str(source),
            stream=True,
            persist=True,
            tracker=tracker,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        for result in results:
            frame = result.orig_img.copy()
            writer.write(annotate_result(frame, result, counter))
            frame_count += 1
    finally:
        writer.release()
    if summary_path is not None:
        summary = {
            "video": str(source),
            "output": str(output),
            "frames": frame_count,
            "tracker": tracker,
            "conf": float(conf),
            "iou": float(iou),
            "imgsz": int(imgsz),
            "line": None,
            "count": None,
        }
        if counter is not None:
            summary["line"] = {
                "start": [float(counter.line.start[0]), float(counter.line.start[1])],
                "end": [float(counter.line.end[0]), float(counter.line.end[1])],
            }
            summary["count"] = counter.total
            summary["crossed_ids"] = sorted(int(track_id) for track_id in counter.crossed_ids)
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return frame_count


def _parse_line(values: list[float] | None) -> LineSegment | None:
    if not values:
        return None
    if len(values) != 4:
        raise ValueError("--line expects four numbers: x1 y1 x2 y2")
    return LineSegment((values[0], values[1]), (values[2], values[3]))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track vehicles in a video with YOLOv8.")
    parser.add_argument("--model", required=True, help="Trained YOLO checkpoint")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Annotated output video path")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker YAML")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device")
    parser.add_argument("--line", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--summary", help="Optional JSON path for frame count and line-crossing count")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    line = _parse_line(args.line)
    frames = track_video(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        line=line,
        summary_path=args.summary,
    )
    print(f"Annotated {frames} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
