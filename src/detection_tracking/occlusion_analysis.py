from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def consecutive_frame_indices(
    start_frame: int,
    frame_count: int = 4,
    total_frames: int | None = None,
) -> list[int]:
    """Return a contiguous frame index range, clamped to the video length if known."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if total_frames is not None and total_frames <= 0:
        raise ValueError("total_frames must be positive when provided")

    start = int(start_frame)
    count = int(frame_count)
    if total_frames is not None and start + count > total_frames:
        start = max(0, total_frames - count)
        count = min(count, total_frames)
    return list(range(start, start + count))


def save_frame_sequence(
    frames: Iterable[np.ndarray],
    output_dir: str | Path,
    start_index: int = 0,
    prefix: str = "frame",
    image_ext: str = ".jpg",
) -> list[Path]:
    """Save an iterable of image frames with stable frame-index filenames."""

    import cv2

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for offset, frame in enumerate(frames):
        frame_index = start_index + offset
        path = output / f"{prefix}_{frame_index:06d}{image_ext}"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"Failed to write frame image: {path}")
        saved_paths.append(path)
    return saved_paths


def extract_consecutive_frames(
    video_path: str | Path,
    output_dir: str | Path,
    start_frame: int,
    frame_count: int = 4,
    prefix: str = "occlusion",
) -> list[Path]:
    """Extract consecutive frames from a video for occlusion/ID-switch analysis."""

    import cv2

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video does not exist: {video}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        indices = consecutive_frame_indices(start_frame, frame_count, total_frames)
        if not indices:
            return []

        capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        frames: list[np.ndarray] = []
        for expected_index in indices:
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"Could not read frame {expected_index} from {video}")
            frames.append(frame)
        return save_frame_sequence(frames, output_dir, start_index=indices[0], prefix=prefix)
    finally:
        capture.release()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract 3-4 consecutive occlusion frames.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output-dir", required=True, help="Directory for extracted frames")
    parser.add_argument("--start-frame", type=int, required=True, help="First occlusion frame index")
    parser.add_argument("--frame-count", type=int, default=4, help="Number of frames to extract")
    parser.add_argument("--prefix", default="occlusion", help="Output filename prefix")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    saved = extract_consecutive_frames(
        video_path=args.video,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        frame_count=args.frame_count,
        prefix=args.prefix,
    )
    print("Saved occlusion frames:")
    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
