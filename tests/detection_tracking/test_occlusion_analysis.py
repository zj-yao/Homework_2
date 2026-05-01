import numpy as np

from src.detection_tracking.occlusion_analysis import (
    consecutive_frame_indices,
    save_frame_sequence,
)


def test_consecutive_frame_indices_selects_requested_contiguous_range():
    assert consecutive_frame_indices(start_frame=12, frame_count=4) == [12, 13, 14, 15]


def test_consecutive_frame_indices_clamps_to_total_frames_when_provided():
    assert consecutive_frame_indices(start_frame=8, frame_count=4, total_frames=10) == [6, 7, 8, 9]


def test_save_frame_sequence_writes_numbered_image_files(tmp_path):
    frames = [
        np.full((8, 8, 3), fill_value=0, dtype=np.uint8),
        np.full((8, 8, 3), fill_value=255, dtype=np.uint8),
    ]

    saved_paths = save_frame_sequence(frames, tmp_path, start_index=42, prefix="occ")

    assert [path.name for path in saved_paths] == ["occ_000042.jpg", "occ_000043.jpg"]
    assert all(path.exists() for path in saved_paths)
