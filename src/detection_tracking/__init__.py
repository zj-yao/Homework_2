"""Task 2 helpers for vehicle detection, tracking, and counting."""

__all__ = [
    "LineCrossingCounter",
    "LineSegment",
    "YoloDatasetConfig",
    "validate_yolo_dataset",
    "write_yolo_data_yaml",
]


def __getattr__(name: str):
    if name in {"LineCrossingCounter", "LineSegment"}:
        from .line_counter import LineCrossingCounter, LineSegment

        return {"LineCrossingCounter": LineCrossingCounter, "LineSegment": LineSegment}[name]
    if name in {"YoloDatasetConfig", "validate_yolo_dataset", "write_yolo_data_yaml"}:
        from .prepare_data import YoloDatasetConfig, validate_yolo_dataset, write_yolo_data_yaml

        return {
            "YoloDatasetConfig": YoloDatasetConfig,
            "validate_yolo_dataset": validate_yolo_dataset,
            "write_yolo_data_yaml": write_yolo_data_yaml,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
