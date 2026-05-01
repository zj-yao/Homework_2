from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _class_predictions(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if predictions.ndim == targets.ndim + 1:
        return predictions.argmax(dim=1)
    return predictions


def intersection_and_union(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = _class_predictions(predictions, targets).detach().cpu().long()
    targets = targets.detach().cpu().long()
    if targets.ndim == 4 and targets.size(1) == 1:
        targets = targets.squeeze(1)

    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    if ignore_index is not None:
        keep = targets != ignore_index
        predictions = predictions[keep]
        targets = targets[keep]

    intersections = torch.zeros(num_classes, dtype=torch.float64)
    unions = torch.zeros(num_classes, dtype=torch.float64)
    for class_id in range(num_classes):
        pred_mask = predictions == class_id
        target_mask = targets == class_id
        intersections[class_id] = torch.logical_and(pred_mask, target_mask).sum()
        unions[class_id] = torch.logical_or(pred_mask, target_mask).sum()
    return intersections, unions


def per_class_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    intersections, unions = intersection_and_union(
        predictions=predictions,
        targets=targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    ious = torch.full((num_classes,), float("nan"), dtype=torch.float64)
    present = unions > 0
    ious[present] = intersections[present] / unions[present]
    return ious


def mean_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> float:
    ious = per_class_iou(
        predictions=predictions,
        targets=targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    present = ~torch.isnan(ious)
    if present.sum() == 0:
        return 0.0
    return float(ious[present].mean().item())


@dataclass
class SegmentationMetricTracker:
    num_classes: int
    ignore_index: int | None = None
    intersections: torch.Tensor = field(init=False)
    unions: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.intersections = torch.zeros(self.num_classes, dtype=torch.float64)
        self.unions = torch.zeros(self.num_classes, dtype=torch.float64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        intersections, unions = intersection_and_union(
            predictions=predictions,
            targets=targets,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        self.intersections += intersections
        self.unions += unions

    def compute(self) -> dict[str, object]:
        ious = torch.full((self.num_classes,), float("nan"), dtype=torch.float64)
        present = self.unions > 0
        ious[present] = self.intersections[present] / self.unions[present]
        miou = float(ious[present].mean().item()) if present.any() else 0.0
        return {
            "miou": miou,
            "per_class_iou": [
                None if torch.isnan(value) else float(value.item()) for value in ious
            ],
        }
