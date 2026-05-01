import pytest
import torch

from src.segmentation.metrics import mean_iou


def test_mean_iou_matches_manual_multiclass_calculation():
    targets = torch.tensor([[[0, 1, 1], [0, 2, 2]]], dtype=torch.long)
    predictions = torch.tensor([[[0, 1, 0], [0, 2, 1]]], dtype=torch.long)

    miou = mean_iou(predictions, targets, num_classes=3)

    assert miou == pytest.approx(0.5)


def test_mean_iou_accepts_logits_and_ignores_absent_classes():
    targets = torch.tensor([[[0, 1], [255, 255]]], dtype=torch.long)
    class_predictions = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
    logits = torch.full((1, 3, 2, 2), -5.0)
    logits.scatter_(1, class_predictions.unsqueeze(1), 5.0)

    miou = mean_iou(logits, targets, num_classes=3, ignore_index=255)

    assert miou == pytest.approx(1.0)
