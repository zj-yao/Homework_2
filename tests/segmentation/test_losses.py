import pytest
import torch
from torch import nn

from src.segmentation.losses import CombinedLoss, DiceLoss, build_loss


def test_dice_loss_is_near_zero_for_perfect_multiclass_predictions():
    targets = torch.tensor([[[0, 1], [2, 1]]], dtype=torch.long)
    logits = torch.full((1, 3, 2, 2), -10.0)
    logits.scatter_(1, targets.unsqueeze(1), 10.0)

    loss = DiceLoss(num_classes=3)(logits, targets)

    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_dice_loss_ignores_pixels_marked_with_ignore_index():
    targets = torch.tensor([[[0, 1], [255, 255]]], dtype=torch.long)
    logits_a = torch.randn(1, 2, 2, 2)
    logits_b = logits_a.clone()
    logits_b[:, :, 1, :] = torch.tensor([[[50.0, -50.0], [-50.0, 50.0]]])

    loss_a = DiceLoss(num_classes=2, ignore_index=255)(logits_a, targets)
    loss_b = DiceLoss(num_classes=2, ignore_index=255)(logits_b, targets)

    assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-6)


def test_combined_loss_matches_weighted_cross_entropy_plus_dice():
    targets = torch.tensor([[[0, 1], [2, 1]]], dtype=torch.long)
    logits = torch.randn(1, 3, 2, 2)
    ce_weight = 0.7
    dice_weight = 0.3

    combined = CombinedLoss(
        num_classes=3,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
    )
    expected = ce_weight * nn.CrossEntropyLoss()(logits, targets)
    expected = expected + dice_weight * DiceLoss(num_classes=3)(logits, targets)

    assert torch.allclose(combined(logits, targets), expected)


def test_cross_entropy_loss_supports_negative_stanford_unknown_pixels():
    targets = torch.tensor([[[0, 1], [-1, 2]]], dtype=torch.long)
    logits = torch.randn(1, 3, 2, 2)

    loss = build_loss("ce", num_classes=3, ignore_index=-1)(logits, targets)

    assert torch.isfinite(loss)
