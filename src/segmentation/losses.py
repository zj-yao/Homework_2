from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss with optional ignored label support."""

    def __init__(
        self,
        num_classes: int | None = None,
        ignore_index: int | None = None,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError("logits must have shape (N, C, H, W)")
        if targets.ndim == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        if targets.ndim != 3:
            raise ValueError("targets must have shape (N, H, W)")

        num_classes = self.num_classes or logits.size(1)
        if logits.size(1) != num_classes:
            raise ValueError("logits channel count must match num_classes")

        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        safe_targets = targets
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            safe_targets = targets.masked_fill(~valid_mask, 0)

        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        probabilities = F.softmax(logits, dim=1)
        target_one_hot = F.one_hot(safe_targets.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).to(dtype=probabilities.dtype)
        valid_mask = valid_mask.unsqueeze(1).to(dtype=probabilities.dtype)

        probabilities = probabilities * valid_mask
        target_one_hot = target_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * target_one_hot, dim=dims)
        cardinality = torch.sum(probabilities + target_one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return 1.0 - dice.mean()


@dataclass
class LossWeights:
    ce: float = 1.0
    dice: float = 1.0


class CombinedLoss(nn.Module):
    """Weighted sum of cross entropy and Dice loss."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        if ce_weight < 0 or dice_weight < 0:
            raise ValueError("loss weights must be non-negative")
        self.weights = LossWeights(ce=ce_weight, dice=dice_weight)
        self.cross_entropy = (
            nn.CrossEntropyLoss()
            if ignore_index is None
            else nn.CrossEntropyLoss(ignore_index=ignore_index)
        )
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = logits.sum() * 0.0
        if self.weights.ce:
            loss = loss + self.weights.ce * self.cross_entropy(logits, targets)
        if self.weights.dice:
            loss = loss + self.weights.dice * self.dice(logits, targets)
        return loss


def build_loss(
    name: str,
    num_classes: int,
    ignore_index: int | None = None,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized in {"ce", "cross_entropy"}:
        return (
            nn.CrossEntropyLoss()
            if ignore_index is None
            else nn.CrossEntropyLoss(ignore_index=ignore_index)
        )
    if normalized == "dice":
        return DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    if normalized in {"ce_dice", "cross_entropy_dice", "combined"}:
        return CombinedLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
        )
    raise ValueError(f"Unsupported segmentation loss: {name}")
