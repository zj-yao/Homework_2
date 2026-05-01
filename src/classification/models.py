from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34
from torchvision.models.resnet import BasicBlock, ResNet


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class SEBasicBlock(BasicBlock):
    def __init__(self, *args, se_reduction: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.se = SEBlock(self.bn2.num_features, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def build_model(model_name: str, num_classes: int = 102, pretrained: bool = True) -> nn.Module:
    name = model_name.lower()
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    elif name == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
    elif name == "resnet18_se":
        model = _build_resnet18_se(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported classification model: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def create_param_groups(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
) -> list[dict[str, object]]:
    backbone_params: list[nn.Parameter] = []
    classifier_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_classifier_parameter(name):
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    if not classifier_params:
        raise ValueError("No classifier parameters found; expected names like fc.*")

    return [
        {"name": "backbone", "params": backbone_params, "lr": backbone_lr},
        {"name": "classifier", "params": classifier_params, "lr": head_lr},
    ]


def freeze_parameters(parameters: Iterable[nn.Parameter]) -> None:
    for parameter in parameters:
        parameter.requires_grad = False


def _build_resnet18_se(pretrained: bool) -> ResNet:
    model = ResNet(SEBasicBlock, [2, 2, 2, 2])
    if pretrained:
        state_dict = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
        model.load_state_dict(state_dict, strict=False)
    return model


def _is_classifier_parameter(name: str) -> bool:
    return name.startswith("fc.") or name.startswith("classifier.") or name.startswith("head.")
