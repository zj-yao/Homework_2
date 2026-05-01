import torch

from src.segmentation.unet import UNet


def test_unet_outputs_class_logits_at_input_resolution():
    model = UNet(in_channels=3, num_classes=7, base_channels=8)
    images = torch.randn(2, 3, 65, 71)

    logits = model(images)

    assert logits.shape == (2, 7, 65, 71)
    assert logits.requires_grad
