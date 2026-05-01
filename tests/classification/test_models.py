import torch

from src.classification.models import build_model, create_param_groups


def test_resnet_factory_replaces_classifier_head_and_runs_forward():
    model = build_model("resnet18", num_classes=102, pretrained=False)
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 102)
    assert model.fc.out_features == 102


def test_resnet18_se_factory_adds_attention_blocks_and_runs_forward():
    model = build_model("resnet18_se", num_classes=7, pretrained=False)
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 7)
    assert any(module.__class__.__name__ == "SEBlock" for module in model.modules())


def test_create_param_groups_separates_backbone_and_head_learning_rates():
    model = build_model("resnet18", num_classes=5, pretrained=False)

    groups = create_param_groups(model, backbone_lr=1e-4, head_lr=1e-3)

    assert [group["name"] for group in groups] == ["backbone", "classifier"]
    assert [group["lr"] for group in groups] == [1e-4, 1e-3]

    grouped_ids = {id(param) for group in groups for param in group["params"]}
    trainable_ids = {id(param) for param in model.parameters() if param.requires_grad}
    assert grouped_ids == trainable_ids
    assert set(map(id, groups[0]["params"])).isdisjoint(set(map(id, groups[1]["params"])))
