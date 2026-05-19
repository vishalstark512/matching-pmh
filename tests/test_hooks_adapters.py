"""Hook adapters and family detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from pmh.hooks import (
    detect_model_family,
    encoder_torchvision_resnet,
    encoder_timm,
    list_hook_families,
    resolve_hook,
    validate_representation,
)


class MiniResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer4 = nn.Conv2d(3, 8, 3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.layer4(x))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class MiniTimm(nn.Module):
    def forward_features(self, x):
        return x.mean(dim=(2, 3)) if x.dim() == 4 else x


def test_validate_4d_pool():
    h = torch.randn(2, 4, 3, 3)
    assert validate_representation(h).shape == (2, 4)


def test_resolve_hook_avgpool():
    m = MiniResNet()
    enc = resolve_hook(m, "avgpool")
    with torch.no_grad():
        feats = torch.relu(m.layer4(torch.randn(2, 3, 8, 8)))
    assert enc(feats).shape == (2, 8)


def test_encoder_timm():
    m = MiniTimm()
    assert detect_model_family(m) == "generic"
    enc = encoder_timm(m)
    assert enc(torch.randn(2, 3, 4, 4)).shape == (2, 3)


def test_list_hook_families():
    families = list_hook_families()
    assert "torchvision_resnet" in families
