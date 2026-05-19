import torch
import torch.nn as nn

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, estimate_from_config


def test_pmh_loss_forward():
    est = estimate_from_config(
        SigmaTaskConfig.for_isotropic(8, 0.1),
    )
    loss_mod = PMHLoss(est, PMHConfig(weight=0.2, warmup_epochs=0))
    h = torch.randn(4, 8, requires_grad=True)
    pen = loss_mod(h)
    assert pen.ndim == 0 and pen.item() >= 0


def test_pmh_loss_capped_total():
    backbone = nn.Linear(10, 6)
    head = nn.Linear(6, 2)
    est = estimate_from_config(SigmaTaskConfig.for_isotropic(6, 0.15))
    pmh = PMHLoss(est, PMHConfig(weight=0.5, cap_ratio=0.3))
    x = torch.randn(8, 10)
    y = torch.randint(0, 2, (8,))
    h = backbone(x)
    task = nn.functional.cross_entropy(head(h), y)
    total, raw = pmh.capped_total(task, h)
    assert total.item() >= task.item()
    total.backward()
