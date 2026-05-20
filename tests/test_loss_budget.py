"""PMH vs task loss budgeting."""

from __future__ import annotations

import torch

from pmh.config import PMHConfig
from pmh.loss_budget import budget_pmh_to_task_loss, suggest_pmh_weight
from pmh.training import PMHLoss


def test_budget_caps_pmh_to_max_ratio():
    cfg = PMHConfig(pmh_max_task_ratio=0.30, cap_basis="task")
    task = torch.tensor(1.0)
    pmh = torch.tensor(0.9)
    applied, diag = budget_pmh_to_task_loss(pmh, task, cfg)
    assert float(applied) <= 0.30 + 1e-5
    assert diag.capped is True
    assert diag.pmh_task_ratio <= 0.30 + 1e-5


def test_budget_passes_through_in_band():
    cfg = PMHConfig.golden_path()
    task = torch.tensor(2.0)
    pmh = torch.tensor(0.2)
    applied, diag = budget_pmh_to_task_loss(pmh, task, cfg)
    assert abs(float(applied) - 0.2) < 1e-6
    assert not diag.capped
    assert 0.05 <= diag.pmh_task_ratio <= 0.30


def test_pmh_loss_capped_total_uses_budget():
    sigma = torch.eye(4) * 0.5
    pmh = PMHLoss(sigma, PMHConfig.golden_path())
    task = torch.tensor(1.0, requires_grad=True)
    h = torch.randn(8, 4, requires_grad=True)
    total, raw = pmh.capped_total(task, h)
    assert pmh.last_budget is not None
    assert pmh.last_budget.pmh_task_ratio <= pmh.config.pmh_max_task_ratio + 1e-5
    assert float(total) >= float(task)


def test_suggest_pmh_weight_positive():
    enc = torch.nn.Linear(8, 4)
    x = torch.randn(4, 8)
    sigma = torch.eye(4) * 0.2
    task = torch.tensor(1.5)
    w = suggest_pmh_weight(enc, x, sigma, task, target_ratio=0.15)
    assert 1e-4 < w < 2.0
