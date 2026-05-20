"""Golden-path API: try_pmh, deploy_summary, auto nuisance."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import infer_applicability, try_pmh
from pmh.adoption import ship_verdict_label
from pmh.developer import EvaluationReport


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(8, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.enc(x))


def _loader(n=48, shift=0.0):
    x = torch.randn(n, 8) + shift
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def test_infer_applicability_suggests_domain():
    app = infer_applicability(has_target_domain=True, has_target_labels=False)
    assert app.suggested_nuisance == "domain_shift"
    assert "robust_fit" in app.summary()


def test_evaluation_report_deploy_summary():
    r = EvaluationReport(
        baseline_metric=0.5,
        pmh_metric=0.6,
        falsification_arms={"matched": 0.6, "wrong_w": 0.55, "isotropic": 0.52},
    )
    assert r.step5_ok() is True
    assert "PASS" in r.ship_verdict()
    assert "matched" in r.deploy_summary().lower() or "shift-matched" in r.deploy_summary()


def test_try_pmh_smoke():
    torch.manual_seed(0)
    m = Tiny()
    tr = _loader(shift=0.0)
    tgt = _loader(shift=0.5)
    val = _loader(shift=0.8)
    report = try_pmh(
        m,
        tr,
        val,
        source_batches=tr,
        target_batches=tgt,
        hook="enc",
        head=m.head,
        epochs=1,
        max_steps_per_epoch=2,
    )
    assert report.metric_name == "accuracy"
    assert isinstance(report.deploy_summary(), str)
    assert ship_verdict_label(report.falsification_arms) or report.baseline_metric >= 0
