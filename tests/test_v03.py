import torch
import torch.nn as nn
import numpy as np

from pmh.integrations import PMHCallback, train_epoch_with_pmh
from pmh import SigmaTaskConfig, PMHConfig, estimate_from_config
from pmh.numpy_api import estimate_sigma_task_numpy, gram_from_diff_numpy
from pmh.sklearn_match import MatchedSubspaceProjector
from pmh.vision import MultiLayerPMHLoss


def test_pmh_callback_step():
    est = estimate_from_config(SigmaTaskConfig.for_isotropic(8, 0.1))
    backbone = nn.Linear(10, 8)
    head = nn.Linear(8, 2)
    cb = PMHCallback.from_artifact(
        est,
        encoder=backbone,
        head=head,
    )
    cb.on_epoch_start(1)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    loss, info = cb.training_step((x, y))
    assert loss.requires_grad
    assert info.total_loss >= info.task_loss


def test_train_epoch_with_pmh():
    est = estimate_from_config(SigmaTaskConfig.for_isotropic(6, 0.1))
    model = nn.Sequential(nn.Linear(5, 6), nn.ReLU(), nn.Linear(6, 2))
    cb = PMHCallback.from_artifact(est, encoder=lambda x: model[0:2](x), head=lambda h: model[2](h))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    data = [(torch.randn(8, 5), torch.randint(0, 2, (8,))) for _ in range(3)]
    stats = train_epoch_with_pmh(model, cb, data, opt, epoch=2)
    assert stats["n_steps"] == 3.0


def test_numpy_d1_projector():
    rng = np.random.default_rng(0)
    n, d = 80, 20
    x_s = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 5, n)
    x_t = x_s + 0.5
    proj = MatchedSubspaceProjector(rank=4, seed=0).fit(x_s, y, x_t, y)
    z = proj.transform(x_s)
    assert z.shape == x_s.shape


def test_multilayer_pmh():
    pmh = MultiLayerPMHLoss(("a", "b"), PMHConfig(weight=0.5))
    fc = torch.randn(4, 8)
    fn = torch.randn(4, 8)
    loss = pmh({"a": fc, "b": fc}, {"a": fn, "b": fn})
    assert loss.item() >= 0


def test_gram_from_diff_numpy():
    s = np.random.randn(30, 5).astype(np.float32)
    t = s + 0.1
    g = gram_from_diff_numpy(s, t)
    assert g.shape == (5, 5)
