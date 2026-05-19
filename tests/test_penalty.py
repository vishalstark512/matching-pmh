import torch

from pmh import cap_pmh_term, pmh_penalty, pmh_penalty_feature_diff, wrong_W_projector


def test_pmh_zero_when_encoder_constant():
    d = 8
    sigma = torch.eye(d)
    x = torch.randn(4, d)

    def enc(_x):
        return torch.zeros(4, d)

    assert pmh_penalty(enc, x, sigma).item() < 1e-6


def test_pmh_positive_with_movement():
    d = 8
    sigma = torch.eye(d)
    x = torch.randn(4, d)
    lin = torch.nn.Linear(d, d, bias=False)

    def enc(inp):
        return lin(inp)

    assert pmh_penalty(enc, x, sigma, n_probes=8).item() > 0


def test_feature_diff_zero_identical():
    a = torch.randn(5, 16)
    assert pmh_penalty_feature_diff(a, a.clone()).item() < 1e-6


def test_cap_limits_pmh():
    task = torch.tensor(1.0)
    pmh = torch.tensor(10.0)
    capped = cap_pmh_term(pmh, task, cap_ratio=0.3, basis="total")
    assert capped.item() < pmh.item()


def test_wrong_W_shape():
    U = wrong_W_projector(20, 5)
    assert U.shape == (20, 5)
    P = U @ U.T
    assert torch.allclose(P, P.T, atol=1e-5)
