import torch

from pmh import SigmaTaskConfig, eigengap_ratio, estimate_from_config, estimate_sigma_task


def test_d2_isotropic():
    s = estimate_sigma_task(dim=10, noise_level=0.2, method="D2")
    assert s.shape == (10, 10)
    assert torch.allclose(s, 0.04 * torch.eye(10), atol=1e-5)


def test_d4_domain_rank():
    src = torch.randn(50, 8)
    tgt = src + 0.3 * torch.randn(50, 8)
    s = estimate_sigma_task(src, tgt, method="D4", rank=3)
    assert s.shape == (8, 8)
    assert torch.all(torch.linalg.eigvalsh(s) >= -1e-5)


def test_d1_requires_rank():
    src, tgt = torch.randn(20, 4), torch.randn(20, 4)
    y = torch.randint(0, 3, (20,))
    s = estimate_sigma_task(src, y, tgt, y, method="D1", rank=2)
    assert s.shape == (4, 4)


def test_d3_aug_modes():
    modes = torch.randn(5, 12)
    s = estimate_sigma_task(modes, method="D3")
    assert s.shape == (12, 12)


def test_d3_estimate_from_config_kwarg_only():
    modes = torch.randn(3, 8)
    artifact = estimate_from_config(
        SigmaTaskConfig.for_augmentation(),
        aug_deltas=modes,
    )
    assert artifact.method == "D3"
    assert artifact.sigma.shape == (8, 8)


def test_d7_embedding_deltas():
    deltas = torch.randn(30, 16)
    s = estimate_sigma_task(deltas, method="D7", rank=8)
    assert s.shape == (16, 16)


def test_eigengap():
    cov = torch.diag(torch.tensor([5.0, 4.0, 0.5, 0.1]))
    g = eigengap_ratio(cov, rank=2)
    assert g == 4.0 / 0.5
