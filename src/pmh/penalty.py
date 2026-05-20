"""PMH penalties: Hutchinson-style probes and feature-difference variant."""

from __future__ import annotations

from collections.abc import Callable

import torch

from pmh._tensor import ensure_psd

Encoder = Callable[[torch.Tensor], torch.Tensor]


def pmh_penalty(
    encoder: Encoder,
    x: torch.Tensor,
    sigma: torch.Tensor,
    *,
    n_probes: int = 4,
    shrinkage: float = 1e-6,
) -> torch.Tensor:
    """Estimate Tr(J_phi^T J_phi Sigma) with stochastic probes.

    **Same-space inputs (paper recipe).** When ``x`` already lives in representation
    space (dim = ``sigma.shape[0]``), perturbs ``x + z L^T`` and measures
    ``||encoder(x') - encoder(x)||^2``.  For training on representations ``h``,
    use ``encoder = lambda t: t`` or ``pmh_penalty_on_rep(h, sigma)``.

    **Different input / representation dims.** Use ``pmh_penalty_on_rep`` after
    ``h = backbone(x)``, or ``pmh_penalty_feature_diff`` for CNN layer features.
    """
    phi0 = encoder(x)
    d = phi0.shape[-1]
    if sigma.shape != (d, d):
        raise ValueError(f"sigma must be ({d}, {d}), got {tuple(sigma.shape)}")

    sigma = ensure_psd(sigma.to(device=phi0.device, dtype=phi0.dtype), shrinkage=shrinkage)
    L = torch.linalg.cholesky(sigma)

    if x.shape[-1] != d:
        return _pmh_jvp_hutchinson(encoder, x, L, phi0, n_probes=n_probes)

    acc = torch.zeros((), device=phi0.device, dtype=phi0.dtype)
    for _ in range(n_probes):
        z = torch.randn(x.shape[0], d, device=phi0.device, dtype=phi0.dtype)
        x_pert = x + z @ L.T
        phi1 = encoder(x_pert)
        acc = acc + (phi1 - phi0).pow(2).sum(dim=-1).mean()
    return acc / max(n_probes, 1)


def pmh_penalty_on_rep(
    h: torch.Tensor,
    sigma: torch.Tensor,
    *,
    n_probes: int = 4,
    shrinkage: float = 1e-6,
) -> torch.Tensor:
    """PMH on representation batch ``h`` ([B, d]) with ``Sigma`` in R^{d x d}."""
    return pmh_penalty(lambda t: t, h, sigma, n_probes=n_probes, shrinkage=shrinkage)


def _pmh_jvp_hutchinson(
    encoder: Encoder,
    x: torch.Tensor,
    L: torch.Tensor,
    phi0: torch.Tensor,
    *,
    n_probes: int,
) -> torch.Tensor:
    """Hutchinson via JVP: average || J (L z) ||^2 over probes."""
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    d = phi0.shape[-1]
    acc = torch.zeros((), device=phi0.device, dtype=phi0.dtype)
    for _ in range(n_probes):
        z = torch.randn(d, device=phi0.device, dtype=phi0.dtype)
        v = L @ z
        _, jvp_out = torch.autograd.functional.jvp(
            encoder,
            (x,),
            (torch.zeros_like(x),),
            create_graph=True,
        )
        # Directional derivative along output direction v (cotangent pullback)
        dot = (jvp_out * v.unsqueeze(0)).sum()
        g, = torch.autograd.grad(dot, x, retain_graph=True, create_graph=True)
        jvp_in, _ = torch.autograd.functional.jvp(
            encoder,
            (x,),
            (g,),
            create_graph=True,
        )
        acc = acc + jvp_in.pow(2).sum(dim=-1).mean()
    return acc / max(n_probes, 1)


def pmh_penalty_feature_diff(
    feats_clean: torch.Tensor,
    feats_noisy: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Layer-wise L2 feature consistency (vision / segmentation blocks)."""

    def _pool(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4:
            return t.mean(dim=(2, 3))
        return t

    a = _pool(feats_clean.float())
    b = _pool(feats_noisy.float())
    if normalize:
        a = torch.nn.functional.normalize(a, dim=1)
        b = torch.nn.functional.normalize(b, dim=1)
    return (a - b).pow(2).sum(dim=1).mean()


def cap_pmh_term(
    pmh_term: torch.Tensor,
    task_loss: torch.Tensor,
    *,
    cap_ratio: float = 0.3,
    basis: str = "task",
) -> torch.Tensor:
    """Cap PMH relative to task loss (paper cap proposition).

    **Recommended:** ``basis='task'`` — PMH term ≤ ``cap_ratio × task_loss``
    (e.g. ``cap_ratio=0.25`` → PMH is at most 25% of task loss).
    """
    if cap_ratio <= 0:
        return pmh_term
    lt = task_loss.detach().float()
    pt = pmh_term.float()
    r = float(cap_ratio)
    if basis == "task":
        cap = r * lt
    elif basis == "total":
        cap = (r / max(1.0 - r, 1e-8)) * lt
    else:
        raise ValueError("basis must be 'total' or 'task'")
    return torch.minimum(pt, cap).to(dtype=pmh_term.dtype)
