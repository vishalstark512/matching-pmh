"""Lemma D7: style-pair Gram or PGD-delta scatter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from pmh._tensor import ensure_psd

Encoder = Callable[[Sequence[str]], torch.Tensor]


def estimate_d7(
    style_pairs: Sequence[dict[str, Any]] | torch.Tensor,
    *,
    encoder: Encoder | None = None,
    rank: int | None = 128,
    shrinkage: float = 1e-6,
) -> torch.Tensor:
    """Style-pair Gram: covariance of embedding deltas across style variants.

    Parameters
    ----------
    style_pairs :
        Either a Tensor [M, d] of embedding deltas (style - base per row), or a list of
        records with keys ``prompt``, ``content_fixed``, ``style_variants`` (dict).
    encoder :
        Required when style_pairs is JSON-like records: maps list of formatted strings
        to [B, d] embeddings.
    """
    if isinstance(style_pairs, torch.Tensor):
        deltas = style_pairs.float()
    else:
        if encoder is None:
            raise ValueError("encoder is required when style_pairs are text records")
        deltas = _embed_style_deltas(style_pairs, encoder)

    deltas = deltas - deltas.mean(0, keepdim=True)
    n = deltas.shape[0]
    sigma = (deltas.T @ deltas) / max(n, 1)
    if rank is not None:
        from pmh.estimators.d4_domain import _truncate_rank

        sigma = _truncate_rank(sigma, rank)
    return ensure_psd(sigma, shrinkage=shrinkage)


def _embed_style_deltas(pairs: Sequence[dict[str, Any]], encoder: Encoder) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for rec in pairs:
        prompt = str(rec["prompt"])
        base = str(rec.get("content_fixed") or rec.get("base", ""))
        variants = rec["style_variants"]
        if not isinstance(variants, dict) or not variants:
            raise ValueError("each record needs non-empty style_variants dict")
        base_text = f"User: {prompt}\nAssistant: {base}"
        h0 = encoder([base_text])[0]
        for _name, text in variants.items():
            t = f"User: {prompt}\nAssistant: {text}"
            h1 = encoder([t])[0]
            rows.append((h1 - h0).reshape(-1))
    if not rows:
        raise ValueError("no style deltas collected")
    return torch.stack(rows, dim=0)
