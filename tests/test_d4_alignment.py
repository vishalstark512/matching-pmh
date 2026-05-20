"""Class-aligned D4 collection and Gram."""

from __future__ import annotations

import torch

from pmh.estimators.d4_domain import estimate_d4, estimate_d4_from_paired_diffs, gram_from_paired_diffs
from pmh.features import align_batch_by_labels, collect_domain_paired_diffs


def test_align_batch_by_labels_matches_class():
    torch.manual_seed(0)
    y_s = torch.tensor([0, 1, 2])
    y_t = torch.tensor([2, 0, 1])
    x_t = torch.arange(9, dtype=torch.float32).view(3, 3)
    _, y_a = align_batch_by_labels(y_s, x_t, y_t)
    assert (y_a == y_s).all()


def test_collect_domain_paired_diffs_uses_labels():
    torch.manual_seed(0)

    def enc(x):
        return x.view(x.size(0), -1).float()

    src = [(torch.randn(8, 2), torch.zeros(8, dtype=torch.long))]
    tgt = [(torch.randn(8, 2) + 3.0, torch.zeros(8, dtype=torch.long))]
    diff, aligned = collect_domain_paired_diffs(enc, src, tgt, max_batches=1)
    assert aligned is True
    assert diff.shape == (8, 2)
    g = gram_from_paired_diffs(diff)
    assert g.shape == (2, 2)


def test_unlabeled_batches_skip_alignment():
    torch.manual_seed(1)

    def enc(x):
        return x

    src = [torch.randn(4, 3)]
    tgt = [torch.randn(4, 3)]
    _, aligned = collect_domain_paired_diffs(enc, src, tgt, max_batches=1)
    assert aligned is False
