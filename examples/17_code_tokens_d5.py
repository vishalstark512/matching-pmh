#!/usr/bin/env python3
"""Walkthrough 15: Code-style token embeddings + D5 (BigCloneBench T5B template).

Identifier token dimensions are nuisance; operator/keyword dims carry signal.
  python examples/17_code_tokens_d5.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, estimate_from_config


class CodeTokenEncoder(nn.Module):
    def __init__(self, vocab: int = 500, d: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.out_dim = d

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(token_ids).mean(dim=1)


def main() -> None:
    torch.manual_seed(4)
    d = 32
    nuisance_idx = list(range(8))

    enc = CodeTokenEncoder()
    tokens = torch.randint(0, 500, (256, 24))
    with torch.no_grad():
        h = enc.encode(tokens)
    h[:, nuisance_idx] += torch.randn(256, len(nuisance_idx))

    artifact = estimate_from_config(SigmaTaskConfig.for_compositional(nuisance_idx), h)
    print(f"[estimate] preflight={artifact.preflight}")

    enc = CodeTokenEncoder()
    head = nn.Linear(d, 2)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    pmh = PMHLoss(artifact, PMHConfig(weight=0.25, cap_ratio=0.3, warmup_epochs=1))

    for epoch in range(1, 21):
        pmh.set_epoch(epoch)
        tok = torch.randint(0, 500, (32, 24))
        y = torch.randint(0, 2, (32,))
        opt.zero_grad()
        h_b = enc.encode(tok)
        task = F.cross_entropy(head(h_b), y)
        total, raw = pmh.capped_total(task, h_b)
        total.backward()
        opt.step()
        if epoch in (1, 20):
            print(f"epoch {epoch:2d}  task={task.item():.4f}  pmh={raw.item():.4f}")

    print("Falsification: re-run with nuisance_idx on signal dims to reproduce T5B E1S negative.")


if __name__ == "__main__":
    main()
