#!/usr/bin/env python3
"""Walkthrough 14: QM9-style molecular graph + D5 compositional (atom coordinates).

Toy message-passing: first 3 dims of node readout = coordinate nuisance (T5A).
  python examples/16_qm9_molecule_d5.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, estimate_from_config


class ToyMolGNN(nn.Module):
    def __init__(self, d_node: int = 16, d_out: int = 16) -> None:
        super().__init__()
        self.msg = nn.Linear(d_node, d_node)
        self.readout = nn.Linear(d_node, d_out)
        self.out_dim = d_out

    def encode_graph(self, node_x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.msg(node_x))
        h = torch.bmm(adj, h)
        return self.readout(h.mean(dim=1))


def main() -> None:
    torch.manual_seed(3)
    b, n, d_node = 32, 8, 16
    nuisance_idx = [0, 1, 2]

    model = ToyMolGNN()
    node = torch.randn(200, n, d_node)
    adj = torch.ones(200, n, n) / n
    with torch.no_grad():
        h_all = torch.stack(
            [model.encode_graph(node[i : i + 1], adj[i : i + 1]).squeeze(0) for i in range(200)]
        )
    h_all[:, :3] += 0.5 * torch.randn(200, 3)

    artifact = estimate_from_config(
        SigmaTaskConfig.for_compositional(nuisance_idx),
        h_all,
    )
    print(f"[estimate] D5 block norm={artifact.sigma[:3, :3].norm().item():.3f} preflight={artifact.preflight}")

    model = ToyMolGNN()
    head = nn.Linear(model.out_dim, 1)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3))

    for epoch in range(1, 16):
        pmh.set_epoch(epoch)
        node_b = torch.randn(b, n, d_node)
        adj_b = torch.ones(b, n, n) / n
        target = torch.randn(b, 1)
        opt.zero_grad()
        h = model.encode_graph(node_b, adj_b)
        task = F.mse_loss(head(h), target)
        total, raw = pmh.capped_total(task, h)
        total.backward()
        opt.step()
        if epoch in (1, 15):
            print(f"epoch {epoch:2d}  mse={task.item():.4f}  pmh={raw.item():.4f}")

    print("Production: map nuisance_idx to atom-coordinate channels in your GNN readout.")


if __name__ == "__main__":
    main()
