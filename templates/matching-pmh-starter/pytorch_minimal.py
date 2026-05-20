"""G1 golden path — replace loaders with yours."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pmh import PMHConfig, check_applicability, evaluate_robust_fit, robust_fit

# --- YOUR model ---
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(32, 16)
        self.head = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.enc(x)))


def _loader(n: int, shift: float) -> DataLoader:
    x = torch.randn(n, 32) + shift
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def main() -> None:
    model = Model()
    train_loader = _loader(200, 0.2)
    source_loader = _loader(150, 0.0)
    target_loader = _loader(150, 0.8)
    val_loader = _loader(80, 0.8)

    print(check_applicability(stack="pytorch", n_source=150, n_target=150).summary())

    # Tunable: PMHConfig.balanced(), rank=16, nuisance="domain_shift"
    out = robust_fit(
        model,
        train_loader,
        source_batches=source_loader,
        target_batches=target_loader,
        hook="auto",
        head=model.head,
        rank=16,
        pmh_config=PMHConfig.balanced(),
        epochs=5,
    )
    print(out.summary())

    report = evaluate_robust_fit(
        model,
        train_loader,
        val_loader,
        source_batches=source_loader,
        target_batches=target_loader,
        hook="auto",
        head=model.head,
        epochs=5,
        pmh_result=out,
        include_falsification=True,
    )
    print(report.summary())


if __name__ == "__main__":
    main()
