"""PyTorch Lightning integration for matched PMH."""

from __future__ import annotations

from typing import Any

import torch

from pmh.artifact import SigmaTaskEstimate
from pmh.config import PMHConfig
from pmh.training import PMHLoss

try:
    import lightning.pytorch as _pl
except ImportError:
    try:
        import pytorch_lightning as _pl  # type: ignore[no-redef]
    except ImportError:
        _pl = None  # type: ignore[assignment]


def lightning_available() -> bool:
    """True if Lightning can be imported (package present and loadable)."""
    try:
        if _pl is not None:
            _ = _pl.LightningModule
            return True
    except Exception:
        pass
    try:
        import lightning.pytorch as pl  # noqa: PLC0415

        _ = pl.LightningModule
        return True
    except Exception:
        pass
    try:
        import pytorch_lightning as pl  # noqa: PLC0415

        _ = pl.LightningModule
        return True
    except Exception:
        return False


def _require_lightning() -> Any:
    if _pl is None:
        raise ImportError(
            'Lightning integration requires lightning or pytorch-lightning. '
            'Install with: pip install "matching-pmh[lightning]"'
        )
    return _pl


def add_pmh_to_loss(
    pl_module: Any,
    batch: Any,
    task_loss: torch.Tensor,
    pmh_loss: PMHLoss,
    *,
    backbone_attr: str = "backbone",
    inputs_key: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add capped PMH to ``task_loss`` inside ``training_step``."""
    backbone = getattr(pl_module, backbone_attr)
    if isinstance(batch, (tuple, list)):
        x = batch[0]
    elif isinstance(batch, dict):
        x = batch[inputs_key] if inputs_key else batch.get("x", batch.get("inputs"))
        if x is None:
            raise KeyError("batch dict needs 'x' or 'inputs' or set inputs_key=")
    else:
        x = batch
    h = backbone(x)
    total, pmh_term = pmh_loss.capped_total(task_loss, h)
    if hasattr(pl_module, "log"):
        pl_module.log("pmh_loss", pmh_term.detach(), on_step=True, on_epoch=True)
    return total, pmh_term


if _pl is not None:

    class PMHLightningCallback(_pl.Callback):
        """Sets PMH epoch schedule; use with :func:`add_pmh_to_loss` in ``training_step``."""

        def __init__(
            self,
            pmh_loss: PMHLoss,
            *,
            log_every_n_steps: int = 50,
        ) -> None:
            super().__init__()
            self.pmh_loss = pmh_loss
            self.log_every_n_steps = log_every_n_steps

        @classmethod
        def from_artifact(
            cls,
            artifact: SigmaTaskEstimate,
            pmh_config: PMHConfig | None = None,
            **kwargs: Any,
        ) -> PMHLightningCallback:
            return cls(PMHLoss(artifact, pmh_config), **kwargs)

        def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
            self.pmh_loss.set_epoch(trainer.current_epoch)

else:

    class PMHLightningCallback:  # type: ignore[no-redef]
        """Placeholder when Lightning is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError('pip install "matching-pmh[lightning]"')
