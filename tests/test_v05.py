import numpy as np
import pytest
import torch
import torch.nn as nn

from pmh.baselines.coral import coral_align


def test_coral_align_shape():
    x_s = np.random.randn(50, 8).astype(np.float32)
    x_t = x_s + 0.5
    a, t = coral_align(x_s, x_t)
    assert a.shape == x_s.shape
    assert t.shape == x_t.shape


def test_coral_changes_source():
    rng = np.random.default_rng(0)
    x_s = rng.standard_normal((100, 16)).astype(np.float32)
    x_t = rng.standard_normal((100, 16)).astype(np.float32) + 2.0
    a, _ = coral_align(x_s, x_t)
    assert not np.allclose(a, x_s, atol=1e-3)


def test_compute_pmh_loss_standalone():
    """Trainer-free path (no transformers.Trainer import)."""
    from pmh import PMHConfig, PMHLoss, estimate_from_config, SigmaTaskConfig
    from pmh.integrations.hf_trainer import compute_pmh_training_loss

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 3)

        def forward(self, input_ids, output_hidden_states=False, **kwargs):
            h = self.fc(input_ids)
            return type("O", (), {"logits": h, "hidden_states": (h,)})()

    art = estimate_from_config(SigmaTaskConfig.for_isotropic(3, 0.1))
    pmh = PMHLoss(art, PMHConfig(weight=0.1))
    batch = {"input_ids": torch.randn(4, 8), "labels": torch.randint(0, 3, (4,))}
    loss, task, pmh_term = compute_pmh_training_loss(M(), batch, pmh)
    assert loss.ndim == 0 and pmh_term.ndim == 0


def _trainer_import_ok() -> bool:
    import os

    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    try:
        from transformers import Trainer  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _trainer_import_ok(), reason="transformers.Trainer unavailable")
def test_pmh_trainer_compute_loss():
    from pmh import PMHConfig, estimate_from_config, SigmaTaskConfig
    from pmh.integrations.hf_trainer import get_pmh_trainer

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 3)

        def forward(self, input_ids, output_hidden_states=False, **kwargs):
            h = self.fc(input_ids)
            return type("O", (), {"logits": h, "hidden_states": (h,)})()

    PMHTrainer = get_pmh_trainer()
    art = estimate_from_config(SigmaTaskConfig.for_isotropic(3, 0.1))
    trainer = PMHTrainer.from_artifact(art, PMHConfig(weight=0.1), model=M())
    m = M()
    batch = {"input_ids": torch.randn(4, 8), "labels": torch.randint(0, 3, (4,))}
    trainer = PMHTrainer.from_artifact(
        estimate_from_config(SigmaTaskConfig.for_isotropic(3, 0.1)),
        PMHConfig(weight=0.1),
        model=m,
        representation_fn=lambda mod, inp: mod.fc(inp["input_ids"]),
    )
    loss = trainer.compute_loss(m, batch)
    assert loss.ndim == 0 and loss.requires_grad
