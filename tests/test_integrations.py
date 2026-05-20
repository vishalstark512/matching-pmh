"""Third-party integrations: torch callbacks, HF style, Lightning, vision, baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from pmh import PMHConfig, SigmaTaskConfig, estimate_from_config
from pmh.baselines.coral import coral_align
from pmh.integrations import PMHCallback, train_epoch_with_pmh
from pmh.integrations.huggingface import (
    StylePairRecord,
    encode_style_deltas,
    estimate_style_sigma,
    load_style_pairs_jsonl,
)
from pmh.numpy_api import gram_from_diff_numpy
from pmh.sklearn_match import MatchedSubspaceProjector
from pmh.vision import MultiLayerPMHLoss


class HashEncoder(torch.nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: object) -> object:
        b, t = input_ids.shape
        h = torch.zeros(b, t, self.dim)
        for i in range(b):
            h[i] = torch.nn.functional.one_hot(input_ids[i] % self.dim, self.dim).float().mean(0)
        return type("Out", (), {"hidden_states": (h,)})()


class ToyTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    chat_template = None

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=128):
        rows = [[hash(w) % 997 for w in t.split()[:max_length]] or [0] for t in texts]
        max_len = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), max_len, dtype=torch.long)
        mask = torch.zeros(len(rows), max_len, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            mask[i, : len(r)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


@pytest.fixture
def style_jsonl(tmp_path: Path) -> Path:
    row = {
        "id": "1",
        "prompt": "Hi",
        "content_fixed": "Hello",
        "style_variants": {"a": "Hello!", "b": "Hello."},
    }
    p = tmp_path / "pairs.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return p


def test_pmh_callback_step():
    est = estimate_from_config(SigmaTaskConfig.for_isotropic(8, 0.1))
    backbone = nn.Linear(10, 8)
    head = nn.Linear(8, 2)
    cb = PMHCallback.from_artifact(est, encoder=backbone, head=head)
    cb.on_epoch_start(1)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    loss, info = cb.training_step((x, y))
    assert loss.requires_grad
    assert info.total_loss >= info.task_loss


def test_train_epoch_with_pmh():
    est = estimate_from_config(SigmaTaskConfig.for_isotropic(6, 0.1))
    model = nn.Sequential(nn.Linear(5, 6), nn.ReLU(), nn.Linear(6, 2))
    cb = PMHCallback.from_artifact(est, encoder=lambda x: model[0:2](x), head=lambda h: model[2](h))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    data = [(torch.randn(8, 5), torch.randint(0, 2, (8,))) for _ in range(3)]
    stats = train_epoch_with_pmh(model, cb, data, opt, epoch=2)
    assert stats["n_steps"] == 3.0


def test_numpy_d1_projector():
    rng = np.random.default_rng(0)
    n, d = 80, 20
    x_s = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, 5, n)
    x_t = x_s + 0.5
    proj = MatchedSubspaceProjector(rank=4, seed=0).fit(x_s, y, x_t, y)
    z = proj.transform(x_s)
    assert z.shape == x_s.shape


def test_multilayer_pmh():
    pmh = MultiLayerPMHLoss(("a", "b"), PMHConfig(weight=0.5))
    fc = torch.randn(4, 8)
    fn = torch.randn(4, 8)
    loss = pmh({"a": fc, "b": fc}, {"a": fn, "b": fn})
    assert loss.item() >= 0


def test_gram_from_diff_numpy():
    s = np.random.randn(30, 5).astype(np.float32)
    t = s + 0.1
    g = gram_from_diff_numpy(s, t)
    assert g.shape == (5, 5)


def test_load_style_pairs_jsonl(style_jsonl: Path):
    recs = load_style_pairs_jsonl(style_jsonl)
    assert len(recs) == 1
    assert isinstance(recs[0], StylePairRecord)


def test_encode_style_deltas_toy():
    rec = StylePairRecord("1", "Q", "A", {"s": "A!"})
    deltas = encode_style_deltas([rec], HashEncoder(32), ToyTokenizer(), batch_size=2)
    assert deltas.shape[0] == 1
    assert deltas.shape[1] == 32


def test_estimate_style_sigma_toy(style_jsonl: Path):
    recs = load_style_pairs_jsonl(style_jsonl)
    art = estimate_style_sigma(recs, HashEncoder(16), ToyTokenizer(), rank=4)
    assert art.method == "D7"
    assert art.sigma.shape == (16, 16)


def _lightning_usable() -> bool:
    try:
        from pmh.integrations.lightning import lightning_available

        return lightning_available()
    except Exception:
        return False


@pytest.mark.skipif(not _lightning_usable(), reason="lightning not installed or broken")
def test_add_pmh_to_loss():
    from pmh import PMHLoss
    from pmh.integrations.lightning import add_pmh_to_loss

    net = nn.Sequential(nn.Linear(5, 4), nn.ReLU())
    art = estimate_from_config(SigmaTaskConfig.for_isotropic(4, 0.1))
    pmh = PMHLoss(art, PMHConfig(weight=0.1))
    x = torch.randn(3, 5)
    task = torch.tensor(1.0, requires_grad=True)
    total, term = add_pmh_to_loss(net, (x,), task, pmh, backbone_attr="0")
    assert total.ndim == 0


@pytest.mark.skipif(not _lightning_usable(), reason="lightning not installed or broken")
def test_lightning_callback_instantiate():
    from pmh.integrations.lightning import PMHLightningCallback

    art = estimate_from_config(SigmaTaskConfig.for_isotropic(4, 0.1))
    cb = PMHLightningCallback.from_artifact(art)
    assert cb.pmh_loss is not None


def test_office31_domain_path_errors():
    from pmh.datasets.office31 import domain_path

    with pytest.raises(FileNotFoundError):
        domain_path("/nonexistent", "amazon")


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
    from pmh import PMHLoss
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
    from pmh.integrations.hf_trainer import get_pmh_trainer

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 3)

        def forward(self, input_ids, output_hidden_states=False, **kwargs):
            h = self.fc(input_ids)
            return type("O", (), {"logits": h, "hidden_states": (h,)})()

    PMHTrainer = get_pmh_trainer()
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
