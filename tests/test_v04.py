import json
import tempfile
from pathlib import Path

import pytest
import torch

from pmh.integrations.huggingface import (
    StylePairRecord,
    encode_style_deltas,
    estimate_style_sigma,
    load_style_pairs_jsonl,
)


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


def test_load_style_pairs_jsonl(style_jsonl: Path):
    recs = load_style_pairs_jsonl(style_jsonl)
    assert len(recs) == 1
    assert isinstance(recs[0], StylePairRecord)


def test_encode_style_deltas_toy():
    rec = StylePairRecord("1", "Q", "A", {"s": "A!"})
    model = HashEncoder(32)
    tok = ToyTokenizer()
    deltas = encode_style_deltas([rec], model, tok, batch_size=2)
    assert deltas.shape[0] == 1
    assert deltas.shape[1] == 32


def test_estimate_style_sigma_toy(style_jsonl: Path):
    recs = load_style_pairs_jsonl(style_jsonl)
    art = estimate_style_sigma(recs, HashEncoder(16), ToyTokenizer(), rank=4)
    assert art.method == "D7"
    assert art.sigma.shape == (16, 16)


def test_add_pmh_to_loss():
    import torch.nn as nn
    from pmh import PMHLoss, SigmaTaskConfig, estimate_from_config
    from pmh.integrations.lightning import add_pmh_to_loss

    net = nn.Sequential(nn.Linear(5, 4), nn.ReLU())
    art = estimate_from_config(SigmaTaskConfig.for_isotropic(4, 0.1))
    pmh = PMHLoss(art, __import__("pmh").PMHConfig(weight=0.1))
    x = torch.randn(3, 5)
    task = torch.tensor(1.0, requires_grad=True)
    total, term = add_pmh_to_loss(net, (x,), task, pmh, backbone_attr="0")
    assert total.ndim == 0


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("lightning") is None
    and __import__("importlib").util.find_spec("pytorch_lightning") is None,
    reason="lightning not installed",
)
def test_lightning_callback_instantiate():
    from pmh import estimate_from_config, SigmaTaskConfig
    from pmh.integrations.lightning import PMHLightningCallback

    art = estimate_from_config(SigmaTaskConfig.for_isotropic(4, 0.1))
    cb = PMHLightningCallback.from_artifact(art)
    assert cb.pmh_loss is not None


def test_office31_domain_path_errors():
    from pmh.datasets.office31 import domain_path
    with pytest.raises(FileNotFoundError):
        domain_path("/nonexistent", "amazon")
