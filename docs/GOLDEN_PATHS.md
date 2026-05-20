# Golden paths (G1–G4 + framework variants)

**Prerequisite:** [Find your application](APPLICATIONS.md) (nuisance + walkthrough). **Not first:** [walkthrough grid](walkthroughs/index.md).

Pick **one** section below. Your application → nuisance mapping lives in [APPLICATIONS](APPLICATIONS.md).

Read **exactly one** section below (G1, G1b, G2, G3, G3b, or G4). Subtype D1–D7: use `pmh-train route` / `wizard` — details in [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md) when needed.

| Path | When |
|------|------|
| **G1** | Plain PyTorch loop or `robust_fit` |
| **G1b** | You already use **Lightning** |
| **G2** | Frozen features + **sklearn** |
| **G3** | HF model + two text corpora (`HFPMHTrainer`) |
| **G3b** | You already use **`transformers.Trainer`** (DPO, LoRA, etc.) |
| **G4** | Your own deltas / \(W\) / saved artifact |

```python
from pmh import suggest_subtype

rec = suggest_subtype(has_target_domain=True, has_target_labels=False)
print(rec.method, rec.nuisance, rec.reason)  # e.g. D4 domain_shift
```

---

<a id="g1"></a>

## G1 — PyTorch, two domains

**You have:** `model`, train loader, source + target loaders (target labels optional).

```python
from pmh import check_applicability, robust_fit, suggest_hook

print(check_applicability(stack="pytorch", n_source=500, n_target=400).summary())

out = robust_fit(
    model,
    train_loader,
    source_batches=source_loader,
    target_batches=target_loader,
    hook="auto",
    head=classifier,
    epochs=20,
)
print(out.preflight_message)

# Optional: ERM vs PMH on labeled target holdout (same report shape as sklearn)
from pmh import evaluate_robust_fit

report = evaluate_robust_fit(
    model, train_loader, val_loader,
    source_batches=source_loader, target_batches=target_loader,
    hook="auto", head=classifier, epochs=20, pmh_result=out,
)
print(report.summary())
```

- Colab: [domain_shift_first_run.ipynb](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/domain_shift_first_run.ipynb)
- Already on Lightning? → **G1b** below

---

<a id="g1b"></a>

## G1b — PyTorch Lightning

**You have:** a `LightningModule`, task loss in `training_step`, and hook features `h = backbone(x)` from source vs target for Phase A.

```python
pip install "matching-pmh[lightning]"
```

```python
from pmh import PMHConfig, PMHLoss, SigmaTaskConfig, estimate_from_config
from pmh.integrations.lightning import PMHLightningCallback, add_pmh_to_loss, _require_lightning
import torch.nn.functional as F

# Phase A — once (unlabeled domain features on the same backbone)
with torch.no_grad():
    h_src = model.backbone(x_source_batch)
    h_tgt = model.backbone(x_target_batch)
artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=32), h_src, h_tgt)
pmh_cfg = PMHConfig.balanced()
pmh_loss = PMHLoss(artifact, pmh_cfg)

# Phase B — inside training_step
def training_step(self, batch, batch_idx):
    x, y = batch
    task = F.cross_entropy(self.head(self.backbone(x)), y)
    total, pmh_term = add_pmh_to_loss(
        self, (x,), task, pmh_loss, backbone_attr="backbone"
    )
    return total

# Epoch schedule for cap/warmup
trainer = pl.Trainer(callbacks=[PMHLightningCallback.from_artifact(artifact, pmh_cfg)])
```

- Template: [`lightning_g1b_minimal.py`](https://github.com/vishalstark512/matching-pmh/blob/main/templates/matching-pmh-starter/lightning_g1b_minimal.py)
- Example: `examples/09_lightning_module.py` · walkthrough [10 Lightning](walkthroughs/10-lightning.md)

---

<a id="g2"></a>

## G2 — Frozen features + sklearn

**You have:** `x_source`, `y_source`, `x_target` (embeddings).

```python
from pmh import check_applicability, evaluate_baseline_vs_pmh

print(check_applicability(stack="sklearn", n_source=len(x_source), n_target=len(x_target)).summary())

report = evaluate_baseline_vs_pmh(
    x_source=x_source, y_source=y_source,
    x_target=x_target, y_target=y_target,
    compare_to=("coral",),
)
print(report.summary())
```

- Colab: [sklearn_frozen_features_first_run.ipynb](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/sklearn_frozen_features_first_run.ipynb)
- Pipeline: [gallery/tabular.md](gallery/tabular.md)

---

<a id="g3"></a>

## G3 — LLM / two text corpora (`HFPMHTrainer`)

**You have:** texts from corpus A and B, same labels; you want **pmh** to estimate and train (no subclass of `transformers.Trainer` required).

```python
pip install "matching-pmh[hf]"
```

```python
from pmh import check_applicability, robust_fit_text_domains

print(check_applicability(stack="hf", n_source=len(texts_a), n_target=len(texts_b)).summary())

out = robust_fit_text_domains(
    model, tokenizer, train_loader,
    source_texts=texts_a,
    target_texts=texts_b,
    epochs=3,
    rank=32,
)
print(out.preflight_message)
```

- Colab: [hf_two_corpora_first_run.ipynb](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/hf_two_corpora_first_run.ipynb)
- Template: [`hf_minimal.py`](https://github.com/vishalstark512/matching-pmh/blob/main/templates/matching-pmh-starter/hf_minimal.py)
- Style / format shift (D7): [walkthrough 6](walkthroughs/06-llm-style-d7.md) · `estimate_style`

---

<a id="g3b"></a>

## G3b — Hugging Face `Trainer` (keep your training stack)

**You have:** an existing `transformers.Trainer` setup (PEFT, DPO, custom callbacks). Add PMH in `compute_loss` via **`get_pmh_trainer()`**.

```python
pip install "matching-pmh[hf]"
```

```python
from pmh import PMHConfig, SigmaTaskConfig, estimate_from_config
from pmh.integrations.hf_trainer import get_pmh_trainer

# Phase A — collect representations h from source vs target (same hook as training)
with torch.no_grad():
    h_src = representation_fn(model, batch_from_site_a)  # [B, d]
    h_tgt = representation_fn(model, batch_from_site_b)
artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=32), h_src, h_tgt)

PMHTrainer = get_pmh_trainer()
trainer = PMHTrainer.from_artifact(
    artifact,
    PMHConfig.balanced(),
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    representation_fn=lambda m, inputs: m.get_hidden(inputs),  # your hook
)
trainer.train()
```

**Rules:** same `representation_fn` for estimate and train; default uses last hidden state if `output_hidden_states=True`.

- Template: [`hf_trainer_g3b_minimal.py`](https://github.com/vishalstark512/matching-pmh/blob/main/templates/matching-pmh-starter/hf_trainer_g3b_minimal.py)
- Example: `examples/10_hf_trainer.py` · walkthrough [7 HF Trainer + DPO](walkthroughs/07-hf-trainer-d7-dpo.md)
- Low-level loss only: `compute_pmh_training_loss` in [integrations-hf-trainer.md](integrations-hf-trainer.md)

**G3 vs G3b:** use **G3** when pmh should own estimate+fit; use **G3b** when you must keep the HF `Trainer` API.

---

<a id="g4"></a>

## G4 — Custom geometry (your deltas or \(W\))

**You have:** precomputed deltas, external \(W\), or a saved `.pt` artifact — same PMH train step.

```python
from pmh import PMHTrainer, PMHConfig, artifact_from_deltas

art = artifact_from_deltas(my_deltas, method="D7", rank=16)
trainer = PMHTrainer.from_artifact(model, art, hook=backbone, pmh_config=PMHConfig.balanced())
trainer.fit(train_loader, epochs=20)
```

Full guide: [CUSTOM_GEOMETRY.md](CUSTOM_GEOMETRY.md) · example `examples/23_custom_geometry_train.py`

---

## Before production

```bash
pmh-train doctor --stack pytorch
```

1. `check_applicability(...)` — go / marginal / no-go  
2. Target holdout metric — `evaluate_baseline_vs_pmh` or your own val loop  
3. `pmh-train validate -c examples/configs/validate_sklearn_synthetic.json` (sklearn)  
   or `validate_pytorch_smoke.json` (PyTorch toy arms)  
4. `export_deployment(artifact, "deploy/bundle", pmh_config=...)` — handoff bundle  
5. [Falsification controls](walkthroughs/08-falsification-controls.md) — full PyTorch compare

**Data on disk:** [DATA_LAYOUT.md](DATA_LAYOUT.md) · **Ship bundle:** [DEPLOYMENT.md](DEPLOYMENT.md)
