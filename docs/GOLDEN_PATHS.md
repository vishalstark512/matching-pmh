# Golden paths (G1–G4 + framework variants)

**Prerequisite:** [Five-step recipe](FIVE_STEP_RECIPE.md) → [Applications](APPLICATIONS.md) → [Integrate](INTEGRATE.md) for install/CLI.

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

**Mode A (Jacobian)** · **Step 5:** compare deploy holdout + [falsification controls](walkthroughs/08-falsification-controls.md) (`compare_arms` or `evaluate_robust_fit` + wrong-W / isotropic).

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
    include_falsification=True,  # default — Step 5 on hook embeddings
)
print(report.summary())
```

- Colab: [domain_shift_first_run.ipynb](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/domain_shift_first_run.ipynb)
- CLI Step 5: `pmh-train evaluate --demo --stack pytorch` or `--source-dir A/ --target-dir B/`
- Parameters: [PARAMETERS_CHEATSHEET.md](PARAMETERS_CHEATSHEET.md)
- Already on Lightning? → **G1b** below

---

<a id="g1b"></a>

## G1b — PyTorch Lightning

**Mode A** · **Step 5:** [falsification controls](walkthroughs/08-falsification-controls.md) before production claims.

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

**Mode B (projection)** · **Step 5:** `evaluate_baseline_vs_pmh` (falsification arms **on by default**).

**You have:** `x_source`, `y_source`, `x_target` (embeddings). Start with the **Office-31-style synthetic** demo (no download), then swap your `.npy` files.

```python
from pmh import check_applicability, evaluate_baseline_vs_pmh, load_g2_demo_arrays

# Same spirit as T1 Office-31 (amazon -> dslr) — synthetic, runs in seconds
x_source, y_source, x_target, y_target = load_g2_demo_arrays(n=500, seed=0)

print(check_applicability(stack="sklearn", n_source=len(x_source), n_target=len(x_target)).summary())

report = evaluate_baseline_vs_pmh(
    x_source=x_source, y_source=y_source,
    x_target=x_target, y_target=y_target,
    compare_to=("coral",),  # optional baseline; Step 5 arms always included
)
print(report.summary())  # matched / wrong-W / isotropic on deploy holdout
```

**Real Office-31** (after basics work):

```bash
python examples/06_office31_sklearn.py --office31-root /path/to/office31
```

- Colab: [sklearn_frozen_features_first_run.ipynb](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/sklearn_frozen_features_first_run.ipynb)
- Quick script: `examples/02_g2_office31_style_demo.py`

---

<a id="g3"></a>

## G3 — LLM / two text corpora (`HFPMHTrainer`)

**Mode A** · **Step 5:** style/geometry checks — [falsification controls](walkthroughs/08-falsification-controls.md); report task and geometry separately ([§7.2](THEORY.md)).

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

**Mode A** · **Step 5:** [falsification controls](walkthroughs/08-falsification-controls.md) (matched / wrong-Σ / isotropic for D7).

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

**Mode A or B** (depends on hook) · **Step 5:** [falsification controls](walkthroughs/08-falsification-controls.md).

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
