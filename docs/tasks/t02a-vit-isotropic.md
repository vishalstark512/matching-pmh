# T2A — ViT / image classifier — isotropic sensor noise (Type 2)

**Source of truth:** `paper_code/T2/Task2A/FINAL.md`

**Lemma:** D2 (isotropic input / acquisition noise) · **Stack:** PyTorch · **Preset:** `t2a_vit_isotropic`

**Nuisance:** $\Sigma_{\text{task}} = \sigma^2 I$ in input or embedding space — **no low-rank $W$** in the paper Type-2 block. Matched isotropic PMH penalizes encoder sensitivity along that probe (Jacobian trace / feature drift).

**Notebook:** [t02a-vit-isotropic.ipynb](../../notebooks/tasks/t02a-vit-isotropic.ipynb) — Run All on a **tiny RGB CNN** (library demo). Full ImageNet ViT-B/16 reproduction stays in `paper_code/T2/Task2A/`.

---

## What T2A achieved (headline)

> **Isotropic PMH on ViT-B/16** preserves **clean ImageNet accuracy** (−0.18 pp) while improving **Gaussian robustness** (+8.28 pp at σ=0.20) and **mean ImageNet-C** (+4.29 pp at severity 3). **Geometry:** layer-averaged TDI **−58%** vs ERM at σ=0.10; input-Jacobian Frobenius **−9%**.

| Evidence | Result (paper, frozen May 2026) |
|----------|----------------------------------|
| Clean top-1 | ERM **97.02%** · PMH **96.84%** |
| Gaussian σ=0.10 | ERM **94.52%** · PMH **96.28%** (+1.76 pp) |
| ImageNet-C mean | ERM **82.90%** · PMH **87.19%** (+4.29 pp) |
| TDI @ σ=0.10 | ERM **0.0656** · PMH **0.0275** (−58%) |

**Paper arms:** `pretrained_baseline` · `erm_finetuned` · `pmh_finetuned` (no wrong-$W$ arm — Type 2 is isotropic-only).

**Library mapping:** `nuisance="isotropic"` · `PMHTrainer` / `evaluate_robust_fit` · `trajectory_tdi_encoder` · preset `t2a_vit_isotropic`.

---

## T2A subtasks (paper_code)

<a id="t2a-vit-imagenet"></a>

### ImageNet ViT-B/16 + isotropic PMH

- **Script:** `paper_code/T2/Task2A/train.py` (`task1a_tune`)
- **Data:** HF `ILSVRC/imagenet-1k` val subset, 100 classes × 50 images
- **Train:** ERM vs CE + isotropic PMH on CLS embeddings
- **Preset:** `t2a_vit_isotropic` (`noise_level=0.10`)

```bash
cd paper_code/T2/Task2A
pip install -r requirements.txt
python train.py task1a_tune --val_source hf --train_source hf_val --out_dir results/run1
python evaluate.py --results_dir results/run1
```

<a id="t2a-geometry-tdi"></a>

### TDI / Jacobian probes

- **Script:** `recompute_task1a_tdi.py`, `eval_extended.py`
- **Metrics:** `tdi` (layer-averaged, label-free), `jacobian_fro`, `esi` (needs labels)

<a id="t2a-imagenet-c"></a>

### ImageNet-C transfer

- **Script:** `eval_imagenet_c.py` — 15 corruption types, severity 3
- PMH trained on **Gaussian only**; gains on noise / frost / blur families

---

## Run with matching-pmh (demo → your ViT)

```bash
pip install matching-pmh torch
```

Open [t02a-vit-isotropic.ipynb](../../notebooks/tasks/t02a-vit-isotropic.ipynb) — uses `pytorch_isotropic_demo_loaders()` + `nuisance="isotropic"`.

**Your pipeline:**

1. Fine-tune ViT (or CNN) with standard CE on **site A** images.
2. `PMHTrainer(..., nuisance="isotropic", noise_level=0.10)` — estimate D2 on hook features, add capped PMH term.
3. Evaluate on **deploy holdout** with acquisition noise (Gaussian or ImageNet-C-style corruptions).
4. Report `trajectory_tdi_encoder` or saved `summary.json`-style geometry before trusting accuracy gains.

```python
from pmh.benchmark.presets import get_preset
from pmh import PMHTrainer, evaluate_robust_fit

preset = get_preset("t2a_vit_isotropic")
# PMHTrainer(model, hook=..., nuisance="isotropic", noise_level=0.10, pmh_config=preset.pmh_config)
```

---

## Do not use PMH when

Large **domain** shift without an isotropic noise model (camera/site change with different $W$) — use **T4** (D4 domain Gram) instead. Type 2 does **not** establish `matched > wrong-W` ordering (that lives in T3B / T6 / T7B).

---

## Replace demo data with yours

Swap `pytorch_isotropic_demo_loaders` for your `train_loader` + `source_batches` (hook = backbone before classifier head). Keep **same label semantics** under noise.

[← All 13 tasks](index.md) · [T2B medical](t02b-chexpert-isotropic.md) · [Quickstart](../QUICKSTART.md)

<a id="t02a-vit-isotropic"></a>
