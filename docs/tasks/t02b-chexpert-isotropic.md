# T2B — Medical imaging — isotropic acquisition noise (Type 2)

**Source of truth:** `paper_code/T2/Task2B/FINAL.md`

**Lemma:** D2 · **Stack:** PyTorch (ResNet-18 embeddings) · **Preset:** `t2b_chexpert_isotropic`

**Nuisance:** Gaussian noise + intensity scaling in **[0,1] RGB** (grayscale X-ray → 3-channel). PMH on **pooled ResNet embeddings** with cap + warmup — same Type-2 recipe as T2A, clinical domain.

**Notebook:** [t02b-chexpert-isotropic.ipynb](../../notebooks/tasks/t02b-chexpert-isotropic.ipynb) — mini CNN demo. Full Pneumonia dataset run: `paper_code/T2/Task2B/`.

---

## What T2B achieved (headline)

> On pneumonia chest X-rays, **isotropic PMH (E1)** gives best **saliency stability (0.723)** and **~9× lower embedding drift** under Gaussian noise vs **B0**, while **two-view training without PMH (E1_no_pmh)** wins **worst-shift accuracy** (78.9% vs B0 62.5% at σ=0.10).

| Metric | B0 | E1_no_pmh | **E1 (PMH)** |
|--------|-----|-----------|--------------|
| Clean acc | **91.67%** | 88.94% | 86.54% |
| Worst-shift acc | 62.50% | **78.85%** | 74.20% |
| Gaussian σ=0.10 | 62.50% | 83.97% | **81.89%** |
| ‖Δφ‖ @ σ=0.10 | 19.05 | 2.85 | **2.21** |
| Saliency stability | 0.560 | 0.707 | **0.723** |

**Honest cite:** Do not claim E1 wins every metric — pair **geometry** (PMH) with **shift accuracy** (E1_no_pmh) per FINAL §7.

**Library mapping:** `nuisance="isotropic"` · `noise_level=0.08` · `warmup_epochs=5` · `pmh.weight=0.5`, `cap_ratio=0.5`.

---

## T2B subtasks (paper_code)

<a id="t2b-pneumonia-clean"></a>

### Pneumonia — clean test

- **Script:** `paper_code/T2/Task2B/train.py` + `eval.py`
- **Arms:** B0 · VAT · E1_no_pmh · E1 · E1_embed_only
- **Preset:** `t2b_chexpert_isotropic`

```bash
cd paper_code/T2/Task2B
pip install -r requirements.txt
python train.py --run E1 --data_dir ./data --epochs 30
python eval.py --compare --data_dir ./data --out_dir runs/eval_out
```

<a id="t2b-robust-shift"></a>

### Robust eval — acquisition shifts

- **Script:** `eval_robust.py`
- Perturbations: `gaussian_*`, `intensity_*`, `gamma_*`, `rotate_*`, `zoom_*`, `contrast_*`, `blur_3`
- JSON: `runs/eval_out_robust/compare_results_robust.json`

<a id="t2b-saliency"></a>

### Saliency stability

- **Script:** `saliency_stability.py`
- Eval noise σ=0.08, 20 batches — **E1 (PMH)** leads cosine map stability

---

## Run with matching-pmh

```bash
pip install matching-pmh torch
```

Notebook: [t02b-chexpert-isotropic.ipynb](../../notebooks/tasks/t02b-chexpert-isotropic.ipynb).

```python
from pmh.benchmark.presets import get_preset

preset = get_preset("t2b_chexpert_isotropic")
# PMHTrainer(..., nuisance="isotropic", noise_level=0.08, pmh_config=preset.pmh_config)
```

**Your clinical pipeline:** frozen ResNet (or ViT) embeddings per hospital → optional D2 PMH fine-tune with two views (clean + perturbed) → robust eval on held-out site with acquisition perturbations.

---

## Pair with T2A

| | T2A | T2B |
|---|-----|-----|
| Domain | ImageNet ViT | Chest X-ray ResNet |
| Headline metric | ImageNet-C + TDI | Saliency + embedding drift |
| Accuracy story | PMH ≈ ERM clean | E1_no_pmh best shift; B0 best clean |

---

## Do not use PMH when

Disease **label definition** changes at deploy, or shift is primarily **anatomical / site** (use D4/T4), not acquisition noise.

[← All 13 tasks](index.md) · [T2A ViT](t02a-vit-isotropic.md) · [Quickstart](../QUICKSTART.md)

<a id="t02b-chexpert-isotropic"></a>
