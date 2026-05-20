# Walkthroughs × daily AI work

Use this page to see **which evidence walkthrough matches your job** and whether you should read it **before** or **after** the [golden paths](../GOLDEN_PATHS.md).

**Rule:** [ADOPT.md](../../ADOPT.md) → golden path → **your data** → [Step 5 falsification](08-falsification-controls.md). Walkthroughs are **depth**, not step 1.

---

## Quick map (most common jobs)

| Your daily AI task | Start here (adopt) | Evidence walkthrough (after) | `pmh-train route` |
|--------------------|--------------------|------------------------------|-------------------|
| New camera / hospital / site for vision model | [G1](../GOLDEN_PATHS.md#g1) · [00_first_run](https://github.com/vishalstark512/matching-pmh/blob/main/examples/00_first_run_domain_shift.py) | [01 PyTorch D4](01-pytorch-domain-d4.md) or [18 quickstart](18-pmh-trainer-quickstart.md) | `vision_classification` / `pose_or_keypoints` |
| ResNet / timm / ViT fine-tune | [G1](../GOLDEN_PATHS.md#g1) | [02 ResNet](02-resnet-vision-d4.md) · [12 ViT](12-vit-cls-d4.md) | `vision_classification` |
| MLOps on frozen embeddings, churn, CRM exports | [G2](../GOLDEN_PATHS.md#g2) · `evaluate --demo` | [03 sklearn D1](03-office31-sklearn-d1.md) · [19 real Office-31](19-office31-real-data.md) | `frozen_embeddings_sklearn` |
| LLM format / tone / template at deploy | [G3](../GOLDEN_PATHS.md#g3) | [06 style D7](06-llm-style-d7.md) | `llm_style_or_format` |
| HF `Trainer` / DPO / LoRA | [G3b](../GOLDEN_PATHS.md#g3b) | [07 HF Trainer](07-hf-trainer-d7-dpo.md) | `llm_style_or_format` |
| PyTorch Lightning project | [G1b](../GOLDEN_PATHS.md#g1b) | [10 Lightning](10-lightning.md) | `pytorch_lightning` |
| Known aug robustness (blur, JPEG, …) | [G1](../GOLDEN_PATHS.md#g1) + D3 | [16 augmentation](16-augmentation-d3.md) | `augmentation_robustness` |
| Time series / sensor drift | [G1](../GOLDEN_PATHS.md#g1) | [11 temporal](11-temporal-d6.md) | `temporal_drift` |
| Tabular blocks (joints, token spans) | [G4](../GOLDEN_PATHS.md#g4) | [05 compositional](05-compositional-d5.md) | `compositional_coordinates` |
| **Before any production claim** | Step 5 API | [**08 falsification**](08-falsification-controls.md) · [17 compare arms](17-compare-arms-your-pipeline.md) | — |
| Batch/HPC `pmh-train estimate` | [INTEGRATE](../INTEGRATE.md) | [09 CLI JSON](09-cli-json-jobs.md) | — |

---

## Per-walkthrough audit

| WT | Adoptability for daily work | Verdict |
|----|---------------------------|---------|
| **01** PyTorch D4 | **High** — default deep learning path; synthetic script; maps 1:1 to `PMHTrainer` / `robust_fit` | Use after G1; best technical reference for site shift |
| **02** ResNet | **High** for vision engineers | Hook on `layer4`; needs `[vision]` extra |
| **03** sklearn D1 | **High** for embedding pipelines | Office-31 story; prefer `evaluate_baseline_vs_pmh` for Step 5 |
| **04** Multi-layer CNN | **Low** daily — paper multilayer DA | Researchers / T4B only |
| **05** Compositional D5 | **Niche** — robotics/NLP token coords | Only if you have `nuisance_indices` |
| **06** LLM style D7 | **Medium–High** for LLM teams | Needs style-pair JSONL; not “two random corpora” |
| **07** HF Trainer | **Medium** — shopify HF stack | G3b golden path is shorter; WT7 for DPO detail |
| **08** Falsification | **Required** — not optional depth | Everyone; pair with `evaluate_*` |
| **09** CLI JSON | **Niche** — cluster jobs | Integrators with folder/npy layout on HPC |
| **10** Lightning | **High** for Lightning users | Short; copy `training_step` pattern |
| **11** Temporal D6 | **Niche** — HAR, finance series | Needs `[N,T,d]` pipeline |
| **12** ViT CLS | **Medium** — timm/HF ViT | CLS + optional D2 isotropic |
| **13** Speech Whisper | **Niche** — ASR deploy mic shift | Audio folder → encoder hook |
| **14** QM9 | **Low** — chemistry research | Paper T5A |
| **15** CodeBERT | **Low** — token-level D5 | Paper specialty |
| **16** Augmentation D3 | **Medium** when shift = named augs | Not for unknown site shift |
| **17** Compare arms | **High post-integrate** | Fair ablation table for reports |
| **18** Quickstart | **Highest** signal/minute | Read instead of WT01 if overwhelmed |
| **19** Office-31 real | **High** after G2 demo | Download data; real D1 protocol |

---

## Gaps we fixed in docs (so walkthroughs work as evidence)

- Broken links to deleted pages (`ADAPTATION_WORKBOOK`, `gallery/`, etc.) → golden paths / integrate / paper alignment  
- Each walkthrough: **adoption banner** + **API note** (`nuisance=` = shift type) + **deployment shift sentence** (07–19 included)  
- [index.md](index.md) lists a **Daily AI task** column per walkthrough  
- Prose outside code fences uses “deployment shift” where it meant deploy geometry (API param names unchanged)  

---

## What walkthroughs still assume (be honest)

| Assumption | Mitigation |
|------------|------------|
| You can name **site A vs B** (or style pairs, or aug modes) | If not, read [WHEN_PMH_HELPS](../WHEN_PMH_HELPS.md) — PMH may not apply |
| PyTorch guides need **hook** `[B,d]` | `suggest_hook(model)` · [INTEGRATE](../INTEGRATE.md) |
| sklearn D1 needs **labels on both** domains for subspace | Use `domain_shift` (D4) if target unlabeled |
| LLM D7 needs **content-fixed style pairs** | Two corpora alone → `robust_fit_text_domains` (G3) |
| Specialty WT (04, 12–15) mirror **paper blocks** | Use `pmh-train list-presets` only for replication |

---

## Suggested reading order by experience

**New to PMH:** [18 quickstart](18-pmh-trainer-quickstart.md) → [08 falsification](08-falsification-controls.md) → WT01 only if you need D4 detail.

**Sklearn/embeddings:** G2 + `pmh-train evaluate` → [03](03-office31-sklearn-d1.md) → [19](19-office31-real-data.md).

**Production ML engineer:** G1 or G1b → [17](17-compare-arms-your-pipeline.md) + [08](08-falsification-controls.md).

**Research / paper:** [paper-presets-by-block](paper-presets-by-block.md) + [PAPER_ALIGNMENT](../PAPER_ALIGNMENT.md) — not the adopt path.
