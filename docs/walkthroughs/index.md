# Walkthroughs (tiered index)

!!! warning "Not step 1"
    **Adoption:** [Find your application](../APPLICATIONS.md) → [Golden paths](../GOLDEN_PATHS.md) (one section).  
    Use this index only after that works, or for **paper replication**.

---

## Tier 1 — After golden path (most integrators)

| # | Guide | Subtype | When |
|---|-------|---------|------|
| 1 | [PyTorch domain D4](01-pytorch-domain-d4.md) | D4 | Default G1 deep dive |
| 3 | [Frozen sklearn D1](03-office31-sklearn-d1.md) | D1 | G2 Office-31 style |
| 8 | [**Falsification controls**](08-falsification-controls.md) | * | **Required before production claims** |
| 18 | [PMHTrainer quickstart](18-pmh-trainer-quickstart.md) | * | Object-oriented API |

---

## Tier 2 — Same as APPLICATIONS profiles

| # | Guide | Maps to `route --task` |
|---|-------|----------------------|
| 2 | [ResNet D4](02-resnet-vision-d4.md) | `vision_classification` |
| 6 | [LLM style D7](06-llm-style-d7.md) | `llm_style_or_format` |
| 7 | [HF Trainer D7/DPO](07-hf-trainer-d7-dpo.md) | `llm_style_or_format` / G3b |
| 9 | [CLI JSON jobs](09-cli-json-jobs.md) | HPC `pmh-train estimate` |
| 10 | [Lightning](10-lightning.md) | `pytorch_lightning` |
| 16 | [Augmentation D3](16-augmentation-d3.md) | `augmentation_robustness` |
| 11 | [Temporal D6](11-temporal-d6.md) | `temporal_drift` |
| 5 | [Compositional D5](05-compositional-d5.md) | `compositional_coordinates` |

---

## Tier 3 — Specialty / paper blocks

| # | Guide | Subtype |
|---|-------|---------|
| 4 | [Multi-layer CNN](04-multilayer-convnet.md) | D3/D4 |
| 12 | [ViT CLS D4](12-vit-cls-d4.md) | D4 / D2 |
| 13 | [Speech Whisper D4](13-speech-whisper-d4.md) | D4 |
| 14 | [QM9 molecule D5](14-qm9-molecule-d5.md) | D5 |
| 15 | [CodeBERT tokens D5](15-codebert-tokens-d5.md) | D5 |
| 17 | [Compare arms in your pipeline](17-compare-arms-your-pipeline.md) | * |
| 19 | [Office-31 real data](19-office31-real-data.md) | D1 |

**Paper presets:** [paper-presets-by-block.md](paper-presets-by-block.md) · [Recipe cards](../recipes/README.md)

---

## Contributor format

[GUIDE_FORMAT.md](GUIDE_FORMAT.md) · [Adaptation workbook](../ADAPTATION_WORKBOOK.md)
