# Walkthroughs (tiered index)

!!! warning "Evidence only — not step 1"
    **Start:** [ADOPT.md](../../ADOPT.md) → [Golden paths](../GOLDEN_PATHS.md) → your data → [Step 5](08-falsification-controls.md).  
    **Which walkthrough for your job?** [Daily AI use map](DAILY_AI_USE.md).

---

## Tier 1 — After golden path (most integrators)

| # | Guide | Daily AI task | Subtype | When |
|---|-------|---------------|---------|------|
| 18 | [PMHTrainer quickstart](18-pmh-trainer-quickstart.md) | First PMH integration in PyTorch | * | Highest signal/minute |
| 8 | [**Falsification controls**](08-falsification-controls.md) | Prove deploy gains before any claim | * | **Required** |
| 1 | [PyTorch domain D4](01-pytorch-domain-d4.md) | Site/camera/hospital shift, custom `nn.Module` | D4 | Default G1 deep dive |
| 3 | [Frozen sklearn D1](03-office31-sklearn-d1.md) | Frozen embeddings, CRM/churn exports | D1 | G2 Office-31 style |
| 17 | [Compare arms in your pipeline](17-compare-arms-your-pipeline.md) | Fair ablation table on **your** train code | * | After integrate, before reports |

---

## Tier 2 — Same as APPLICATIONS profiles

| # | Guide | Daily AI task | Maps to `route --task` |
|---|-------|---------------|------------------------|
| 2 | [ResNet D4](02-resnet-vision-d4.md) | timm/ResNet fine-tune, `layer4` hook | `vision_classification` |
| 10 | [Lightning](10-lightning.md) | Existing `LightningModule` projects | `pytorch_lightning` |
| 6 | [LLM style D7](06-llm-style-d7.md) | Template/tone shift with style pairs | `llm_style_or_format` |
| 7 | [HF Trainer D7/DPO](07-hf-trainer-d7-dpo.md) | HF `Trainer`, LoRA, DPO preferences | `llm_style_or_format` / G3b |
| 16 | [Augmentation D3](16-augmentation-d3.md) | Known aug robustness (blur, JPEG) | `augmentation_robustness` |
| 11 | [Temporal D6](11-temporal-d6.md) | Sensor/time-series drift | `temporal_drift` |
| 5 | [Compositional D5](05-compositional-d5.md) | Known coord blocks (joints, tokens) | `compositional_coordinates` |
| 9 | [CLI JSON jobs](09-cli-json-jobs.md) | HPC / Makefile `pmh-train estimate` | — |
| 19 | [Office-31 real data](19-office31-real-data.md) | Real download + sklearn T1 table | `frozen_embeddings_sklearn` |

---

## Tier 3 — Specialty / paper blocks

| # | Guide | Daily AI task | Subtype |
|---|-------|---------------|---------|
| 4 | [Multi-layer CNN](04-multilayer-convnet.md) | Paper multilayer DA (research) | D3/D4 |
| 12 | [ViT CLS D4](12-vit-cls-d4.md) | ViT/timm CLS + optional D2 | D4 / D2 |
| 13 | [Speech Whisper D4](13-speech-whisper-d4.md) | ASR mic/room deploy shift | D4 |
| 14 | [QM9 molecule D5](14-qm9-molecule-d5.md) | Chemistry replication | D5 |
| 15 | [CodeBERT tokens D5](15-codebert-tokens-d5.md) | Code token-group D5 | D5 |

**Paper presets:** [paper-presets-by-block.md](paper-presets-by-block.md) · [PAPER_ALIGNMENT](../PAPER_ALIGNMENT.md)

---

## Contributor format

[GUIDE_FORMAT.md](GUIDE_FORMAT.md) · [Daily AI use audit](DAILY_AI_USE.md)
