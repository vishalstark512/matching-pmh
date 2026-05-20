# Examples and Walkthroughs

Use this page when you want to know: **which example is closest to my task?**

Do not learn theory labels first. For the full reusable map, start with [13 task patterns](../TASK_PATTERNS.md), then open the closest example.

---

## Closest Example by Task

| I want to improve... | Closest example | Open |
|----------------------|-----------------|------|
| Segmentation in a new visual domain | Multi-layer vision | [Multi-layer CNN](04-multilayer-convnet.md) |
| Pose / keypoints from a new camera | Pose block | [Augmentation / pose-style controls](16-augmentation-d3.md) |
| Image classification with ResNet | Vision domain shift | [ResNet](02-resnet-vision-d4.md) |
| Image classification with ViT | ViT / CLS | [ViT CLS](12-vit-cls-d4.md) |
| Frozen embeddings or sklearn | Office-31 | [Frozen sklearn](03-office31-sklearn-d1.md) |
| Real Office-31 benchmark | Office-31 real data | [Office-31 real data](19-office31-real-data.md) |
| LLM style, format, tone, template | Qwen / style | [LLM style](06-llm-style-d7.md) |
| HF Trainer, DPO, LoRA | Qwen / DPO-style setup | [HF Trainer + DPO](07-hf-trainer-d7-dpo.md) |
| Speech / ASR | Whisper-style speech | [Speech Whisper](13-speech-whisper-d4.md) |
| Time-series / sensor drift | HAR / temporal | [Temporal](11-temporal-d6.md) |
| Molecules / graphs | QM9 | [QM9 molecule](14-qm9-molecule-d5.md) |
| Code models / token groups | CodeBERT | [CodeBERT tokens](15-codebert-tokens-d5.md) |
| Adversarial-style perturbations | CIFAR-style perturbation example | [Published evidence map](../PAPER_ALIGNMENT.md) |

---

## Most Useful After You Run a Recipe

| Guide | Use after |
|-------|-----------|
| [Falsification controls](08-falsification-controls.md) | Any production claim |
| [Compare arms in your pipeline](17-compare-arms-your-pipeline.md) | You need a fair report table |
| [PMHTrainer quickstart](18-pmh-trainer-quickstart.md) | You want lower-level PyTorch control |
| [Frozen sklearn](03-office31-sklearn-d1.md) | You adopted the frozen-features recipe |
| [Office-31 real data](19-office31-real-data.md) | You want a real sklearn benchmark |

---

## Framework Variants

| Guide | Use when |
|-------|----------|
| [ResNet](02-resnet-vision-d4.md) | You need a concrete torchvision hook |
| [Lightning](10-lightning.md) | You keep a `LightningModule` |
| [LLM style](06-llm-style-d7.md) | You have content-fixed style pairs |
| [HF Trainer / DPO](07-hf-trainer-d7-dpo.md) | You must keep `transformers.Trainer`, LoRA, or DPO |
| [Augmentation](16-augmentation-d3.md) | Production change is a known transform family |

---

## Specialty / Published Evidence

These are not first-use docs.

| Guide | Why it exists |
|-------|---------------|
| [PyTorch domain](01-pytorch-domain-d4.md) | Detailed train-vs-production mechanics |
| [Multi-layer CNN](04-multilayer-convnet.md) | Paper multilayer DA |
| [Compositional](05-compositional-d5.md) | Known coordinate blocks |
| [CLI JSON jobs](09-cli-json-jobs.md) | HPC / Makefile estimate jobs |
| [Temporal](11-temporal-d6.md) | Sequence drift |
| [ViT CLS](12-vit-cls-d4.md) | ViT CLS hook detail |
| [Speech Whisper](13-speech-whisper-d4.md) | ASR mic/room shift |
| [QM9 molecule](14-qm9-molecule-d5.md) | Chemistry replication |
| [CodeBERT tokens](15-codebert-tokens-d5.md) | Token-group example |

Published evidence map: [paper-presets-by-block.md](paper-presets-by-block.md) · [PAPER_ALIGNMENT](../PAPER_ALIGNMENT.md)

Contributor format: [GUIDE_FORMAT.md](GUIDE_FORMAT.md) · [Daily AI use audit](DAILY_AI_USE.md)
