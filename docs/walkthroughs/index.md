# Walkthroughs

**Before picking a walkthrough:** [Getting started](../GETTING_STARTED.md) · [Choose your setup](../CHOOSE_YOUR_SETUP.md)

**18 templates** from nuisance story → estimate $\Sigma_{\mathrm{task}}$ → train → falsify.  
Copy the closest row, swap **your** data and model. Runnable scripts live in `examples/`.

| # | Walkthrough | Est. | Stack (example) | Script |
|---|-------------|------|---------------------|--------|
| 1 | [PyTorch domain shift (D4)](01-pytorch-domain-d4.md) | D4 | Any `nn.Module` | `01_domain_shift_d4.py` |
| 2 | [ResNet hook + D4](02-resnet-vision-d4.md) | D4 | torchvision | `12_resnet_hook_d4.py` |
| 3 | [Office-31 + sklearn (D1)](03-office31-sklearn-d1.md) | D1 | Frozen features | `06_office31_sklearn.py` |
| 4 | [Multi-layer ConvNet](04-multilayer-convnet.md) | D3/D4 | CNN feature maps | `07_vision_multilayer.py` |
| 5 | [Compositional D5](05-compositional-d5.md) | D5 | Coords / tokens | `03_…`, `13_…` |
| 6 | [LLM style D7](06-llm-style-d7.md) | D7 | HF style JSONL | `08_hf_style_d7.py` |
| 7 | [HF Trainer + DPO](07-hf-trainer-d7-dpo.md) | D7 | HF Trainer / LoRA | `10_…`, `11_…` |
| 8 | [Falsification controls](08-falsification-controls.md) | * | Any pipeline | `04_falsification_controls.py` |
| 9 | [CLI JSON jobs](09-cli-json-jobs.md) | * | HPC / repro | `pmh-train` |
| 10 | [Lightning](10-lightning.md) | D4 | Lightning | `09_lightning_module.py` |
| 11 | [Temporal D6](11-temporal-d6.md) | D6 | Sequence / sensor drift | API |
| 12 | [ViT CLS + D4](12-vit-cls-d4.md) | D4 | ViT / patch encoder | `14_vit_cls_d4.py` |
| 13 | [Speech encoder + D4](13-speech-whisper-d4.md) | D4 | Speech / mel encoder | `15_speech_encoder_d4.py` |
| 14 | [QM9 molecules D5](14-qm9-molecule-d5.md) | D5 | GNN / molecules | `16_qm9_molecule_d5.py` |
| 15 | [Code tokens D5](15-codebert-tokens-d5.md) | D5 | Token embeddings | `17_code_tokens_d5.py` |
| 16 | [Augmentations D3](16-augmentation-d3.md) | D3 | Known aug modes | `18_augmentation_d3.py` |
| 17 | [Compare arms (your pipeline)](17-compare-arms-your-pipeline.md) | * | Your model + data | `20_compare_training_arms.py` |
| 18 | [PMHTrainer quickstart](18-pmh-trainer-quickstart.md) | D4 | PyTorch facade | `01_domain_shift_d4.py` |

Walkthroughs are **templates**—not paper task IDs. For a stack-agnostic checklist, see [ADAPT_YOUR_PIPELINE.md](../ADAPT_YOUR_PIPELINE.md).

---

## Suggested learning path

```mermaid
flowchart TD
  Q[Quickstart] --> W1[1 PyTorch D4]
  W1 --> W8[8 Controls]
  W8 --> branch{Your domain?}
  branch -->|Vision| V[2 ResNet · 12 ViT · 4 Multi-layer · 16 D3]
  branch -->|NLP/LLM| L[6 D7 style · 7 HF Trainer]
  branch -->|Science| S[14 QM9 · 15 Code · 13 Speech]
  branch -->|Classical| C[3 Office-31 · 9 CLI]
```

1. [QUICKSTART.md](../QUICKSTART.md) — 10 minutes  
2. [Walkthrough 1](01-pytorch-domain-d4.md) — core two-phase loop  
3. [Walkthrough 8](08-falsification-controls.md) — credible claims  
4. Pick one row from the table for your stack  

---

## Theory & integration

- [THEORY.md](../THEORY.md) — mathematics  
- [ARCHITECTURES.md](../ARCHITECTURES.md) — hook points  
- [PHILOSOPHY.md](../PHILOSOPHY.md) — API design  
- [examples/README.md](https://github.com/vishalstark512/matching-pmh/blob/main/examples/README.md) — script catalog  
- [Gallery](../gallery/README.md) — vision / tabular / NLP copy-paste  
- [Hook cookbook](../hooks.md) — timm, HF, ResNet  
