# Examples catalog

Runnable templates for the [matching principle](https://github.com/vishalstark512/matching-pmh).  
Each script is **self-contained** (copy → adapt). Walkthrough prose: [docs/walkthroughs/](../docs/walkthroughs/index.md).

```bash
pip install matching-pmh torch
# optional extras per table below
```

---

## Core loop

| Script | D | Description |
|--------|---|-------------|
| [01_domain_shift_d4.py](01_domain_shift_d4.py) | D4 | Minimal backbone + domain shift |
| [02_save_load_artifact.py](02_save_load_artifact.py) | — | Artifact I/O |
| [04_falsification_controls.py](04_falsification_controls.py) | * | matched / wrong-W / isotropic |
| [minimal_loop.py](minimal_loop.py) | D4 | Shortest train snippet |

---

## Vision

| Script | D | Extra | Walkthrough |
|--------|---|-------|-------------|
| [07_vision_multilayer.py](07_vision_multilayer.py) | D3/D4 | — | [04](../docs/walkthroughs/04-multilayer-convnet.md) |
| [12_resnet_hook_d4.py](12_resnet_hook_d4.py) | D4 | `[vision]` | [02](../docs/walkthroughs/02-resnet-vision-d4.md) |
| [14_vit_cls_d4.py](14_vit_cls_d4.py) | D4 | — | [12](../docs/walkthroughs/12-vit-cls-d4.md) |
| [18_augmentation_d3.py](18_augmentation_d3.py) | D3 | — | [16](../docs/walkthroughs/16-augmentation-d3.md) |

---

## Classical ML

| Script | D | Extra | Walkthrough |
|--------|---|-------|-------------|
| [06_office31_sklearn.py](06_office31_sklearn.py) | D1 | `[sklearn,vision]` | [03](../docs/walkthroughs/03-office31-sklearn-d1.md) |

---

## Structure & science

| Script | D | Walkthrough |
|--------|---|-------------|
| [03_compositional_d5.py](03_compositional_d5.py) | D5 | [05](../docs/walkthroughs/05-compositional-d5.md) |
| [13_compositional_train_d5.py](13_compositional_train_d5.py) | D5 | [05](../docs/walkthroughs/05-compositional-d5.md) |
| [16_qm9_molecule_d5.py](16_qm9_molecule_d5.py) | D5 | [14](../docs/walkthroughs/14-qm9-molecule-d5.md) |
| [17_code_tokens_d5.py](17_code_tokens_d5.py) | D5 | [15](../docs/walkthroughs/15-codebert-tokens-d5.md) |

---

## Speech & language

| Script | D | Extra | Walkthrough |
|--------|---|-------|-------------|
| [15_speech_encoder_d4.py](15_speech_encoder_d4.py) | D4 | — | [13](../docs/walkthroughs/13-speech-whisper-d4.md) |
| [08_hf_style_d7.py](08_hf_style_d7.py) | D7 | `[hf]` | [06](../docs/walkthroughs/06-llm-style-d7.md) |
| [10_hf_trainer.py](10_hf_trainer.py) | D7 | `[hf]` | [07](../docs/walkthroughs/07-hf-trainer-d7-dpo.md) |
| [11_dpo_lora_style_pmh.py](11_dpo_lora_style_pmh.py) | D7 | `[hf-lora]` | [07](../docs/walkthroughs/07-hf-trainer-d7-dpo.md) |

---

## Framework integrations

| Script | Framework |
|--------|-----------|
| [09_lightning_module.py](09_lightning_module.py) | PyTorch Lightning |
| [05_yaml_config.py](05_yaml_config.py) | JSON config loading |

---

## Data & configs

| Path | Role |
|------|------|
| [data/style_pairs_sample.jsonl](data/style_pairs_sample.jsonl) | D7 style estimation |
| [data/preference_pairs_sample.jsonl](data/preference_pairs_sample.jsonl) | DPO-style training |
| [configs/d4_estimate.json](configs/d4_estimate.json) | CLI D4 job template |
| [configs/d7_style_estimate.json](configs/d7_style_estimate.json) | CLI D7 job |
| [configs/dpo_train_job.json](configs/dpo_train_job.json) | CLI train job |

---

## Smoke-test all (dev)

```bash
pytest tests/test_examples_smoke.py -q
```
