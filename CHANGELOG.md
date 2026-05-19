# Changelog

## 0.7.0

- **Lab-grade docs:** Quickstart, Philosophy, 16 walkthroughs (ViT, Whisper, QM9, CodeBERT, D3 aug), examples catalog.
- **New examples:** `14_vit_cls_d4.py`, `15_speech_encoder_d4.py`, `16_qm9_molecule_d5.py`, `17_code_tokens_d5.py`, `18_augmentation_d3.py`.
- **Community:** CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue templates.
- **CI:** smoke-test runnable examples.
- README: badges, integration map, citation block.

## 0.6.4

- Eleven walkthroughs under `docs/walkthroughs/` (PyTorch, ResNet, Office-31, multi-layer CNN, D5, LLM D7, HF Trainer, controls, CLI, Lightning, D6).
- New examples: `12_resnet_hook_d4.py`, `13_compositional_train_d5.py`.

## 0.6.3

- README: display math ($$ blocks) for Problem / Object / Repair / Unification.
- New [docs/ARCHITECTURES.md](docs/ARCHITECTURES.md): two-phase workflow, where `h` hooks in, patterns for PyTorch, CNN/ViT, HF Trainer, D5, sklearn, D7.

## 0.6.2

- README and [docs/THEORY.md](docs/THEORY.md): matching principle for **any task / any architecture**; five-step recipe; falsification controls; explicit scope limits.

## 0.6.1

- **README / PyPI description:** plain-language problem statement, workflow, D1–D7 decision table, and per-domain use cases (vision, D5, D6, D7/LLM, CLI, falsification arms).

## 0.6.0

- **`pmh-train` CLI**: `list-methods`, `estimate --config job.json`, `preflight`, `run --config job.json`.
- **`pmh.catalog`**: D1–D7 input requirements and job validation.
- **Example 11**: Qwen/T7A JSONL (`style_pairs` + `preference_pairs`), optional LoRA + DPO+PMH demo.
- **Bundled samples**: `examples/data/*.jsonl`, `examples/configs/*.json`.
- **HF**: `load_preference_pairs_jsonl` for DPO schema.
- **Publishing**: TestPyPI workflow (`publish-testpypi.yml`), updated `PUBLISHING.md`.

## 0.5.0

- HF `PMHTrainer`, CORAL baseline, CI matrix, Office-31 example.

## 0.4.0

- Hugging Face D7, Lightning callback, Office-31 loader.

## 0.3.0

- Torch/sklearn/vision integrations, MkDocs.

## 0.2.0

- Artifacts, `PMHLoss`, configs.

## 0.1.0

- Core estimators D1–D7 and penalties.
