# 13 paper tasks (T1 → T7)

Tasks are listed **in paper order**. Your pipeline does not need to match a paper ID — pick the row whose **deploy change** sounds like yours, open the notebook, Run All on the demo, then edit §8 with your data.

Full examples and estimation detail: **[README — Find your deployment story](../../README.md#find-your-deployment-story-t1-through-t7)**.

Matching principle ([main.pdf](../../main.pdf)): estimate $\Sigma_{\text{task}}$ → matched PMH on `h` → Step 5 (matched vs wrong vs isotropic on deploy holdout).

| # | Task | Page | Notebook |
|---|------|------|----------|
| 1 | **T1** Classical ML + matched projection (ridge, SVM, k-NN, logistic) | [t01-classical.md](t01-classical.md) | [t01-classical.ipynb](../../notebooks/tasks/t01-classical.ipynb) |
| 2 | **T2A** ViT / image classifier | [t02a-vit-isotropic.md](t02a-vit-isotropic.md) | [t02a-vit-isotropic.ipynb](../../notebooks/tasks/t02a-vit-isotropic.ipynb) |
| 3 | **T2B** Medical imaging | [t02b-chexpert-isotropic.md](t02b-chexpert-isotropic.md) | [t02b-chexpert-isotropic.ipynb](../../notebooks/tasks/t02b-chexpert-isotropic.ipynb) |
| 4 | **T3A** Pose / keypoints | [t03a-pose-gradient.md](t03a-pose-gradient.md) | [t03a-pose-gradient.ipynb](../../notebooks/tasks/t03a-pose-gradient.ipynb) |
| 5 | **T3B** Depth estimation | [t03b-depth-augmentation.md](t03b-depth-augmentation.md) | [t03b-depth-augmentation.ipynb](../../notebooks/tasks/t03b-depth-augmentation.ipynb) |
| 6 | **T4A** Vision domain shift (single-layer / ResNet) | [t04a-vision-domain.md](t04a-vision-domain.md) | [t04a-vision-domain.ipynb](../../notebooks/tasks/t04a-vision-domain.ipynb) |
| 7 | **T4B** Vision domain shift (multilayer FPN / U-Net) | [t04b-multilayer-vision.md](t04b-multilayer-vision.md) | [t04b-multilayer-vision.ipynb](../../notebooks/tasks/t04b-multilayer-vision.ipynb) |
| 8 | **T5A** Molecules / graphs (QM9-style) | [t05a-qm9-molecule.md](t05a-qm9-molecule.md) | [t05a-qm9-molecule.ipynb](../../notebooks/tasks/t05a-qm9-molecule.ipynb) |
| 9 | **T5B** Code models | [t05b-code-tokens.md](t05b-code-tokens.md) | [t05b-code-tokens.ipynb](../../notebooks/tasks/t05b-code-tokens.ipynb) |
| 10 | **T6A** Speech / ASR | [t06a-speech-whisper.md](t06a-speech-whisper.md) | [t06a-speech-whisper.ipynb](../../notebooks/tasks/t06a-speech-whisper.ipynb) |
| 11 | **T6B** Time-series / HAR | [t06b-temporal-har.md](t06b-temporal-har.md) | [t06b-temporal-har.ipynb](../../notebooks/tasks/t06b-temporal-har.ipynb) |
| 12 | **T7A** LLM | [t07a-llm-style.md](t07a-llm-style.md) | [t07a-llm-style.ipynb](../../notebooks/tasks/t07a-llm-style.ipynb) |
| 13 | **T7B** Adversarial / PGD perturbations | [t07b-adversarial-pgd.md](t07b-adversarial-pgd.md) | [t07b-adversarial-pgd.ipynb](../../notebooks/tasks/t07b-adversarial-pgd.ipynb) |

## Which task fits your deploy change?

| Task | What changes at deploy | Examples | What we estimate | `nuisance=` |
|------|------------------------|----------|------------------|-------------|
| **T1** | Frozen embeddings shift between sites | Office-31; two hospitals’ features; lab A→B tabular | Source−target subspace on features | `subspace` | [t01-classical.md](t01-classical.md) |
| **T2A** | Generic input noise / corruption | ImageNet-C; camera noise; blur/JPEG | Isotropic noise level σ | `isotropic` | [t02a-vit-isotropic.md](t02a-vit-isotropic.md) |
| **T2B** | Scanner / hospital appearance on X-ray | CheXpert site shift; DICOM pipeline change | Isotropic σ (medical deploy stress) | `isotropic` | [t02b-chexpert-isotropic.md](t02b-chexpert-isotropic.md) |
| **T3A** | Camera/lighting; same keypoints | Studio→in-the-wild pose; broadcast→fan photos | Augmentation feature deltas | `augmentation` | [t03a-pose-gradient.md](t03a-pose-gradient.md) |
| **T3B** | Photometric shift; depth meaning fixed | Lighting on depth maps; synthetic→real RGB-D | Augmentation deltas | `augmentation` | [t03b-depth-augmentation.md](t03b-depth-augmentation.md) |
| **T4A** | New camera, site, or visual domain | Photo→sketch; warehouse A→B; day→night cls | Train vs deploy feature Gram | `domain_shift` | [t04a-vision-domain.md](t04a-vision-domain.md) |
| **T4B** | Sim→real texture + layout (segmentation) | GTA5→Cityscapes; synthetic IR→real seg | Domain Gram (multilayer in paper) | `domain_shift` | [t04b-multilayer-vision.md](t04b-multilayer-vision.md) |
| **T5A** | Atom positions move; property label fixed | QM9 conformers; docked poses | Nuisance coordinates (positions) | `compositional` | [t05a-qm9-molecule.md](t05a-qm9-molecule.md) |
| **T5B** | Token groups change; task label fixed | Renames; comment strip; obfuscation | Nuisance token/block indices | `compositional` | [t05b-code-tokens.md](t05b-code-tokens.md) |
| **T6A** | Mic, room, codec — same words | Libri conditions; new microphone | Temporal / content-residual (see doc) | `temporal` | [t06a-speech-whisper.md](t06a-speech-whisper.md) |
| **T6B** | Sensor drift over time | HAR placement; IMU aging | Temporal residual on sequences | `temporal` | [t06b-temporal-har.md](t06b-temporal-har.md) |
| **T7A** | Tone/format; facts unchanged | Bulleted vs prose; formal vs casual bot | Style pairs (same content) | `style` | [t07a-llm-style.md](t07a-llm-style.md) |
| **T7B** | Adversarial perturbations at deploy | PGD robustness; spoof patches | Subspace from attack deltas | `style` (PGD path) | [t07b-adversarial-pgd.md](t07b-adversarial-pgd.md) |

**T1** bundles seven classical subtasks in one notebook. **T2–T7** map to `paper_code/T2` … `T7`. Clone any row for a *similar* deploy change — not only the benchmark named in the paper.

Regenerate: `python scripts/render_handcrafted_tasks.py`

[Quickstart](../QUICKSTART.md) · [Will PMH help?](../WHEN_PMH_HELPS.md) · [API](../api/index.md)
