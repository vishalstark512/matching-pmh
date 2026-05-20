# Find your application

**CLI is the detailed walkthrough** — this page is a finder + anchors. Spine: [FIVE_STEP_RECIPE](FIVE_STEP_RECIPE.md) · code: [GOLDEN_PATHS](GOLDEN_PATHS.md).

```bash
pmh-train route --search hospital
pmh-train route --task pose_or_keypoints
```

```python
from pmh import explain_task
print(explain_task("pose_or_keypoints"))
```

---

## Finder

<a id="application-finder"></a>

| Application | What changes | Fit | Details |
|-------------|--------------|-----|---------|
| Pose / keypoints — new camera or site | Camera / studio / hospital **look** changes (lightin… | **YES** | [↓](#pose_or_keypoints) |
| Image classification — new camera or site | Image **appearance** shifts (camera, geography, devi… | **YES** | [↓](#vision_classification) |
| Object detection — same classes, new domain | Scene **style** and sensor change; bounding-box clas… | **TRY** | [↓](#vision_detection) |
| Segmentation — same classes, new domain | Pixel **texture and sensor** shift; per-pixel class … | **TRY** | [↓](#vision_segmentation) |
| Text classification — new corpus or channel | **Wording and channel** shift (support tickets vs ch… | **YES** | [↓](#nlp_text_classification) |
| LLM — format / tone / template shift (same facts) | **Surface form** (markdown, bullets, JSON vs prose) … | **YES** | [↓](#llm_style_or_format) |
| Tabular / clinical — new hospital or cohort | **Cohort / hospital distribution** in the same featu… | **YES** | [↓](#tabular_same_schema) |
| Speech / audio — new mic, room, or channel | **Acoustic channel** (mic, room, codec) — same words… | **YES** | [↓](#speech_or_audio) |
| Frozen embeddings (.npy) — adapt without training CNN | **Feature distribution** between sites; you already … | **YES** | [↓](#frozen_embeddings_sklearn) |
| Known augmentations — robust to blur, color, crop, … | **Named transforms** you apply in training (blur, JP… | **YES** | [↓](#augmentation_robustness) |
| Temporal drift — same patient / session label over time | **Measurement drift over time** (sensor aging, progr… | **YES** | [↓](#temporal_drift) |
| PyTorch Lightning — keep your LightningModule | Same as your underlying task (usually **site / camer… | **YES** | [↓](#pytorch_lightning) |
| Compositional features — nuisance in part of h | **Only some dimensions** of the representation move … | **YES** | [↓](#compositional_coordinates) |
| Other PyTorch task (regression, multi-head, custom) | Whatever **environmental factor** changes between tr… | **TRY** | [↓](#generic_pytorch) |

**YES** = usual fit · **TRY** = validate on deploy metric first.

---

<a id="pose_or_keypoints"></a>

## Pose / keypoints — new camera or site

**Fit:** YES — Good fit when keypoint definitions are identical across cameras/sites and you can pass unlabeled frames from the deploy camera.

| | |
|--|--|
| **What changes** | Camera / studio / hospital **look** changes (lighting, viewpoint, sensor) but each keypoint index still means the same body joint. |
| **Mapping** | D4 · `nuisance='domain_shift'` |
| **Golden path** | [G1 — PyTorch (docs/GOLDEN_PATHS.md#g1)](GOLDEN_PATHS.md#g1) |
| **Data** | Labeled pose data from training site; unlabeled (or labeled) batches from deploy site in the same skeleton format. |
| **Not for** | Different keypoint sets, new body parts at deploy, or no frames from deploy camera. |

**CLI:** `pmh-train route --task pose_or_keypoints` · **Example:** `examples/00_first_run_domain_shift.py`

---

<a id="vision_classification"></a>

## Image classification — new camera or site

**Fit:** YES — Default path for train-on-A / deploy-on-B with the same class names.

| | |
|--|--|
| **What changes** | Image **appearance** shifts (camera, geography, device) while class names and meanings stay fixed. |
| **Mapping** | D4 · `nuisance='domain_shift'` |
| **Golden path** | [G1 — PyTorch (docs/GOLDEN_PATHS.md#g1)](GOLDEN_PATHS.md#g1) |
| **Data** | Source train/val; target domain loader (labels optional for D4). |
| **Not for** | New classes only at test time; different label semantics per site. |

**CLI:** `pmh-train route --task vision_classification` · **Example:** `examples/00_first_run_domain_shift.py`

---

<a id="vision_detection"></a>

## Object detection — same classes, new domain

**Fit:** TRY — PMH applies to the **shared backbone**; you wire source/target image loaders. Box heads and matching are your framework — start with backbone-only shift.

| | |
|--|--|
| **What changes** | Scene **style** and sensor change; bounding-box class IDs unchanged. |
| **Mapping** | D4 · `nuisance='domain_shift'` |
| **Golden path** | [G1 — PyTorch (docs/GOLDEN_PATHS.md#g1)](GOLDEN_PATHS.md#g1) |
| **Data** | Batches of images from train and deploy domains (same class list). |
| **Not for** | Different category sets per region without relabeling. |

**CLI:** `pmh-train route --task vision_detection` · **Example:** `examples/00_first_run_domain_shift.py`

---

<a id="vision_segmentation"></a>

## Segmentation — same classes, new domain

**Fit:** TRY — Same as detection: penalize backbone/encoder `h`, keep pixel loss.

| | |
|--|--|
| **What changes** | Pixel **texture and sensor** shift; per-pixel class IDs (road, person, …) unchanged. |
| **Mapping** | D4 · `nuisance='domain_shift'` |
| **Golden path** | [G1 — PyTorch (docs/GOLDEN_PATHS.md#g1)](GOLDEN_PATHS.md#g1) |
| **Data** | Paired domain image loaders; same label map. |
| **Not for** | New stuff classes at deploy only. |

**CLI:** `pmh-train route --task vision_segmentation` · **Example:** `examples/00_first_run_domain_shift.py`

---

<a id="nlp_text_classification"></a>

## Text classification — new corpus or channel

**Fit:** YES — Encoder hook + source/target text batches; same label set.

| | |
|--|--|
| **What changes** | **Wording and channel** shift (support tickets vs chat) but intent/label set fixed. |
| **Mapping** | D4 · `nuisance='domain_shift'` |
| **Golden path** | [G1 — PyTorch (docs/GOLDEN_PATHS.md#g1)](GOLDEN_PATHS.md#g1) |
| **Data** | Labeled source; target corpus (unlabeled OK for D4). |
| **Not for** | Topic drift that changes label meaning; new intent labels at deploy. |

**CLI:** `pmh-train route --task nlp_text_classification` · **Example:** `examples/08_hf_style_d7.py`

---

<a id="llm_style_or_format"></a>

## LLM — format / tone / template shift (same facts)

**Fit:** YES — Use D7 style pairs when content is fixed but surface form changes.

| | |
|--|--|
| **What changes** | **Surface form** (markdown, bullets, JSON vs prose) — not the underlying facts or instructions. |
| **Mapping** | D7 · `nuisance='style'` |
| **Golden path** | [G3 — HF corpora (docs/GOLDEN_PATHS.md#g3)](GOLDEN_PATHS.md#g3) |
| **Data** | Style pair JSONL or two corpora with matched content. |
| **Not for** | Factual drift, new knowledge at deploy, safety policy changes only. |

**CLI:** `pmh-train route --task llm_style_or_format` · **Example:** `examples/08_hf_style_d7.py`

---

<a id="tabular_same_schema"></a>

## Tabular / clinical — new hospital or cohort

**Fit:** YES — Often G2: frozen features per row, PMHMatcher then sklearn classifier.

| | |
|--|--|
| **What changes** | **Cohort / hospital distribution** in the same feature columns; disease definition unchanged. |
| **Mapping** | D1 · `nuisance='subspace'` |
| **Golden path** | [G2 — sklearn (docs/GOLDEN_PATHS.md#g2)](GOLDEN_PATHS.md#g2) |
| **Data** | Feature matrix + labels on source; features from target cohort. |
| **Not for** | Different schemas, new columns only at deploy, label definition change. |

**CLI:** `pmh-train route --task tabular_same_schema` · **Example:** `examples/06_office31_sklearn.py`

---

<a id="speech_or_audio"></a>

## Speech / audio — new mic, room, or channel

**Fit:** YES — Encoder hook on spectrogram or wav2vec trunk; D4 domain shift.

| | |
|--|--|
| **What changes** | **Acoustic channel** (mic, room, codec) — same words / phoneme labels. |
| **Mapping** | D4 · `nuisance='domain_shift'` |
| **Golden path** | [G1 — PyTorch (docs/GOLDEN_PATHS.md#g1)](GOLDEN_PATHS.md#g1) |
| **Data** | Source transcripts/labels; target-domain audio batches. |
| **Not for** | New vocabulary or language at deploy without relabeling. |

**CLI:** `pmh-train route --task speech_or_audio` · **Example:** `examples/00_first_run_domain_shift.py`

---

<a id="frozen_embeddings_sklearn"></a>

## Frozen embeddings (.npy) — adapt without training CNN

**Fit:** YES — Fastest path — no PyTorch training loop required.

| | |
|--|--|
| **What changes** | **Feature distribution** between sites; you already extracted h and won't fine-tune the encoder. |
| **Mapping** | D4 · `nuisance='domain_shift'` |
| **Golden path** | [G2 — sklearn (docs/GOLDEN_PATHS.md#g2)](GOLDEN_PATHS.md#g2) |
| **Data** | source_features.npy, target_features.npy (+ labels on source). |
| **Not for** | Need to adapt the neural encoder itself (use G1 instead). |

**CLI:** `pmh-train route --task frozen_embeddings_sklearn` · **Example:** `examples/06_office31_sklearn.py`

---

<a id="augmentation_robustness"></a>

## Known augmentations — robust to blur, color, crop, …

**Fit:** YES — You can list finite transforms and run them on training data; PMH estimates sensitivity along those modes (D3).

| | |
|--|--|
| **What changes** | **Named transforms** you apply in training (blur, JPEG, rotation policy) — not an unknown new camera. |
| **Mapping** | D3 · `nuisance='augmentation'` |
| **Golden path** | [G1 — PyTorch (docs/GOLDEN_PATHS.md#g1)](GOLDEN_PATHS.md#g1) |
| **Data** | Labeled train set + code to apply each aug mode; stack aug deltas [M, d]. |
| **Not for** | Unknown deploy camera with no relation to your aug list; use D4 domain_shift instead. |

**CLI:** `pmh-train route --task augmentation_robustness` · **Example:** `examples/18_augmentation_d3.py`

---

<a id="temporal_drift"></a>

## Temporal drift — same patient / session label over time

**Fit:** YES — Sequences [N,T,d] with label fixed over time; D6 estimates drift directions.

| | |
|--|--|
| **What changes** | **Measurement drift over time** (sensor aging, progression) while entity-level label is fixed. |
| **Mapping** | D6 · `nuisance='temporal'` |
| **Golden path** | [G1 — PyTorch (docs/GOLDEN_PATHS.md#g1)](GOLDEN_PATHS.md#g1) |
| **Data** | Batches shaped [N, T, d] with fixed label per sequence; T≥2. |
| **Not for** | Independent snapshots with no time axis; use D4. |

**CLI:** `pmh-train route --task temporal_drift` · **Example:** `examples/00_first_run_domain_shift.py`

---

<a id="pytorch_lightning"></a>

## PyTorch Lightning — keep your LightningModule

**Fit:** YES — Same nuisance as G1; wire Phase A estimate + PMHLoss in training_step.

| | |
|--|--|
| **What changes** | Same as your underlying task (usually **site / camera** D4) — Lightning is the training shell. |
| **Mapping** | D4 · `nuisance='domain_shift'` |
| **Golden path** | [G1b — Lightning (docs/GOLDEN_PATHS.md#g1b)](GOLDEN_PATHS.md#g1) |
| **Data** | source_batches + target_batches for Phase A; task loss in training_step. |
| **Not for** | Plain script with no Lightning — use G1 robust_fit instead. |

**CLI:** `pmh-train route --task pytorch_lightning` · **Example:** `examples/09_lightning_module.py`

---

<a id="compositional_coordinates"></a>

## Compositional features — nuisance in part of h

**Fit:** YES — You know which coordinates of h are nuisance (joints, tokens, atom blocks); PMH uses D5 with nuisance_indices.

| | |
|--|--|
| **What changes** | **Only some dimensions** of the representation move with deploy shift (not the whole vector). |
| **Mapping** | D5 · `nuisance='compositional'` |
| **Golden path** | [G1 — PyTorch (docs/GOLDEN_PATHS.md#g1)](GOLDEN_PATHS.md#g1) |
| **Data** | Feature matrix [N,d] + list of nuisance column indices. |
| **Not for** | Global camera shift with no index structure — use D4. |

**CLI:** `pmh-train route --task compositional_coordinates` · **Example:** `examples/16_qm9_molecule_d5.py`

---

<a id="generic_pytorch"></a>

## Other PyTorch task (regression, multi-head, custom)

**Fit:** TRY — If labels mean the same on A and B and you have a representation `h`, use G1. Otherwise PMH is not automatic — run the gate below.

| | |
|--|--|
| **What changes** | Whatever **environmental factor** changes between train and deploy without changing your target definition. |
| **Mapping** | D4 · `nuisance='domain_shift'` |
| **Golden path** | [G1 — PyTorch (docs/GOLDEN_PATHS.md#g1)](GOLDEN_PATHS.md#g1) |
| **Data** | source_batches + target_batches from deploy environment. |
| **Not for** | Pure i.i.d. training with no deploy domain. |

**CLI:** `pmh-train route --task generic_pytorch` · **Example:** `examples/00_first_run_domain_shift.py`

---

## Not PMH

New classes at deploy or unrelated labels → [WHEN_PMH_HELPS](WHEN_PMH_HELPS.md).

D1–D7 detail: [estimators/index.md](estimators/index.md) (when identification step requires it).
