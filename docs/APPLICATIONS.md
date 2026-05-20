# Find My Task

Start here if you are thinking:

> I want my model to work better when production looks different from training.

You do **not** need to learn theory terms first. Pick the task closest to yours, copy the recipe, then use the worked example only if you want deeper evidence. For the full set from the paper, use [13 task patterns](TASK_PATTERNS.md).

```bash
pmh-train route --search segmentation
pmh-train route --search pose
pmh-train route --search speech
```

---

## Task Finder

| I want to improve... | Production change | Closest worked example | Start here |
|----------------------|-------------------|-----------------------|------------|
| Image classification | New camera, scanner, geography, store, hospital | Vision domain adaptation | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Segmentation | New city, camera, scanner, weather, lighting | Multi-layer vision block | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Object detection | New scene style, sensor, camera, store layout | Vision domain adaptation | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Pose / keypoints | New studio, camera angle, hospital room | Pose block | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Depth estimation | New lighting, texture, photometric conditions | Depth block | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Medical imaging | New hospital, scanner, protocol, patient mix | CheXpert / vision blocks | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Speech / ASR | New mic, room, codec, accent mix | Whisper / speech block | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Time-series / sensors | Sensor drift, device aging, session drift | HAR / temporal block | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Frozen embeddings | You already exported features from train and deploy | Office-31 block | [Embeddings/sklearn recipe](GOLDEN_PATHS.md#use-existing-embeddings-or-sklearn) |
| Tabular / clinical rows | New hospital or cohort, same columns and label | Office-31-style feature block | [Embeddings/sklearn recipe](GOLDEN_PATHS.md#use-existing-embeddings-or-sklearn) |
| LLM style / format | Same task, new tone, template, JSON wrapper, chat format | Qwen / style block | [LLM/text recipe](GOLDEN_PATHS.md#llm-or-text-style) |
| HF Trainer / DPO / LoRA | Same as above, but keep your trainer stack | Qwen / DPO block | [LLM/text recipe](GOLDEN_PATHS.md#llm-or-text-style) |
| Code models | Token groups or code style changes, same label | CodeBERT block | [Advanced evidence](walkthroughs/15-codebert-tokens-d5.md) |
| Molecules / graphs | Position, conformer, or node block changes | QM9 block | [Advanced evidence](walkthroughs/14-qm9-molecule-d5.md) |
| Adversarial robustness | Perturbations are the production threat | CIFAR PGD block | [Advanced evidence](PAPER_ALIGNMENT.md) |

---

## If You Want the Full Paper Set

The paper covers 13 reusable task patterns: frozen features, ViT, medical imaging, pose, depth, vision domain shift, segmentation-style multi-layer vision, molecules, code, speech, time series, LLM style, and adversarial-style perturbations.

Open [13 task patterns](TASK_PATTERNS.md) to see how each one transfers to your own data and architecture.

---

## CLI Task Shortcuts

These anchors keep `pmh-train route --task ...` stable while the public docs stay task-first.

| CLI task | Human task | Start |
|----------|------------|-------|
| [`pose_or_keypoints`](#pose_or_keypoints) | Pose / keypoints from a new camera or room | [Train/fine-tune](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| [`vision_classification`](#vision_classification) | Image classification in a new visual environment | [Train/fine-tune](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| [`vision_detection`](#vision_detection) | Detection in a new visual environment | [Train/fine-tune](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| [`vision_segmentation`](#vision_segmentation) | Segmentation in a new visual environment | [Train/fine-tune](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| [`nlp_text_classification`](#nlp_text_classification) | Text classification across channels | [Train/fine-tune](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| [`llm_style_or_format`](#llm_style_or_format) | LLM/text style, tone, template, or format changes | [LLM/text](GOLDEN_PATHS.md#llm-or-text-style) |
| [`tabular_same_schema`](#tabular_same_schema) | Tabular rows from a new cohort or hospital | [Embeddings/sklearn](GOLDEN_PATHS.md#use-existing-embeddings-or-sklearn) |
| [`speech_or_audio`](#speech_or_audio) | Speech/audio from a new mic, room, or codec | [Train/fine-tune](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| [`frozen_embeddings_sklearn`](#frozen_embeddings_sklearn) | Exported embeddings or sklearn features | [Embeddings/sklearn](GOLDEN_PATHS.md#use-existing-embeddings-or-sklearn) |
| [`augmentation_robustness`](#augmentation_robustness) | Known transforms such as blur, JPEG, crop, color | [Train/fine-tune](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| [`temporal_drift`](#temporal_drift) | Time-series or sensor drift | [Train/fine-tune](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| [`pytorch_lightning`](#pytorch_lightning) | Existing Lightning project | [Train/fine-tune](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| [`compositional_coordinates`](#compositional_coordinates) | Known feature blocks, token groups, molecules, graph parts | [13 task patterns](TASK_PATTERNS.md) |
| [`generic_pytorch`](#generic_pytorch) | Any custom PyTorch task with stable labels | [Train/fine-tune](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |

<a id="pose_or_keypoints"></a>
<a id="vision_classification"></a>
<a id="vision_detection"></a>
<a id="vision_segmentation"></a>
<a id="nlp_text_classification"></a>
<a id="llm_style_or_format"></a>
<a id="tabular_same_schema"></a>
<a id="speech_or_audio"></a>
<a id="frozen_embeddings_sklearn"></a>
<a id="augmentation_robustness"></a>
<a id="temporal_drift"></a>
<a id="pytorch_lightning"></a>
<a id="compositional_coordinates"></a>
<a id="generic_pytorch"></a>

---

## The Only Question That Matters

For your task, fill this in:

> My model is trained on **A** and deployed on **B**. The input changes because of **X**, but the label still means **Y**.

Examples:

- "trained on warehouse cameras, deployed on store cameras, product class stays the same"
- "trained on Hospital A scans, deployed on Hospital B scans, disease label stays the same"
- "trained on support tickets, deployed on chat messages, intent label stays the same"
- "trained on one prompt format, deployed with JSON output, factual task stays the same"

If you can write that sentence, PMH may fit. If you cannot, fix the task definition first.

---

## What You Need

| Recipe | You need |
|--------|----------|
| Train/fine-tune a model | normal training data, examples from the production environment, one model layer to read features from |
| Use existing embeddings | feature arrays from training and production, plus labels for evaluation |
| LLM/text style | two text sources or style pairs where the task stays the same |

Every path needs a production holdout. That is how you prove the model improved where it matters.

---

## When Not to Use PMH

Do not use PMH if:

- production has new classes;
- label definitions changed;
- the product goal changed;
- you cannot collect any production examples;
- you only want a generic "make it robust" switch.

Read [When PMH helps](WHEN_PMH_HELPS.md) for the boundary cases.
