# 13 Task Patterns

The paper examples are not meant to be copied only as fixed benchmarks. They are **patterns** you can reuse on your own data and architecture.

Use this page like a lookup table:

1. find the task closest to yours;
2. read what production problem happens there;
3. copy the recipe;
4. replace the example model and data with your own.

---

## How to Adapt Any Pattern

For every task, fill four slots:

| Slot | What to write |
|------|---------------|
| Training data | Where your normal labeled data comes from |
| Production data | Where future inputs come from |
| Same answer | The label, mask, box, transcript, score, or property that should mean the same thing |
| Model feature | The layer, embedding, hidden state, or exported feature array PMH can read |

If those four slots are clear, you can usually adapt one of the patterns below.

---

## Pattern Finder

| # | If your task is... | Production problem that happens | Use on your architecture |
|---|--------------------|---------------------------------|--------------------------|
| 1 | Frozen image features / sklearn / Office-style transfer | Features from production cluster differently than training features | Export features from any encoder, then use [existing embeddings](GOLDEN_PATHS.md#use-existing-embeddings-or-sklearn) |
| 2 | ViT / image classifier under sensor noise | Test images have small sensor or embedding noise that should not change the class | Hook CLS/pooled features, then use [train/fine-tune](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| 3 | Medical imaging such as chest X-ray | Hospital, scanner, or acquisition protocol changes while disease labels stay fixed | Hook the image encoder; evaluate on a held-out production hospital |
| 4 | Pose / keypoints | Camera angle, lighting, room, or body framing changes; keypoint order stays fixed | Hook the visual backbone; keep your normal keypoint loss |
| 5 | Depth estimation | Lighting, color, or texture changes; depth target means the same thing | Hook the encoder; keep your normal depth loss |
| 6 | Vision domain shift | New camera, geography, warehouse, robot, or store changes image appearance | Use source and production image loaders with [train/fine-tune](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| 7 | Segmentation / multi-layer vision | Low-level texture and high-level scene style both change | Hook the shared encoder/backbone; start simple, then inspect deeper walkthroughs |
| 8 | Molecules / graphs | Position, conformer, or node-block changes should not change the property target | Use your graph representation; advanced example: [QM9](walkthroughs/14-qm9-molecule-d5.md) |
| 9 | Code models | Imports, comments, identifiers, or token groups change while label stays fixed | Use token or pooled code embeddings; advanced example: [CodeBERT](walkthroughs/15-codebert-tokens-d5.md) |
| 10 | Speech / ASR | Microphone, room, codec, or accent mix changes; transcript/label stays fixed | Hook the audio encoder; example: [Speech](walkthroughs/13-speech-whisper-d4.md) |
| 11 | Time-series / sensors | Sensor drift, device aging, or session drift changes measurements over time | Use sequence features; example: [Temporal](walkthroughs/11-temporal-d6.md) |
| 12 | LLM style / format / tone | Same task, but production asks for JSON, bullets, a new tone, or a new template | Use hidden states or style pairs with [LLM/text style](GOLDEN_PATHS.md#llm-or-text-style) |
| 13 | Adversarial-style perturbations | Small input changes are the production threat | Treat perturbations as the production change; see [published evidence map](PAPER_ALIGNMENT.md) |

---

## What Changes by Architecture?

The task pattern stays the same; only the feature you give PMH changes.

| Your architecture | Feature to use |
|-------------------|----------------|
| ResNet / ConvNet | penultimate or pooled backbone feature |
| ViT / timm / HF ViT | CLS token or pooled patch feature |
| U-Net / segmentation model | encoder or bottleneck feature |
| Detection model | shared backbone before detection heads |
| Pose model | image backbone before keypoint head |
| Whisper / wav2vec-style audio model | audio encoder feature |
| Transformer text model | pooled hidden state or selected layer hidden state |
| Graph neural network | graph-level pooled embedding |
| sklearn / tabular | exported feature matrix |

---

## What Changes by Data?

| Your data situation | What to do |
|---------------------|------------|
| You train the neural model | Use [train/fine-tune](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| You only have embeddings | Use [existing embeddings](GOLDEN_PATHS.md#use-existing-embeddings-or-sklearn) |
| You have text pairs or two text channels | Use [LLM/text style](GOLDEN_PATHS.md#llm-or-text-style) |
| You have a custom known change direction | Use [custom change directions](CUSTOM_GEOMETRY.md) |

---

## What to Prove

For every pattern, the final claim is the same:

> On production-like data, the PMH model beats the normal model and beats sanity checks that use the wrong production change.

That is why the examples include control runs. They are not extra theory; they are how you avoid shipping a lucky regularizer.
