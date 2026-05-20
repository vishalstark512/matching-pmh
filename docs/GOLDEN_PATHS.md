# End-to-End Recipes

Pick the path that matches your pipeline and run it end to end. Each recipe has the same shape:

1. run a tiny demo;
2. replace demo data with your training and production data;
3. train or score with PMH;
4. test on production-like data;
5. ship only if PMH beats the sanity checks.

| I want to improve... | Run |
|----------------------|-----|
| segmentation, pose, detection, image classification, speech, sensors, or any model I train | [Train or fine-tune a model](#train-or-fine-tune-a-model) |
| a pipeline that already has embeddings, `.npy` files, tabular features, or sklearn | [Use existing embeddings or sklearn](#use-existing-embeddings-or-sklearn) |
| an LLM/text workflow where style, template, tone, channel, or format changes | [LLM or text style](#llm-or-text-style) |

If you want to start from one of the 13 paper tasks and adapt it to another dataset or architecture, see [13 task patterns](TASK_PATTERNS.md).

---

<a id="train-or-fine-tune-a-model"></a>
<a id="pytorch-training"></a>

## Train or fine-tune a model

Use this for segmentation, pose/keypoints, detection, image classification, medical imaging, speech, time series, text encoders, Lightning modules, or any custom model you train.

Plain English: you have examples from training and production. The label means the same thing. You want the model to care less about what changed between those environments.

### 1. Run the demo

```bash
pip install matching-pmh torch
python examples/00_first_run_domain_shift.py
pmh-train evaluate --demo --stack pytorch
```

### 2. Replace the demo with your data

| Your task | Training data | Production data | Output/label must stay the same |
|-----------|---------------|-----------------|---------------------------------|
| Segmentation | labeled source images + masks | production-like images | same mask classes |
| Pose | labeled source frames + keypoints | production-like frames | same skeleton/keypoint order |
| Detection | labeled source images + boxes | production-like images | same object classes |
| Speech | labeled source audio | production-like audio | same transcripts or labels |
| Time series | labeled source sequences | production-like sequences | same target definition |

You provide three loaders:

```python
train_loader = ...          # normal supervised training data
source_loader = ...         # batches from training environment A
target_loader = ...         # batches from production environment B
deploy_holdout_loader = ... # labeled production-like validation/test data
```

For segmentation, detection, pose, or speech, use your normal model and loss. PMH only needs a feature layer before the task head, often `hook="auto"` or a backbone layer.

### 3. Train with PMH and evaluate

```python
from pmh import check_applicability, evaluate_robust_fit, robust_fit

print(check_applicability(stack="pytorch", n_source=500, n_target=400).summary())

pmh_run = robust_fit(
    model,
    train_loader,
    source_batches=source_loader,
    target_batches=target_loader,
    hook="auto",
    epochs=20,
)

report = evaluate_robust_fit(
    model,
    train_loader,
    deploy_holdout_loader,
    source_batches=source_loader,
    target_batches=target_loader,
    hook="auto",
    pmh_result=pmh_run,
)
print(report.summary())
```

### 4. What to change for your task

| If your task is... | Change |
|--------------------|--------|
| Segmentation | use your segmentation `train_loader`; hook the encoder/backbone before the mask head |
| Pose/keypoints | use your pose loss and keypoint labels; hook the image backbone |
| Detection | keep your box/class losses; start by hooking the shared backbone |
| Speech | hook the audio encoder before the transcript/classification head |
| Lightning | keep your `LightningModule`; see [Lightning walkthrough](walkthroughs/10-lightning.md) |

The report should compare your normal model, PMH, and sanity checks on `deploy_holdout_loader`.

---

<a id="use-existing-embeddings-or-sklearn"></a>
<a id="frozen-features--sklearn"></a>

## Use existing embeddings or sklearn

Use this when your pipeline already exports features, embeddings, tabular rows, or `.npy` arrays and you want to test PMH without retraining the encoder.

### 1. Run the demo

```bash
pip install "matching-pmh[sklearn]"
pmh-train evaluate --demo
```

### 2. Replace the demo arrays

You need arrays like this:

```python
x_source = ...  # features from training environment A, shape [n, d]
y_source = ...  # labels for A
x_target = ...  # features from production environment B, shape [m, d]
y_target = ...  # labels for production-like holdout B
```

This fits:

- CLIP/ViT/ResNet embeddings exported from image folders;
- tabular features from two hospitals or customer cohorts;
- CRM/churn features from two regions;
- frozen encoder outputs from any model.

### 3. Run the full comparison

```python
from pmh import evaluate_baseline_vs_pmh, load_g2_demo_arrays

x_source, y_source, x_target, y_target = load_g2_demo_arrays(n=500, seed=0)
report = evaluate_baseline_vs_pmh(x_source, y_source, x_target, y_target)
print(report.summary())
```

Then replace `load_g2_demo_arrays(...)` with your arrays.

### 4. Folder layout option

```text
site_a/
  features.npy
  labels.npy
site_b/
  features.npy
  labels.npy
```

```bash
pmh-train evaluate --source-dir site_a --target-dir site_b
```

---

<a id="llm-or-text-style"></a>

## LLM or text style

Use this when the underlying task is stable but production changes surface form: chat vs ticket, Markdown vs JSON, bullets vs paragraphs, formal vs casual, one prompt template vs another.

This is not for "new facts" or a changed policy. It is for the same task under a different text channel or format.

### 1. Route your task

```bash
pip install "matching-pmh[hf]"
pmh-train route --task llm_style_or_format
```

### 2. Prepare texts

You need either two corpora for the same task:

```python
texts_a = [...]  # training channel, old prompt, old format
texts_b = [...]  # production channel, new prompt, new format
```

or style pairs where the content stays fixed:

```json
{"content": "same underlying answer", "style_a": "paragraph", "style_b": "json"}
```

### 3. Train or fine-tune

```python
from pmh import check_applicability, robust_fit_text_domains

print(check_applicability(stack="hf", n_source=len(texts_a), n_target=len(texts_b)).summary())

pmh_run = robust_fit_text_domains(
    model,
    tokenizer,
    train_loader,
    source_texts=texts_a,
    target_texts=texts_b,
    epochs=3,
    rank=32,
)
print(pmh_run.preflight_message)
```

### 4. Evaluate

Use your normal LLM/text metric on production-format examples. Then use the PMH report to check that the improvement is not just a generic penalty.

---

## Variants

| If you use... | Do this |
|---------------|---------|
| PyTorch Lightning | Put the same PMH loss in `training_step`; see [Lightning walkthrough](walkthroughs/10-lightning.md) |
| Hugging Face `Trainer`, DPO, or LoRA | Keep your trainer and add PMH in the loss; see [HF walkthrough](walkthroughs/07-hf-trainer-d7-dpo.md) |
| ResNet / torchvision | Use the penultimate feature layer; see [ResNet walkthrough](walkthroughs/02-resnet-vision-d4.md) |
| ViT / timm / HF ViT | Use CLS or pooled patch features; see [ViT walkthrough](walkthroughs/12-vit-cls-d4.md) |
| Your own saved change directions | Use [custom change directions](CUSTOM_GEOMETRY.md) |

Older anchors still work:

<a id="g1"></a>
<a id="g1b"></a>
<a id="g2"></a>
<a id="g3"></a>
<a id="g3b"></a>
<a id="g4"></a>

---

## Before Production

```bash
pmh-train doctor --stack pytorch
```

1. Check that labels mean the same thing in training and production.
2. Keep a production-like holdout.
3. Run the recipe.
4. Compare your normal model, PMH, and wrong controls.
5. Ship only if PMH wins on the production-like metric.
