# Adopt PMH

Use this when you want a production model to work better in a new environment: a new camera, hospital, room, cohort, format, or channel.

The rule is simple: **production looks different, but the answer should stay the same**.

---

## Start From Your Task

| I want to improve... | Go to |
|----------------------|-------|
| Segmentation, detection, image classification | [Train/fine-tune a model](docs/GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Pose / keypoints | [Train/fine-tune a model](docs/GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Speech / ASR / audio | [Train/fine-tune a model](docs/GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Time-series / sensor model | [Train/fine-tune a model](docs/GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Frozen embeddings / sklearn / tabular rows | [Use existing embeddings](docs/GOLDEN_PATHS.md#use-existing-embeddings-or-sklearn) |
| LLM or text format/style/channel | [LLM/text style](docs/GOLDEN_PATHS.md#llm-or-text-style) |
| Molecules, code, adversarial-style robustness | [13 task patterns](docs/TASK_PATTERNS.md) |

---

## Quick Check

Answer these before writing code:

| Question | If yes | If no |
|----------|--------|-------|
| Does deploy look, sound, read, or behave differently from train? | Continue | You may not need PMH |
| Do labels mean the same thing in both places? | Continue | PMH is the wrong tool |
| Can you collect deploy batches, features, or style pairs? | Continue | Collect deploy signal first |
| Can you keep a deploy holdout for evaluation? | Continue | Do not claim production robustness |

Examples that fit:

- Hospital A -> Hospital B, same disease labels.
- Warehouse camera -> store camera, same product classes.
- Old customer cohort -> new customer cohort, same churn label.
- Support tickets -> chat messages, same intent labels.
- Markdown answer -> JSON answer, same LLM task.

Examples that do not fit:

- new classes only at deploy;
- a changed policy or taxonomy;
- no deploy data and no concrete shift story.

---

## The Workflow

```mermaid
flowchart LR
  Task["Pick your task"] --> Change["Say what changed"]
  Change --> Learn["Learn that change"]
  Learn --> Train["Train model with PMH"]
  Train --> Prove["Prove on production-like data"]
```

```
pick task -> say what changed -> learn that change -> train -> prove before shipping
```

| Step | What you do | Tool |
|------|-------------|------|
| Pick task | segmentation, pose, speech, LLM, embeddings, etc. | [Find my task](docs/APPLICATIONS.md) |
| Say what changed | camera, hospital, mic, format, cohort, season | `pmh-train route --search camera` |
| Learn the change | use train and production examples | `robust_fit`, `PMHMatcher` |
| Train | keep your normal task loss | PMH recipe |
| Prove | compare against wrong controls | `evaluate_*`, `compare_arms` |

---

## Run This First

```bash
pip install matching-pmh torch
pmh-train doctor
pmh-train evaluate --demo
```

The demo is deliberately small: it shows the Step 5 report shape before you touch your own data.

---

## Pick One Recipe

| Your setup | Use | First move |
|------------|-----|------------|
| Training or fine-tuning a model | [Train/fine-tune](docs/GOLDEN_PATHS.md#train-or-fine-tune-a-model) | `python examples/00_first_run_domain_shift.py` |
| Frozen embeddings, `.npy`, sklearn, tabular features | [Use existing embeddings](docs/GOLDEN_PATHS.md#use-existing-embeddings-or-sklearn) | `pmh-train evaluate --demo` |
| LLM/text style, format, tone, or channel shift | [LLM/text style](docs/GOLDEN_PATHS.md#llm-or-text-style) | `pmh-train route --task llm_style_or_format` |
| Need to find your task | [Applications](docs/APPLICATIONS.md) | `pmh-train route --search hospital` |

Lightning, HF Trainer, and custom saved directions are variants under those recipes, not separate starting points.

---

## Evidence Rule

Do not ship the sentence "PMH improved our model" from one metric alone.

The production claim is:

> PMH improved the production-like holdout, and the improvement was stronger than wrong controls.

That is what separates "we added another regularizer" from "we made the model better for the production change we actually expect."

---

## Read Next

1. [Five-step recipe](docs/FIVE_STEP_RECIPE.md) - the workflow.
2. [Applications](docs/APPLICATIONS.md) - find your production job.
3. [13 task patterns](docs/TASK_PATTERNS.md) - map paper examples to your own data and architecture.
4. [Production recipes](docs/GOLDEN_PATHS.md) - copy code.
5. [Integrate](docs/INTEGRATE.md) - CLI, install, stack details.
6. [Evidence walkthroughs](docs/walkthroughs/index.md) - only after your recipe runs.

[Full README](README.md) · [Docs site](https://vishalstark512.github.io/matching-pmh/) · [PyPI](https://pypi.org/project/matching-pmh/)
