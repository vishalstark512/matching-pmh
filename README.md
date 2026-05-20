# matching-pmh

**PMH helps production models handle the everyday failure mode: training data and production data look different, but the answer should stay the same.**

Most AI systems hit this problem before anyone gives it a name:

- a vision model trained on warehouse cameras is deployed in stores;
- a clinical model trained at Hospital A is deployed at Hospital B;
- an audio model trained on one microphone is used in another room;
- a support-ticket classifier sees chat messages instead;
- an LLM workflow keeps the same task but deploys with a new format, tone, or template.

The class labels did not change. The input world did.

PMH gives that failure mode a repeatable engineering loop: **say what changes -> learn that change from data -> train the model to ignore it -> prove it worked before shipping.**

[![PyPI](https://img.shields.io/pypi/v/matching-pmh.svg)](https://pypi.org/project/matching-pmh/)
[![Python](https://img.shields.io/pypi/pyversions/matching-pmh.svg)](https://pypi.org/project/matching-pmh/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/vishalstark512/matching-pmh/blob/main/LICENSE)
[![CI](https://github.com/vishalstark512/matching-pmh/actions/workflows/ci.yml/badge.svg)](https://github.com/vishalstark512/matching-pmh/actions/workflows/ci.yml)

[Docs](https://vishalstark512.github.io/matching-pmh/) · [Adopt](ADOPT.md) · [Production recipes](docs/GOLDEN_PATHS.md) · [PyPI](https://pypi.org/project/matching-pmh/) · [Colab](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/domain_shift_first_run.ipynb)

---

## Find Your Use Case

| I want to improve... | Production change | Start |
|----------------------|-------------------|-------|
| Segmentation | new city, camera, scanner, weather, lighting | [Train/fine-tune recipe](docs/GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Pose / keypoints | new camera angle, studio, hospital room | [Train/fine-tune recipe](docs/GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Image classification | new camera, geography, device, hospital | [Train/fine-tune recipe](docs/GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Object detection | new scene style, sensor, store layout | [Train/fine-tune recipe](docs/GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Speech / ASR | new mic, room, codec, accent mix | [Train/fine-tune recipe](docs/GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Time-series / sensors | sensor drift, device aging, session drift | [Train/fine-tune recipe](docs/GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Frozen embeddings / sklearn | features from train and production already exist | [Embeddings recipe](docs/GOLDEN_PATHS.md#use-existing-embeddings-or-sklearn) |
| Tabular / clinical rows | new hospital or cohort, same columns and label | [Embeddings recipe](docs/GOLDEN_PATHS.md#use-existing-embeddings-or-sklearn) |
| LLM / text workflow | same task, new tone, template, JSON wrapper, channel | [LLM/text recipe](docs/GOLDEN_PATHS.md#llm-or-text-style) |
| Molecules, code, adversarial-style robustness | you want the closest worked example | [13 task patterns](docs/TASK_PATTERNS.md) |

---

## The Problem

Standard training optimizes for the training distribution. Production asks for something harder: **perform well when the environment changes in ways that should not change the answer.**

Data augmentation helps when you already know the transform. Adversarial training helps against worst-case perturbations. Domain adaptation often aligns distributions but does not always prove the improvement came from the deployment factor you care about.

PMH is practical: it asks you to say what changed in production, learn that change from examples, and train the model to be less sensitive to it while keeping your normal task loss primary.

```
Train A + Deploy B, same labels
        -> learn what changed
        -> train the model to ignore that change
        -> ship only if production-like tests beat controls
```

---

## When PMH Is the Right Tool

| You have | PMH fit |
|----------|---------|
| Same label semantics in train and deploy | Yes |
| Batches or features from the deploy environment | Yes |
| A PyTorch, sklearn, HF, or Lightning pipeline you can evaluate | Yes |
| New classes only at deploy | No |
| Label definitions changed between sites | No |
| No deploy data, no style pairs, no shift story | Not yet |

## Install and Smoke Test

```bash
pip install matching-pmh torch
pmh-train doctor
pmh-train evaluate --demo
```

Optional extras:

```bash
pip install "matching-pmh[sklearn]"   # frozen embeddings / tabular features
pip install "matching-pmh[hf]"        # LLM and text pipelines
pip install "matching-pmh[lightning]" # PyTorch Lightning
```

---

## Use PMH in Your Pipeline

1. **Say what changes.** Site, camera, cohort, microphone, prompt format, season?
2. **Check that labels are stable.** If class meaning changed, PMH is the wrong tool.
3. **Learn the change once.** Use training and production batches or frozen features.
4. **Train with PMH.** Keep your normal task loss primary.
5. **Prove before shipping.** PMH must beat wrong controls on a production-like holdout.

```bash
pmh-train shifts
pmh-train route --search hospital
pmh-train route --task vision_classification
```

---

## Copy One Production Recipe

### PyTorch training: new site, camera, sensor, or cohort

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

### Frozen embeddings / sklearn: fastest adoption path

```python
from pmh import evaluate_baseline_vs_pmh, load_g2_demo_arrays

x_source, y_source, x_target, y_target = load_g2_demo_arrays(n=500, seed=0)
report = evaluate_baseline_vs_pmh(x_source, y_source, x_target, y_target)
print(report.summary())  # normal model / PMH / wrong controls
```

### LLM or text style shift

Use PMH when the task is the same but deployment changes wording, format, tone, or template. Start with:

```bash
pmh-train route --task llm_style_or_format
```

Then use the HF recipe in [production recipes](docs/GOLDEN_PATHS.md#llm-or-text-style).

---

## What Counts as Evidence?

A single improved metric is not enough. PMH ships with controls because production robustness claims need a sanity check.

| Check | Meaning |
|-------|---------|
| Normal model | Your usual training or classifier |
| PMH | Train against the production change you identified |
| Wrong-change check | Same machinery pointed at the wrong change |
| Generic-regularizer check | Generic smoothing, not your production change |

The claim is credible only when **PMH wins on production-like data and beats the wrong controls**.

---

## Where to Go Next

| Need | Page |
|------|------|
| One-page field guide | [ADOPT.md](ADOPT.md) |
| The plain production workflow | [Five-step recipe](docs/FIVE_STEP_RECIPE.md) |
| Find your task | [Applications](docs/APPLICATIONS.md) |
| Map the 13 paper tasks to your own model | [13 task patterns](docs/TASK_PATTERNS.md) |
| Copy code | [Production recipes](docs/GOLDEN_PATHS.md) |
| Install, CLI, stack details | [Integrate](docs/INTEGRATE.md) |
| When PMH will not help | [Will PMH help?](docs/WHEN_PMH_HELPS.md) |
| Paper evidence and walkthroughs | [Evidence walkthroughs](docs/walkthroughs/index.md) |

---

## Citation

```bibtex
@software{matching_pmh,
  title  = {matching-pmh: Matched PMH training for deployment shift robustness},
  author = {Rajput, Vishal},
  year   = {2026},
  url    = {https://github.com/vishalstark512/matching-pmh}
}
```

## License

MIT — see [LICENSE](https://github.com/vishalstark512/matching-pmh/blob/main/LICENSE).
