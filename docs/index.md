# matching-pmh

**PMH helps production models handle the common failure mode where training data and production data look different, but the answer should stay the same.**

This is the failure mode behind many "model worked in validation, failed in production" stories: a new hospital, camera, country, sensor, customer segment, microphone, text channel, or prompt format changes the inputs, but the answer is supposed to stay the same.

PMH turns that risk into a concrete workflow:

```mermaid
flowchart LR
  Problem["Deploy world changes"] --> StableLabels["Labels stay stable"]
  StableLabels --> SayChange["Say what changed"]
  SayChange --> Learn["Learn that change"]
  Learn --> Train["Train model to ignore it"]
  Train --> Evidence["Prove on production-like data"]
```

```
production changes, answer stays the same
  -> learn what changed
  -> train the model to ignore that change
  -> trust only if it beats wrong controls
```

---

## Find Your Task

| I want to improve... | Production change | Start |
|----------------------|-------------------|-------|
| Segmentation | new camera, scanner, city, weather, lighting | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Pose / keypoints | new view, studio, camera, room | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Image classification / detection | new device, store, geography, sensor | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Speech / ASR | new mic, room, codec, accent mix | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Time-series / sensors | sensor drift, device aging, session drift | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| Frozen embeddings / sklearn | features already exist from train and production | [Embeddings recipe](GOLDEN_PATHS.md#use-existing-embeddings-or-sklearn) |
| LLM / text | same task, new tone, template, JSON wrapper, channel | [LLM/text recipe](GOLDEN_PATHS.md#llm-or-text-style) |
| Molecules, code, adversarial-style robustness | find closest worked example | [13 task patterns](TASK_PATTERNS.md) |

---

## Start Here

```bash
pip install matching-pmh torch
pmh-train doctor
pmh-train evaluate --demo
```

Then choose one path:

| Your production problem | Use |
|-------------------------|-----|
| Training or fine-tuning a PyTorch model across sites, cameras, sensors, cohorts | [Train/fine-tune recipe](GOLDEN_PATHS.md#train-or-fine-tune-a-model) |
| You already have frozen embeddings, `.npy` files, or sklearn features | [Embeddings recipe](GOLDEN_PATHS.md#use-existing-embeddings-or-sklearn) |
| LLM/text task is the same, but format, tone, template, or channel changes | [LLM/text recipe](GOLDEN_PATHS.md#llm-or-text-style) |
| You are not sure what fits | [Find your application](APPLICATIONS.md) |

---

## The Rule

Use PMH when:

- labels mean the same thing in train and deploy;
- you can collect batches, features, or style pairs from the deploy environment;
- you can evaluate on a deploy holdout before claiming success.

Do not use PMH for new classes, changed label definitions, or "robust to everything" without a deployment story.

---

## The Mental Model

PMH is not a pile of task-specific examples. It is one production loop:

1. **Say what changed.** What is different in production?
2. **Learn the change.** Use examples from training and production.
3. **Train.** Keep your task loss, add PMH so the model cares less about that change.
4. **Prove.** PMH must beat wrong controls on production-like data.

## Read Next

| Need | Page |
|------|------|
| One-page adoption guide | [Adopt PMH](../ADOPT.md) |
| The full production workflow | [Five-step recipe](FIVE_STEP_RECIPE.md) |
| Find your task | [Applications](APPLICATIONS.md) |
| Map the 13 paper tasks to your own model | [13 task patterns](TASK_PATTERNS.md) |
| Copy code | [Production recipes](GOLDEN_PATHS.md) |
| Install, CLI, stack details | [Integrate](INTEGRATE.md) |
| When PMH will not help | [Will PMH help?](WHEN_PMH_HELPS.md) |
| Paper evidence | [Evidence walkthroughs](walkthroughs/index.md) |

[PyPI](https://pypi.org/project/matching-pmh/) · [GitHub](https://github.com/vishalstark512/matching-pmh)
