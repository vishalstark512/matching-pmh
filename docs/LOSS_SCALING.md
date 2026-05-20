# PMH loss scale (5--30% of task loss)

The most common training mistake is letting the PMH term **dominate** or **vanish** relative to your classification (or task) loss. The library **enforces an upper bound** and **warns on a weak lower bound**.

---

## Rule of thumb

| PMH term vs task loss | What it means |
|----------------------|---------------|
| **&lt; 5%** | PMH may be too weak to change representations — increase `weight` or check hook / estimate |
| **5--30%** | Healthy band (default target) |
| **&gt; 30%** | **Automatically capped** — PMH must not swamp the task objective |

This is **not** optional: every `PMHLoss.capped_total` / `MultiLayerPMHLoss` step applies `budget_pmh_to_task_loss` when `cap_basis="task"` (default).

---

## Defaults (`PMHConfig.golden_path()`)

```python
from pmh import PMHConfig

cfg = PMHConfig.golden_path()
# cap_basis="task"
# pmh_max_task_ratio=0.25  → PMH ≤ 25% of task loss
# pmh_min_task_ratio=0.05   → warn if below 5%
# weight=0.3, warmup + ramp
```

Use with `PMHTrainer`, `robust_fit`, or `try_pmh`:

```python
trainer = PMHTrainer(model, hook=hook, pmh_config=PMHConfig.golden_path(), ...)
```

---

## Tune weight once (probe batch)

```python
from pmh.loss_budget import suggest_pmh_weight

w = suggest_pmh_weight(encoder, x_batch, artifact.sigma, task_loss, target_ratio=0.15)
cfg = PMHConfig.golden_path()
cfg.weight = w
```

---

## Read ratios during training

Epoch stats from `train_epoch_with_pmh` include `pmh_task_ratio` when available:

```python
stats = trainer.fit(...)
# stats["pmh_task_ratio"]  → fraction of task loss (epoch average)
```

Per-step warnings fire if PMH is under 5% after warmup (`warn_underpowered_pmh=True`).

---

## Presets

| Preset | Max PMH / task | Notes |
|--------|----------------|-------|
| `conservative()` | 15% | Gentle first run |
| `balanced()` / `golden_path()` | 25% | Default |
| `aggressive()` | 30% | Still capped |

---

## Paper vs product

The paper’s **cap proposition** limits PMH relative to task loss. This repo defaults to **`cap_basis="task"`** so practitioners see “PMH is X% of task loss” directly. Legacy `cap_basis="total"` is still available for reproduction scripts.
