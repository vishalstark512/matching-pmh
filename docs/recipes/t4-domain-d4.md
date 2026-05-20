# Recipe: D4 — Domain Gram (exemplar T4, PyTorch Jacobian)

**Preset:** `t4_domain_d4` · **Lemma:** D4 · **Mode:** A (Jacobian penalty on hook `h`)

---

## Use this when

- You **fine-tune in PyTorch** and deployment is a **different site / sensor / corpus** with the **same labels**.
- You can sample **source** and **target** batches (target labels optional for estimation).
- Domain shift is best modeled as **feature-space domain Gram** (default `nuisance="domain_shift"`), not fixed augmentations (→ D3) or style pairs (→ D7).

**Default product path:** [Golden path G1](../GOLDEN_PATHS.md) uses the same nuisance without the paper preset name.

---

## Data contract

| Object | Requirement |
|--------|-------------|
| `model` | `nn.Module` with trainable encoder + head |
| `hook` | Layer with `h` shape `[batch, d]` (or use `hook="auto"`) |
| `source_batches` | Labeled or unlabeled batches from domain A |
| `target_batches` | Batches from domain B (for \(\hat W\) estimate) |
| `train_loader` | Your usual supervised training stream |

---

## Preset defaults

| Field | Value |
|-------|--------|
| `sigma_method` | D4 |
| `default_rank` | **64** |
| `pmh_config` | weight **0.5**, cap **0.3**, warmup **2** epochs |
| `wrong_rank` | 64 |
| `arms` | `b0`, `matched`, `wrong_w`, `isotropic` |
| `application_mode` | `jacobian` |
| `pytorch_benchmark` | `epochs`: 15 (indicative) |

!!! note "Controls naming"
    Training control **`trace_iso`** (alias `isotropic`) ≠ sklearn arm **`isotropic`** (D4 Gram). See [CORRECT_USAGE.md](../CORRECT_USAGE.md).

---

## Minimal code (train + compare arms)

```python
from pmh import compare_arms
from pmh.benchmark.presets import get_preset

p = get_preset("t4_domain_d4")

result = compare_arms(
    model,
    train_loader,
    source_batches=source_loader,
    target_batches=target_loader,
    hook=hook,  # or "auto"
    preset="t4_domain_d4",
    include_geometry=True,
    report_dir="results/t4",
)
```

---

## Developer API (single model)

```python
from pmh import robust_fit, evaluate_robust_fit

out = robust_fit(
    model,
    train_loader,
    source_batches=source_loader,
    target_batches=target_loader,
    hook="auto",
    head=classifier,
    nuisance="domain_shift",
    rank=64,
    epochs=15,
)

report = evaluate_robust_fit(
    model, train_loader, val_loader,
    source_batches=source_loader,
    target_batches=target_loader,
    hook=out.hook_used,
    head=classifier,
    epochs=15,
    pmh_result=out,
)
print(report.summary())
```

---

## Manual trainer (same preset fields)

```python
from pmh import PMHTrainer
from pmh.benchmark.presets import get_preset

p = get_preset("t4_domain_d4")
trainer = PMHTrainer(
    model,
    hook=hook,
    nuisance=p.nuisance,
    rank=p.default_rank,
    pmh_config=p.pmh_config,
)
trainer.fit(
    train_loader,
    source_batches=source_loader,
    target_batches=target_loader,
    epochs=15,
)
```

---

## Falsification arms

| Arm | Meaning |
|-----|---------|
| `b0` | ERM — no PMH |
| `matched` | Penalty along estimated domain nuisance |
| `wrong_w` | Random subspace ⊥ matched W |
| `isotropic` | Training `trace_iso` control (not sklearn D4 arm) |

**Pass:** matched improves target metric vs `b0`; `wrong_w` does not beat matched on both accuracy and geometry ([Walkthrough 8](../walkthroughs/08-falsification-controls.md)).

Paper **E1 pixel-isotropic** arm (multiscale DA) is a separate experiment — not identical to `trace_iso`.

---

## Multilayer (T4 paper)

Paper T4A/B often use **per-layer** domain Gram (`gram_rank` 64). Library:

- [Walkthrough 4 — Multi-layer CNN](../walkthroughs/04-multilayer-convnet.md)
- `MultiLayerPMHLoss` / multiple hooks

Start with single-hook `t4_domain_d4` before adding layers.

---

## Related

| Doc | Purpose |
|-----|---------|
| [Walkthrough 1](../walkthroughs/01-pytorch-domain-d4.md) | Full D4 guide |
| [examples/01_domain_shift_d4.py](https://github.com/vishalstark512/matching-pmh/blob/main/examples/01_domain_shift_d4.py) | Runnable demo |
| [Golden path G1](../GOLDEN_PATHS.md) | Product default |
| [BENCHMARKS.md](../BENCHMARKS.md) | TDI / geometry on val |

**Paper scripts:** `Paper2/T4/Task4A/`, `Paper2/T4/Task4B/`
