# PMH parameters — one-page cheat sheet

Copy into notebooks or keep beside your training script.

---

## The one line

**Estimate once** (`rank`, `nuisance`) → **train with cap** (`PMHConfig`) → **Step 5** on deploy holdout.

---

## PyTorch / HF (Mode A)

```python
from pmh import PMHConfig, PMHTrainer, robust_fit, evaluate_robust_fit

cfg = PMHConfig.balanced()          # or .conservative() / .aggressive()
# cfg = PMHConfig(weight=0.2, cap_ratio=0.25, warmup_epochs=5)

out = robust_fit(
    model, train_loader,
    source_batches=src, target_batches=tgt,
    hook="auto", head=head,
    rank=16,                    # subspace size — try 8–64
    nuisance="domain_shift",    # or subspace, style, …
    pmh_config=cfg,
    epochs=20,
)

report = evaluate_robust_fit(
    model, train_loader, val_loader,
    source_batches=src, target_batches=tgt,
    hook="auto", head=head, pmh_result=out,
    include_falsification=True,
)
print(report.summary())
```

| Knob | Typical | Notes |
|------|---------|--------|
| `rank` | 8–64 | Higher if many deploy samples |
| `nuisance` | `domain_shift` | **Shift type** API key — see [WHAT_IS_DEPLOYMENT_SHIFT](WHAT_IS_DEPLOYMENT_SHIFT.md) |
| `weight` | 0.15–0.5 | PMH strength |
| `cap_ratio` | 0.2–0.4 | Keeps PMH from dominating task loss |
| `warmup_epochs` | 2–5 | Train task-only first |

---

## sklearn / frozen features (Mode B)

```python
from pmh import evaluate_baseline_vs_pmh, load_g2_demo_arrays

x_s, y_s, x_t, y_t = load_g2_demo_arrays(n=500)
report = evaluate_baseline_vs_pmh(
    x_s, y_s, x_t, y_t,
    rank=16,
    nuisance="domain_shift",
)
print(report.summary())
```

| Knob | Typical | Notes |
|------|---------|--------|
| `rank` | 16–32 | Office-31-style: 32 |
| `nuisance` | `domain_shift` / `subspace` | D1 paired labels → `subspace` |
| `test_size` | 0.35 | Deploy holdout fraction |

No `PMHConfig` on sklearn path (no gradients).

---

## CLI

```bash
pmh-train evaluate --demo                    # sklearn synthetic
pmh-train evaluate --demo --stack pytorch    # PyTorch synthetic
pmh-train evaluate --stack pytorch --source-dir A/ --target-dir B/
pmh-train evaluate --rank 32 --pmh-preset conservative
```

---

## Preflight (after estimate)

| Status | Action |
|--------|--------|
| `pass` | Run Step 5, then tune `weight` / `rank` if needed |
| `marginal` | More target data or lower `rank`; report falsification arms |
| `fail` | Do not ship — fix data / hook / rank first |

Full detail: [PMH_PARAMETERS.md](PMH_PARAMETERS.md)
