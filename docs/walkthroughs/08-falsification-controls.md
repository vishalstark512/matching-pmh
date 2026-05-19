# Walkthrough 8: Falsification controls

**Goal:** Show improvements come from **matched** $\Sigma_{\mathrm{task}}$, not generic Jacobian shrinkage.

**Script:** `examples/04_falsification_controls.py`

---

## The three (four) arms

| Arm | `PMHLoss(..., mode=)` | Theory |
|-----|------------------------|--------|
| **Matched** | `"matched"` (default) | $\Sigma' \approx \hat\Sigma_{\mathrm{task}}$ |
| **Wrong-W** | `"wrong_w"` | Random rank-$r$ subspace → ≈ uninformative |
| **Isotropic** | `"isotropic"` | $\Sigma' \propto I$ (VAT-like) |
| **Signal-W** | projector utilities / custom | Should **hurt** task metric |

---

## Minimal comparison

```python
from pmh import PMHLoss, PMHConfig, SigmaTaskConfig, estimate_from_config

artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=8), h_src, h_tgt)
cfg = PMHConfig(weight=1.0, cap_ratio=0.3, warmup_epochs=0)

h = backbone(x)  # requires_grad True
task = criterion(head(h), y)

for name, mode in [("matched", "matched"), ("wrong_w", "wrong_w"), ("iso", "isotropic")]:
    pmh = PMHLoss(artifact, cfg, mode=mode)
    total, raw = pmh.capped_total(task, h)
    # train separate runs; log target-domain metric per run
```

---

## Run the toy script

```bash
python examples/04_falsification_controls.py
```

Prints three scalar PMH values on one batch (single seed; for training claims, average over seeds / full runs).

---

## Reporting checklist

For each benchmark table row:

1. **B0** — ERM / no PMH
2. **Matched** — your Dk estimate
3. **Wrong-W** — same rank as matched
4. **Isotropic** — matched trace scale
5. (Optional) **Signal-W** — hurts when signal subspace is known

**Acceptable claim pattern:**

- Matched > B0 on **deployment** metric
- Wrong-W ≈ isotropic (not better than matched)
- Signal-W worse than B0 when applicable

**Weak claim:** matched > B0 but wrong-W also wins → may be generic regularization.

---

## Office-31 analogue

`06_office31_sklearn.py` implements matched projection vs wrong-W vs CORAL in sklearn space—see [Walkthrough 3](03-office31-sklearn-d1.md).
