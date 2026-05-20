# PMH parameters (what you can change)

Yes — **all main knobs are user-settable**. You do not need to fork the library.

**One-page copy-paste:** [PARAMETERS_CHEATSHEET.md](PARAMETERS_CHEATSHEET.md) (also at the top of each Colab notebook).

---

## Two phases, two config objects

| Phase | Object | What it controls |
|-------|--------|------------------|
| **Estimate** (once) | `SigmaTaskConfig` | How \(\hat\Sigma_{\mathrm{task}}\) is built (D1–D7 / `nuisance`) |
| **Train** (loop) | `PMHConfig` | How strong the PMH penalty is vs task loss |

---

## `PMHConfig` — training penalty (Mode A: PyTorch / HF)

Used by `PMHTrainer`, `PMHLoss`, `robust_fit`, Lightning callback.

| Field | Default (`balanced()`) | Meaning |
|-------|------------------------|---------|
| `weight` | `0.3` | PMH term scale before capping |
| `cap_ratio` | `0.3` | Max PMH / (task + PMH) per step |
| `cap_basis` | `"total"` | Cap relative to total or task loss only |
| `n_probes` | `4` | Random directions for Jacobian penalty |
| `shrinkage` | `1e-6` | Numerical stability in penalty |
| `warmup_epochs` | `2` | Epochs with **no** PMH |
| `warmup_ramp_epochs` | `10` | Linear ramp to full `weight` |

**Presets** (copy or tweak):

```python
from pmh import PMHConfig

PMHConfig.conservative()   # gentler first try
PMHConfig.balanced()       # default in robust_fit / PMHTrainer
PMHConfig.aggressive()     # stronger regularization
PMHConfig.finetune_llm()   # long warmup for LLMs
```

**Custom:**

```python
cfg = PMHConfig(weight=0.2, cap_ratio=0.25, warmup_epochs=5, warmup_ramp_epochs=15)
trainer = PMHTrainer(model, hook=backbone, pmh_config=cfg, rank=32)
```

---

## `SigmaTaskConfig` / estimate knobs

| Knob | Where | Typical use |
|------|-------|-------------|
| `nuisance` | `PMHTrainer`, `PMHMatcher`, `robust_fit` | `domain_shift`, `subspace`, `style`, … |
| `rank` | Trainer / Matcher / `evaluate_*` | Subspace dimension \(r\) (D1/D4/D7) |
| `shrinkage` | `PMHTrainer` | Covariance regularization |
| `method` | `SigmaTaskConfig.for_*()` | Force D1–D7 estimator |

```python
from pmh import PMHTrainer, PMHConfig

trainer = PMHTrainer(
    model,
    hook=backbone,
    nuisance="domain_shift",
    rank=32,
    shrinkage=1e-5,
    pmh_config=PMHConfig.conservative(),
)
```

---

## Mode B (sklearn / frozen features)

`PMHMatcher` uses **estimate** knobs (`nuisance`, `rank`, `seed`), not `PMHConfig` (no gradient training).

```python
from pmh import PMHMatcher

matcher = PMHMatcher(nuisance="subspace", rank=32, seed=0)
matcher.fit(x_source, y_source, x_target, y_target)
```

CLI:

```bash
pmh-train evaluate --demo --rank 32 --nuisance domain_shift
```

---

## Preflight (read-only gate)

After estimate, `artifact.preflight` is `pass` / `marginal` / `fail` from eigengap — not a training hyperparameter. Fix with **more deploy data**, different `rank`, or another `nuisance` (see [TROUBLESHOOTING](TROUBLESHOOTING.md)).

---

## What not to tune first

- Paper block presets (`pmh-train list-presets`) — replication only  
- Internal `wrong_seed` / geometry probes — use `compare_arms` / `evaluate_*` instead  

**Newbie order:** `PMHConfig.balanced()` + `rank=16` or `32` → run Step 5 → then tune `weight` / `cap_ratio` if matched beats controls but task loss is unstable.
