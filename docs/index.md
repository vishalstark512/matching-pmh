# matching-pmh documentation

**Train on site A. Deploy on site B. Same labels.**

This library implements the **Perturbation Matching Hypothesis** from [`main.pdf`](../main.pdf): estimate $\Sigma_{\text{task}}$ (covariance of label-preserving deployment change), train with a **matched** PMH loss on encoder Jacobian $J_\varphi$, then require **Step 5 evidence** on deploy holdout (matched vs wrong-direction vs isotropic).

The thirteen tasks are **worked examples** of one theory — not thirteen different products.

**Paper vs library:** Metrics cited in tasks and the paper come from **`paper_code/`** reproduction scripts. The **`pmh` library** is for your pipeline; expect tuning — it does not guarantee the same numbers as `FINAL.md` without that block-specific setup.

---

## Start here (practitioners)

1. **[START.md](START.md)** — one function (`try_pmh`), ship verdict, auto shift type (no D1–D7 required).
2. **[MIGRATE.md](MIGRATE.md)** — if you already use CORAL, augmentation, HF, or PGD.  
   **[LOSS_SCALING.md](LOSS_SCALING.md)** — PMH must stay ~5--30% of task loss (enforced).  
   **[GLOSSARY.md](GLOSSARY.md)** — plain words, not D1--D7 first.
3. **One command:** `pmh-train try --quick` (auto shift type → deploy report → ship verdict).
4. **Interactive:** `pmh-train doctor` or `python -c "from pmh import run_wizard; run_wizard()"`

## Read in this order

5. **[README](../README.md)** — principle, five-step recipe, T1–T7 deploy table.
6. **[PRINCIPLE.md](PRINCIPLE.md)** — short theory spine (optional before `main.pdf`).
7. **[Quickstart](QUICKSTART.md)** — install and commands.
8. **[13 tasks](tasks/index.md)** — examples by deploy change.
9. **[Will PMH help?](WHEN_PMH_HELPS.md)** — honest expectations + controls.
10. **[`main.pdf`](../main.pdf)** — full proofs (on demand).
11. **[API](api/index.md)** — reference.

---

## Five steps (same in every stack)

| Step | PyTorch | sklearn | HF |
|------|---------|---------|-----|
| Estimate $\hat{\Sigma}_{\text{task}}$ | `PMHTrainer.estimate` | `PMHMatcher.fit` | style pairs / D7 |
| Apply matched PMH | `trainer.fit` / `robust_fit` | `Pipeline` + head | LM loop + artifact |
| Evidence (Step 5) | `evaluate_robust_fit` | `evaluate_baseline_vs_pmh` | holdout + report |

`nuisance=` selects the estimator row (D1–D7); the step structure does not change.

---

## Quick links

[13 tasks](tasks/index.md) · [**Paper findings (HTML)**](findings.html) · [Notebooks](../notebooks/README.md) · [WHEN_PMH_HELPS](WHEN_PMH_HELPS.md) · [API](api/index.md)

Regenerate findings page: `python scripts/build_findings_html.py`
