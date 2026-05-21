# Start here (no paper required)

**Train on site A. Deploy on site B. Same labels.**

You do **not** need to know nuisance types (D1–D7). The library infers the shift type from what data you have, trains, then tells you **ship** or **do not ship** on deploy holdout.

**Note:** Published block numbers are in [`main.pdf`](../main.pdf) and [findings.html](findings.html). The library is for general integration — it follows the same recipe but **will not automatically replicate** those results; plan on iteration until Step 5 passes on your deploy holdout.

---

## One function (PyTorch)

```python
from pmh import try_pmh
from pmh.pytorch_eval import pytorch_demo_loaders

bundle = pytorch_demo_loaders(n=400, seed=0)
report = try_pmh(
    bundle.model,
    bundle.train_loader,
    bundle.val_loader,
    source_batches=bundle.source_batches,
    target_batches=bundle.target_batches,
    hook=bundle.encoder,
    head=bundle.head,
    epochs=5,
)
print(report.deploy_summary())
print(report.ship_verdict())
report.save_html("deploy_report.html")  # optional Step 5 one-pager
```

- **`nuisance=None`** (default inside `robust_fit`) — auto-picked from your flags  
- **`report.ship_verdict()`** — plain English Step 5 outcome  
- **`report.deploy_summary()`** — baseline vs matched vs controls  
- **`PMHConfig.golden_path()`** — PMH term **capped at 25%** of task loss (warn if below 5%); see [LOSS_SCALING.md](LOSS_SCALING.md)  

Fast CPU smoke:

```bash
pmh-train try --quick
pmh-train try --quick --html deploy_report.html
# or: PMH_QUICK=1 python scripts/demos/first_run_domain_shift.py
```

Paper block synthesis (not your deploy numbers): [findings.html](findings.html).

---

## Run in Colab (no local install)

| Demo | Open |
|------|------|
| **T1** frozen features (sklearn) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/tasks/t01-classical.ipynb) |
| **T4A** domain shift (PyTorch) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/tasks/t04a-vision-domain.ipynb) |
| **T4B** multilayer RGB CNN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/tasks/t04b-multilayer-vision.ipynb) |

In Colab: `!pip install -q matching-pmh torch` then Run All, or paste the `try_pmh` block above with demo loaders.

---

## Tell us what data you have (not the lemma name)

```python
from pmh import infer_applicability, format_shift_types

print(infer_applicability(
    has_target_domain=True,      # batches from deploy site B
    has_target_labels=False,     # set True if B labels exist
    has_augmentation_modes=False,
    has_style_pairs=False,
).summary())

print(format_shift_types())  # plain-language shift catalog
```

| You have… | Set flag | Typical auto pick |
|-----------|----------|-------------------|
| Deploy site batches, no labels | `has_target_domain=True` | `domain_shift` |
| Labels on source **and** deploy | `has_target_labels=True` | `subspace` |
| Same content, two text formats | `has_style_pairs=True` | `style` |
| Named aug modes (blur, crop, …) | `has_augmentation_modes=True` | `augmentation` |

---

## Frozen features (sklearn)

```python
from pmh import load_g2_demo_arrays, evaluate_baseline_vs_pmh

xs, ys, xt, yt = load_g2_demo_arrays()
report = evaluate_baseline_vs_pmh(xs, ys, xt, yt)
print(report.deploy_summary())
```

See [MIGRATE.md](MIGRATE.md) if you already use CORAL or a `Pipeline`.

---

## Interactive setup

```bash
pmh-train try --quick              # golden path CLI
pmh-train try --stack sklearn      # frozen-feature demo + CORAL in report
pmh-train doctor
python -c "from pmh import run_wizard; run_wizard()"
```

---

## Theory on demand

- Short spine: [PRINCIPLE.md](PRINCIPLE.md)  
- Full proofs: [`main.pdf`](../main.pdf)  
- Task examples (T1–T7): [tasks/index.md](tasks/index.md)  
