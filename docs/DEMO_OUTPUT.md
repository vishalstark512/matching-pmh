# What you should see (demo output)

Copy of typical terminal output for the **developer onboarding** path. Use this as a README preview if you do not have a screen recording yet.

---

## `python examples/00_first_run_domain_shift.py`

```text
matching-pmh first run (synthetic domain shift)
------------------------------------------------
Target accuracy (baseline ERM):  0.470
Target accuracy (with PMH):      0.530
Preflight: marginal - Weak shift signal. Use more source/target batches or lower rank (see Troubleshooting glossary).

Next: docs/FIRST_HOUR.md  |  pmh-train wizard
      python examples/01_domain_shift_d4.py
```

Numbers vary by seed/hardware; the **pattern** matters: two accuracies + preflight hint.

---

## `pmh-train wizard --non-interactive --stack pytorch`

```text
Recommended: Domain shift (default)
  Source vs target batches; target labels not required.
  nuisance='domain_shift'  stack=pytorch
  Install: pip install matching-pmh torch
  Example: examples/00_first_run_domain_shift.py
  Doc: docs/COLAB.md (or docs/FIRST_HOUR.md)
  Colab: https://colab.research.google.com/github/.../domain_shift_first_run.ipynb

Snippet:
from pmh import PMHTrainer, PMHConfig
trainer = PMHTrainer(
    model, hook=backbone, nuisance='domain_shift',
    pmh_config=PMHConfig.balanced(),
)
trainer.fit(train_loader, source_batches=src, target_batches=tgt, epochs=20)

Next steps:
  1. pip install matching-pmh torch
  2. python examples/00_first_run_domain_shift.py
  3. docs/FIRST_HOUR.md
  ...
```

---

## sklearn Colab / frozen features

```text
Source (500, 128), target pool (325, 128), target test (175, 128)
Target accuracy (baseline, source-only train): 0.xxx
Preflight: pass - Geometry estimate looks usable. Proceed; run controls before large claims.
Target accuracy (PMH adapt + clf):            0.xxx
```

---

## When output looks wrong

| Output | Meaning |
|--------|---------|
| `ModuleNotFoundError: sklearn` | `pip install "matching-pmh[sklearn]"` |
| `ValueError: domain_shift (D4) requires target_batches` | PyTorch path: pass `target_batches=` |
| `preflight=fail` | See [Troubleshooting glossary](TROUBLESHOOTING.md#plain-language-glossary) |
| PMH accuracy much worse than baseline | Tune `PMHConfig.balanced()` or rank; check target holdout leakage |

Record a GIF from this page: run the commands above in a terminal, capture ~20 seconds, add to `docs/assets/` when ready.
