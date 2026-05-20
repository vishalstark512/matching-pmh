# matching-pmh

**Train on site A. Deploy on site B. Same labels.**

---

## Documentation (one path)

| Step | Page |
|------|------|
| **1** | [**Find your application**](APPLICATIONS.md) — nuisance in plain English + walkthrough |
| **2** | [**Golden paths**](GOLDEN_PATHS.md) — copy **one** of G1 / G1b / G2 / G3 / G3b / G4 |
| **3** | [First hour](FIRST_HOUR.md) — install + demo |
| **4** | [Your project](GETTING_STARTED.md) — afternoon checklist |

**Map of the whole site:** [MAP.md](MAP.md) · **Gates only:** [START_HERE.md](START_HERE.md)

```bash
pip install matching-pmh
pmh-train route --search pose
pmh-train route --task pose_or_keypoints
pmh-train doctor
```

```python
from pmh import explain_task
print(explain_task("pose_or_keypoints"))
```

---

## Sidebar

| Tab | Use |
|-----|-----|
| **Adopt** | Onboarding only — start here |
| **Integrate** | CLI, hooks, data, deployment, APIs |
| **Gallery** | Short templates by domain |
| **Research** | Paper, benchmarks, 19 walkthroughs — **after** Adopt works |
| **Reference** | D1–D7 lemmas, theory, training primitives |
| **Contributors** | How docs are organized |

---

## Default code

```python
from pmh import check_applicability, robust_fit

print(check_applicability(stack="pytorch", n_source=500, n_target=400).summary())
out = robust_fit(
    model, train_loader,
    source_batches=src_loader, target_batches=tgt_loader,
    hook="auto", epochs=20,
)
```

[PyPI](https://pypi.org/project/matching-pmh/) · [GitHub](https://github.com/vishalstark512/matching-pmh)
