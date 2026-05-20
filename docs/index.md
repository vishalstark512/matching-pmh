# matching-pmh

**Train on site A. Deploy on site B. Same labels.**

Estimate deployment geometry once → train with capped PMH → compare **matched / wrong-W / isotropic** on **deploy holdout**.

---

## Read this (4 pages)

| # | Page |
|---|------|
| 0 | [**What is deployment shift?**](WHAT_IS_DEPLOYMENT_SHIFT.md) — plain English (read if “nuisance” confuses you) |
| 1 | [**Five-step recipe**](FIVE_STEP_RECIPE.md) — the whole product |
| 2 | [**Applications**](APPLICATIONS.md) + `pmh-train route --task …` |
| 3 | [**Golden paths**](GOLDEN_PATHS.md) — copy **one** of G1–G4 |
| 4 | [**Integrate**](INTEGRATE.md) — install, CLI, your stack |

Then: [Will PMH help?](WHEN_PMH_HELPS.md) · [Troubleshooting](TROUBLESHOOTING.md)

**Paper / benchmarks (optional):** [Evidence walkthroughs](walkthroughs/index.md)

```bash
pip install matching-pmh
pmh-train recipe
pmh-train route --task pose_or_keypoints
```

```python
from pmh import robust_fit, check_applicability
print(check_applicability(stack="pytorch", n_source=500, n_target=400).summary())
```

[PyPI](https://pypi.org/project/matching-pmh/) · [GitHub](https://github.com/vishalstark512/matching-pmh)
