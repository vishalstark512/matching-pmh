# Quickstart

PMH is the **matching principle** from [`main.pdf`](../main.pdf): estimate $\hat{\Sigma}_{\text{task}}$, train with matched PMH on $h$, run **Step 5** on deploy holdout. Pick the [closest task](tasks/index.md) in **T1–T7 order**.

```bash
pip install matching-pmh torch
pip install "matching-pmh[sklearn]"   # T1
pmh-train try --quick                 # train + ship verdict (see docs/START.md)
pmh-train doctor
pmh-train evaluate --demo
pmh-train route --list
```

---

## T1 — frozen features (sklearn, paper block 1)

[Task doc](tasks/t01-classical.md) · [Notebook](../notebooks/tasks/t01-classical.ipynb)

```python
from pmh import load_g2_demo_arrays, evaluate_baseline_vs_pmh

xs, ys, xt, yt = load_g2_demo_arrays()
print(evaluate_baseline_vs_pmh(xs, ys, xt, yt, preset="t1_synthetic_sklearn").summary())
```

---

## T4A — vision domain shift (PyTorch, block 4)

[Task doc](tasks/t04a-vision-domain.md) · [Notebook](../notebooks/tasks/t04a-vision-domain.ipynb)

```python
from pmh import check_applicability

print(check_applicability(stack="pytorch", has_target_domain=True).summary())
```

---

## Step 5 + benchmark arms

```python
from pmh import evaluate_robust_fit, format_five_step_guide
from pmh.benchmark.protocol import run_benchmark_protocol

print(format_five_step_guide("domain_shift"))
# report = run_benchmark_protocol(...)  # matched / wrong / isotropic + geometry
```

Ship only when **matched** beats **wrong-direction** and **isotropic** on deploy holdout ([WHEN_PMH_HELPS](WHEN_PMH_HELPS.md)).

---

## Next

| Goal | Doc |
|------|-----|
| Deploy-change table (T1–T7) | [README](../README.md#find-your-deployment-story-t1-through-t7) |
| All task pages | [tasks/index.md](tasks/index.md) |
| Short theory spine | [PRINCIPLE.md](PRINCIPLE.md) |
| Theory + theorems | [`main.pdf`](../main.pdf) |
| Honest expectations | [Will PMH help?](WHEN_PMH_HELPS.md) |
