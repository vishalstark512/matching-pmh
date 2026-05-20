# matching-pmh

**Train on site A. Deploy on site B. Same labels.**

---

## New here?

1. **[What is matching-pmh?](WHAT_IS_PMH.md)** — no paper required  
2. **[Your first hour](FIRST_HOUR.md)** — install, demo, copy-paste  
3. **[Getting started](GETTING_STARTED.md)** — integrate on your project  

Not sure which API?

```bash
pmh-train wizard
```

```python
from pmh.onboarding import print_setup_guide
print_setup_guide(stack="pytorch", has_target_domain=True)
```

---

## I want to…

| Goal | Document |
|------|----------|
| **Understand the idea** | [WHAT_IS_PMH.md](WHAT_IS_PMH.md) |
| **Will it help my problem?** | [WHEN_PMH_HELPS.md](WHEN_PMH_HELPS.md) |
| **Run a 5-minute demo** | [Colab](COLAB.md) · [FIRST_HOUR](FIRST_HOUR.md) · `examples/00_first_run_domain_shift.py` |
| **Integrate PyTorch / sklearn** | [GETTING_STARTED.md](GETTING_STARTED.md) |
| **Pick by stack** | [CHOOSE_YOUR_SETUP.md](CHOOSE_YOUR_SETUP.md) · [Gallery](gallery/README.md) |
| **Fix errors** | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) · [If you see this error](TROUBLESHOOTING.md#if-you-see-this-error-copy-paste) |
| **What success looks like** | [DEMO_OUTPUT.md](DEMO_OUTPUT.md) |
| **vs CORAL** | [COMPARE_TO_CORAL.md](COMPARE_TO_CORAL.md) |
| **Prove it (controls)** | [Walkthrough 8](walkthroughs/08-falsification-controls.md) |
| **Paper / benchmarks** | [Recipe cards](recipes/README.md) · [CORRECT_USAGE](CORRECT_USAGE.md) · [PAPER_ALIGNMENT](PAPER_ALIGNMENT.md) |

---

## Default code (PyTorch)

```python
from pmh import PMHTrainer, PMHConfig

trainer = PMHTrainer(
    model, hook=backbone, nuisance="domain_shift",
    pmh_config=PMHConfig.balanced(),
)
trainer.fit(train_loader, source_batches=src, target_batches=tgt, epochs=20)
```

Install: `pip install matching-pmh` (v1.5.0+) · PyPI: https://pypi.org/project/matching-pmh/

---

## Research & reference

[Theory](THEORY.md) · [Estimators D1–D7](estimators/index.md) · [Benchmarks](BENCHMARKS.md) · [Onboarding plan](DEVELOPER_ONBOARDING_PLAN.md)
