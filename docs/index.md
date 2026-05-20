# matching-pmh

**Train on site A. Deploy on site B. Same labels.**

One package: estimate deployment nuisance → apply matched PMH → validate with controls.

---

## Read in this order

| Step | Page | Time |
|------|------|------|
| 1 | [What is PMH?](WHAT_IS_PMH.md) | 5 min |
| 2 | [Will it help my problem?](WHEN_PMH_HELPS.md) | 5 min |
| 3 | [Your first hour](FIRST_HOUR.md) — install, demo, wizard | 30–60 min |
| 4 | [Pick shift type D1–D7](NUISANCE_SUBTYPES.md) + [Golden paths](GOLDEN_PATHS.md) (G1/G1b/G2/G3/G3b/G4) | 10 min |
| 5 | [Integrate your project](GETTING_STARTED.md) | 1 afternoon |

**Not sure where to start?**

```bash
pip install matching-pmh
pmh-train wizard
```

```python
from pmh import suggest_subtype
print(suggest_subtype(has_target_domain=True, has_target_labels=False))
```

---

## I want to…

| Goal | Go to |
|------|--------|
| Check install | `pmh-train doctor` |
| Load `.npy` / folders | [DATA_LAYOUT.md](DATA_LAYOUT.md) |
| Ship Σ̂ bundle | [DEPLOYMENT.md](DEPLOYMENT.md) |
| Run without installing | [Colab](COLAB.md) |
| Copy-paste by stack | [Golden paths](GOLDEN_PATHS.md) (incl. Lightning **G1b**, HF Trainer **G3b**) |
| Fix an error | [Troubleshooting](TROUBLESHOOTING.md) |
| Compare to CORAL | [vs CORAL](COMPARE_TO_CORAL.md) |
| See example console output | [Demo output](DEMO_OUTPUT.md) |
| Replicate a paper block | [Research → Paper alignment](PAPER_ALIGNMENT.md) · [Recipe cards](recipes/README.md) |
| API details | [Developer API](api/developer.md) · [PMHTrainer](api/pmh-trainer.md) |
| All walkthroughs | [Walkthrough index](walkthroughs/index.md) |

---

## Default code (PyTorch, D4 domain shift)

```python
from pmh import PMHTrainer, PMHConfig

trainer = PMHTrainer(
    model, hook=backbone, nuisance="domain_shift",
    pmh_config=PMHConfig.balanced(),
)
trainer.fit(train_loader, source_batches=src, target_batches=tgt, epochs=20)
```

Install: `pip install matching-pmh` · [PyPI](https://pypi.org/project/matching-pmh/)

---

## Doc sections (sidebar)

| Section | Who it's for |
|---------|----------------|
| **Start here** | Everyone new |
| **Integrate** | Wiring PMH into your stack |
| **Gallery** | Examples by domain (vision / tabular / NLP) |
| **Research** | Paper blocks, benchmarks, fidelity |
| **Reference** | Estimators, training API, theory |
| **Contributors** | Internal plans and doc maintenance |

Older pages ([Quickstart](QUICKSTART.md), [Choose your setup](CHOOSE_YOUR_SETUP.md), …) redirect here or to the pages above.
