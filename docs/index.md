# matching-pmh

<p align="center"><em>Name deployment nuisances. Estimate Σ_task. Train with matched PMH on your representations.</em></p>

[**Matching Principle**](THEORY.md) library: estimate label-preserving deployment geometry (D1–D7), then add a matched Jacobian penalty on `h = φ_θ(x)`—on your encoder, not a fixed benchmark recipe.

| | |
|---|---|
| Install | `pip install matching-pmh` |
| Import | `import pmh` |
| CLI | `pmh-train list-methods` |
| PyPI | https://pypi.org/project/matching-pmh/ |

---

## Start here (adoption)

**If you want to use PMH on your own model and data, read in this order:**

1. **[Getting started](GETTING_STARTED.md)** — adoption guide (recommended entry point)  
2. **[Choose your setup](CHOOSE_YOUR_SETUP.md)** — pick API by stack and data  
3. **[Gallery](gallery/README.md)** — copy-paste templates (vision / tabular / NLP)  
4. **[Troubleshooting](TROUBLESHOOTING.md)** — when something breaks  

Then: [Adapt your pipeline](ADAPT_YOUR_PIPELINE.md) checklist · [Hook cookbook](hooks.md) · [18 walkthroughs](walkthroughs/index.md)

---

## 60-second example

```python
from pmh import PMHTrainer, PMHConfig

trainer = PMHTrainer(
    model, hook=backbone, head=head,
    nuisance="domain_shift",
    pmh_config=PMHConfig.balanced(),
    artifact_path="artifacts/sigma.pt",
)
trainer.fit(train_loader, source_batches=src_loader, target_batches=tgt_loader, epochs=20)
```

---

## I want to…

| Goal | Document |
|------|----------|
| **Integrate on my project** | [GETTING_STARTED.md](GETTING_STARTED.md) |
| **Which API / nuisance?** | [CHOOSE_YOUR_SETUP.md](CHOOSE_YOUR_SETUP.md) |
| **ResNet / ViT / HF hook** | [hooks.md](hooks.md) |
| **Copy-paste by domain** | [gallery/README.md](gallery/README.md) |
| **Prove it works (controls)** | [walkthroughs/08-falsification-controls.md](walkthroughs/08-falsification-controls.md) |
| **Math** | [THEORY.md](THEORY.md) |
| **Error / preflight fail** | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |

---

## Two-phase recipe

1. Name label-preserving deployment nuisance  
2. Estimate **Σ_task** (D1–D7)  
3. Train with **PMHLoss** on **h = φ(x)**  
4. Compare **matched** vs **wrong-W** vs **isotropic**  

---

## Install extras

```bash
pip install "matching-pmh[sklearn]"   # PMHMatcher
pip install "matching-pmh[hf]"        # D7 / HFPMHTrainer
pip install "matching-pmh[vision]"  # ResNet examples
```
