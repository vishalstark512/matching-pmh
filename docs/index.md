# matching-pmh

<p align="center"><em>Estimate deployment nuisance geometry. Train any encoder with matched PMH.</em></p>

Reference implementation of the [**Matching Principle**](THEORY.md) for machine learning research and production fine-tuning.

| | |
|---|---|
| Install | `pip install matching-pmh` |
| Import | `import pmh` |
| CLI | `pmh-train list-methods` |
| PyPI | https://pypi.org/project/matching-pmh/ |
| Source | https://github.com/vishalstark512/matching-pmh |

---

## New here?

1. **[Quickstart](QUICKSTART.md)** — running example in 10 minutes  
2. **[Walkthrough 1](walkthroughs/01-pytorch-domain-d4.md)** — PyTorch + domain shift (D4)  
3. **[Walkthrough 8](walkthroughs/08-falsification-controls.md)** — matched vs wrong-W vs isotropic  

Start with **[Adapt your pipeline](ADAPT_YOUR_PIPELINE.md)**, then pick a template from **[17 walkthroughs](walkthroughs/index.md)** (ViT, ResNet, speech, molecules, LLM D7, …).

---

## I want to…

| Goal | Document |
|------|----------|
| Plug into my data & trainer | [ADAPT_YOUR_PIPELINE.md](ADAPT_YOUR_PIPELINE.md) |
| Wire PMH into my model | [ARCHITECTURES.md](ARCHITECTURES.md) |
| Understand the math | [THEORY.md](THEORY.md) |
| See why the API is shaped this way | [PHILOSOPHY.md](PHILOSOPHY.md) |
| Choose D1 vs D4 vs D7 | [Nuisance cookbook](nuisance_types.md) |
| Run JSON / HPC jobs | [CLI](cli.md) |
| Browse runnable scripts | [examples/README.md](../examples/README.md) |
| Hugging Face / DPO | [HF integration](integrations-hf.md) |

---

## Two-phase recipe

```mermaid
flowchart LR
  A[Phase A: estimate Sigma_task] --> B[Phase B: L_task + PMHLoss]
```

1. Name label-preserving deployment nuisance  
2. Estimate with **D1–D7**  
3. Preflight (`pass` / `marginal` / `fail`)  
4. Train on representations $h=\phi(x)$  
5. Report **controls** (wrong-W, isotropic)  

---

## Install extras

```bash
pip install "matching-pmh[vision]"    # ResNet / ViT examples
pip install "matching-pmh[hf-lora]"   # Qwen DPO walkthrough
pip install "matching-pmh[all]"       # dev + docs
```

---

## Citation

See [`CITATION.cff`](../CITATION.cff) and the Grand Unification / Matching Principle manuscript.
