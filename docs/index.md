# matching-pmh documentation

This package implements the **matching principle** for machine learning:

1. **Estimate** \(\Sigma_{\mathrm{task}}\) — covariance of label-preserving deployment nuisance.
2. **Train** with a matched PMH penalty on representations \(h = \phi(x)\).

It is the public, installable companion to the Grand Unification research line (theory + thirteen empirical blocks). The paper proves *when* range matching is necessary; this library shows *how* to estimate and train in practice.

## Start here

| I want to… | Read |
|------------|------|
| Understand the principle and use it on **my** model | [THEORY.md](THEORY.md) then [README](../README.md) |
| Pick D1 vs D4 vs D7 for my task | [Nuisance cookbook](nuisance_types.md) |
| Run a job from JSON | [CLI](cli.md) |
| Wire PyTorch training | [Training](training.md) |
| Use Hugging Face / DPO | [HF integration](integrations-hf.md), example `11_dpo_lora_style_pmh.py` |
| See symptom → method | [Getting started](getting-started.md) |

## Install

```bash
pip install matching-pmh
pip install "matching-pmh[hf]"      # language-model style (D7)
pip install "matching-pmh[hf-lora]" # DPO + LoRA example
```

## Links

- PyPI: https://pypi.org/project/matching-pmh/
- Source: https://github.com/vishalstark512/matching-pmh
