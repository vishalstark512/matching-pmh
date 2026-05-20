# Paper block presets (T1–T7)

Use these when you want **library defaults** aligned with `Paper2/T1`–`T7` scripts — shift type (`nuisance=`), rank, PMH weights, and (for sklearn) the T1 pool/test protocol.

> **API note:** `nuisance=` names the **deployment shift type** (D1–D7), not bad data. [What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)

```bash
pmh-train list-presets
```

See also [CORRECT_USAGE.md](../CORRECT_USAGE.md) and [PAPER_ALIGNMENT.md](../PAPER_ALIGNMENT.md).

---

## Quick map

**One-page recipes:** [T1](../PAPER_ALIGNMENT.md) · [T2A](../PAPER_ALIGNMENT.md) · [T4](../PAPER_ALIGNMENT.md) · [T7A](../PAPER_ALIGNMENT.md)

| Preset | Block | Estimator | Walkthrough |
|--------|-------|-----------|-------------|
| `t1_office31_sklearn` | T1 | D1 | [Recipe card](../PAPER_ALIGNMENT.md) · [3 — Office-31 sklearn](03-office31-sklearn-d1.md) |
| `t1_synthetic_sklearn` | T1 | D1 | [3](03-office31-sklearn-d1.md) (Path A) |
| `t2a_vit_isotropic` | T2A | D2 | [Recipe](../PAPER_ALIGNMENT.md) · [12 — ViT CLS](12-vit-cls-d4.md) |
| `t2b_chexpert_isotropic` | T2B | D2 | gallery / chest X-ray |
| `t3b_depth_d3` | T3B | D3 | [16 — Augmentations](16-augmentation-d3.md) |
| `t4_domain_d4` | T4 | D4 | [Recipe card](../PAPER_ALIGNMENT.md) · [1 — PyTorch D4](01-pytorch-domain-d4.md) |
| `t5_compositional_d5` | T5 | D5 | [5 — Compositional](05-compositional-d5.md) |
| `t6_temporal_d6` | T6 | D6 | [11 — Temporal](11-temporal-d6.md) |
| `t7a_style_d7` | T7A | D7 | [Recipe](../PAPER_ALIGNMENT.md) · [6 — LLM style](06-llm-style-d7.md) |
| `t7b_pgd_d7` | T7B | D7 | PGD / adversarial δ |

---

## Sklearn (frozen features)

```python
from pmh import compare_arms_sklearn

# T1 Office-31: rank 32, pool=200, test=250, paper_protocol=True
result = compare_arms_sklearn(
    x_src, y_src, x_tgt, y_tgt,
    preset="t1_office31_sklearn",
    report_dir="results/t1",
)
```

Multi-seed (paper-style):

```python
compare_arms_sklearn(..., preset="t1_office31_sklearn", seeds=[0, 42, 142])
```

---

## PyTorch (training + compare)

```python
from pmh import compare_arms, PMHTrainer, PMHConfig
from pmh.benchmark.presets import get_preset

p = get_preset("t4_domain_d4")
trainer = PMHTrainer(
    model,
    hook=hook,
    nuisance=p.nuisance,
    rank=p.default_rank,
    pmh_config=p.pmh_config,
)

compare_arms(
    model,
    train_loader,
    source_batches=src,
    target_batches=tgt,
    hook=hook,
    preset="t4_domain_d4",
    include_geometry=True,
)
```

ViT isotropic (T2A — Jacobian, no wrong-W in paper):

```python
trainer = PMHTrainer(
    model,
    hook=hook,
    nuisance="isotropic",
    pmh_config=get_preset("t2a_vit_isotropic").pmh_config,
)
```

LLM style (T7A):

```python
cfg = get_preset("t7a_style_d7")
# estimate: SigmaTaskConfig.for_alignment(rank=128, shrinkage=0.1)
# train: PMHConfig(weight=0.7, cap_ratio=0.3, warmup_epochs=5)
```

---

## Notes

- **T1 Office-31** reference table is a **protocol check**, not a headline PMH win (CORAL can beat matched on a linear head).
- **Sklearn `isotropic` arm** = D4 domain Gram (unmatched); **PyTorch `trace_iso`** is a different training control — see [CORRECT_USAGE.md](../CORRECT_USAGE.md).
- Do **not** cite pre-2026-05-19 benchmark rows with B0 ≈ 0.7 / rank 16.
