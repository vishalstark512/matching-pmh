# Recipe: D2 — Isotropic sensitivity (exemplar T2A, ViT CLS)

**Preset:** `t2a_vit_isotropic` · **Lemma:** D2 · **Mode:** A (Jacobian on CLS / pooled tokens)

---

## Use this when

- **Vision Transformer** (timm, HF ViT) on images with **domain shift** (new camera, site, scanner).
- Paper block **T2A** uses **isotropic input noise** on representations (σ=0.10) — matched PMH ≡ isotropic penalty on `h`.
- You want **ERM vs isotropic PMH** only — **no wrong-W arm** in this block (by paper design).

**Not this preset:** cross-domain Gram shift → use [T4 domain D4](t4-domain-d4.md) with `t4_domain_d4` and `nuisance="domain_shift"`.

---

## Data contract

| Object | Requirement |
|--------|-------------|
| `model` | ViT or timm model + classification head |
| `hook` | CLS token or mean pool — `encoder_timm(..., pool="cls")` |
| `source_batches` / `target_batches` | Image batches (target labels optional for D2 estimate) |
| `train_loader` | Supervised training stream |

---

## Preset defaults

| Field | Value |
|-------|--------|
| `sigma_method` | D2 |
| `nuisance` | `isotropic` |
| `estimate_kwargs` | `noise_level` **0.10** |
| `pmh_config` | weight **0.3**, cap **0.3**, warmup **0** |
| `arms` | `b0`, `matched` only |
| `application_mode` | `jacobian` |

!!! note
    D2 matched and isotropic training controls coincide; falsification is **ERM vs PMH**, not wrong-W.

---

## Minimal code

```python
from pmh.hooks import encoder_timm
from pmh import PMHTrainer
from pmh.benchmark.presets import get_preset

p = get_preset("t2a_vit_isotropic")
hook = encoder_timm(vit_model, layer="blocks", pool="cls")

trainer = PMHTrainer(
    model,
    hook=hook,
    nuisance=p.nuisance,
    pmh_config=p.pmh_config,
)
trainer.fit(
    train_loader,
    source_batches=source_loader,
    target_batches=target_loader,
    epochs=20,
)
```

Trajectory TDI (paper metric): `trainer.measure_trajectory_tdi(val_loader, sigma=0.01)`.

---

## Hooks

| Stack | Hook helper |
|-------|-------------|
| timm ViT | `encoder_timm(model, layer="blocks", pool="cls")` |
| HF ViT | `encoder_hf_hidden_states(model, layer=-1, pool="cls")` |

See [hooks.md](../hooks.md) · [Walkthrough 12](../walkthroughs/12-vit-cls-d4.md).

---

## Falsification (T2A)

| Arm | Expectation |
|-----|-------------|
| `b0` | ERM baseline |
| `matched` | Isotropic PMH at σ=0.10 |

Compare target metric + `trajectory_tdi` — not wrong-W (omitted by design).

---

## Related

| Doc | Purpose |
|-----|---------|
| [Walkthrough 12](../walkthroughs/12-vit-cls-d4.md) | Full ViT guide |
| [examples/14_vit_cls_d4.py](https://github.com/vishalstark512/matching-pmh/blob/main/examples/14_vit_cls_d4.py) | Runnable demo |
| [Recipe T4](t4-domain-d4.md) | Domain Gram (D4) instead of D2 |

**Paper:** `Paper2/T2/Task2A/`
