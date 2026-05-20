# Recipe: D1 — Cross-domain subspace (exemplar T1, Office-31)

**Preset:** `t1_office31_sklearn` · **Lemma:** D1 · **Mode:** B (feature projection, not end-to-end Jacobian)

---

## Use this when

- You have **frozen embeddings** (e.g. ResNet-18 pool) from source and target domains.
- **Labels exist on both** domains (D1 class-aligned cross-domain SVD).
- You want a **paper-faithful** sklearn benchmark with falsification arms and CORAL baseline.

**Do not use for:** end-to-end CNN fine-tuning (use [T4 domain D4](t4-domain-d4.md) or custom `PMHTrainer`). **Do not expect** matched PMH to beat CORAL on this linear setup — see [When PMH helps](../WHEN_PMH_HELPS.md#honest-reference-numbers-do-not-cherry-pick).

---

## Data contract

| Tensor | Shape | Notes |
|--------|-------|--------|
| `x_source`, `y_source` | `[N_s, d]` | Train region (paper: 1500) |
| `x_target`, `y_target` | `[N_t, d]` | Same `d`, same label semantics |
| Protocol | pool **200** + test **250** | `paper_protocol=True` — W from pool only, metric on test |

Extract features locally; do not commit `.npy` or images ([DATA_POLICY.md](../DATA_POLICY.md)).

---

## Preset defaults

| Field | Value |
|-------|--------|
| `sigma_method` | D1 |
| `default_rank` | **32** |
| `n_pairs_per_class` | 40 |
| `sklearn_benchmark` | `paper_protocol=True`, pool=200, test=250 |
| `arms` | `b0`, `matched`, `wrong_w`, `isotropic`, `coral` |
| `application_mode` | `projection` |

!!! note "Isotropic arm"
    Sklearn **`isotropic`** = **D4 domain Gram** (unmatched control), not D2 input noise.

---

## Minimal code (falsification table)

```python
from pmh import compare_arms_sklearn

result = compare_arms_sklearn(
    x_source, y_source,
    x_target, y_target,
    preset="t1_office31_sklearn",
    report_dir="results/t1_office31",
    seeds=[0, 42, 142],  # optional multi-seed
)
print(result.summary())
```

Regenerate the reference markdown table (no data in git):

```bash
python scripts/generate_reference_benchmark.py --office31-root /path/to/office31
```

---

## Developer path (no falsification table)

```python
from pmh import evaluate_baseline_vs_pmh

report = evaluate_baseline_vs_pmh(
    x_source, y_source, x_target, y_target,
    rank=32,
    compare_to=("coral",),
)
print(report.summary())
# Advanced: compare_arms_sklearn(..., preset="t1_office31_sklearn")
```

---

## Falsification arms (what to expect)

| Arm | Meaning | Pass criterion (paper-style) |
|-----|---------|------------------------------|
| `b0` | ERM on source features | Baseline |
| `matched` | D1 projection + train on matched subspace | Should not lose to `wrong_w` on **both** acc and geometry |
| `wrong_w` | Random ⊥ matched W | Should **not** beat `matched` alone |
| `isotropic` | D4 Gram (unmatched) | Different regularizer — not a free win |
| `coral` | CORAL-aligned features | Often **beats** matched on this benchmark |

Reference numbers: [benchmarks/office31_amazon_to_dslr.md](../benchmarks/office31_amazon_to_dslr.md).

---

## Inspect preset in Python

```python
from pmh.benchmark.presets import get_preset

p = get_preset("t1_office31_sklearn")
print(p.lemma, p.default_rank, p.sklearn_benchmark)
print(p.notes)
```

---

## Related

| Doc | Purpose |
|-----|---------|
| [Walkthrough 3](../walkthroughs/03-office31-sklearn-d1.md) | Full step-by-step |
| [Walkthrough 19](../walkthroughs/19-office31-real-data.md) | Download real Office-31 + benchmark |
| [examples/21_benchmark_sklearn_table.py](https://github.com/vishalstark512/matching-pmh/blob/main/examples/21_benchmark_sklearn_table.py) | CLI table |
| [CORRECT_USAGE.md](../CORRECT_USAGE.md) | Protocol pitfalls |
| `t1_synthetic_sklearn` | Quick demo without Office-31 download |

**Paper script (not imported by pip):** `Paper2/T1/classical_pmh/office31_pmh.py`
