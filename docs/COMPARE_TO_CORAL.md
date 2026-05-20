# PMH vs CORAL and domain adaptation

If you already use **CORAL**, **DANN**, or feature alignment, here is how **matching-pmh** fits.

---

## Same goal, different object

| | CORAL / moment matching | matching-pmh |
|---|-------------------------|--------------|
| **Aligns** | Source **features** to target (often before a linear classifier) | Encoder **sensitivity** along deployment nuisance directions |
| **Object** | Covariance of **features** (marginal) | **Σ_task** = covariance of **label-preserving** nuisance |
| **Training** | Often pre-processing + ERM | **Phase A** estimate Σ̂, **Phase B** matched penalty on `h` |
| **Claim** | “We matched second moments” | “We penalize **J_φ** along estimated Σ_task” + **controls** |

CORAL is close to a **D4-style estimator** of geometry; matched PMH is the **training repair** with wrong-W / isotropic falsification.

---

## When to use which

| Situation | Suggestion |
|-----------|------------|
| Frozen features + linear classifier only | `PMHMatcher` or CORAL baseline in `compare_arms_sklearn` |
| End-to-end fine-tuning | `PMHTrainer` + `PMHLoss` |
| You already have CORAL features | Keep CORAL as **b0 arm**; add **matched** on the same `h` |
| Need credible paper-style controls | `compare_arms` (matched vs wrong-W vs isotropic) |

```python
from pmh.baselines.coral import coral_align  # optional baseline
from pmh import compare_arms_sklearn

x_src_c, _ = coral_align(x_src, x_tgt)
# ... train on x_src_c as CORAL arm in your table
```

---

## Migration checklist (CORAL → PMH)

1. Keep your **nuisance sentence** (domain shift is D4).  
2. Pick hook `h` (not only pre-pooled features unless Phase A uses the same encoder).  
3. `PMHTrainer(..., nuisance="domain_shift").fit(..., source_batches=..., target_batches=...)`.  
4. Run `compare_arms_sklearn` or `compare_arms` with **CORAL as extra baseline** if desired.  
5. Report **matched** vs **wrong_w** vs **isotropic**, not only vs B0.

→ [Integrate your project](GETTING_STARTED.md)
