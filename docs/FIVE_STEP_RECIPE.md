# Five-step recipe

**Train on site A. Deploy on site B. Same labels.**

**One line:** estimate deployment geometry once → train with capped PMH → compare **matched / wrong-W / isotropic** on **deploy holdout**.

Product spine from the paper (§7, Fig. 4). Everything in matching-pmh implements this pipeline—not thirteen separate task scripts.

Meta-structure: [META_STRUCTURE.md](META_STRUCTURE.md) · Code: `pmh.recipe` · CLI: `pmh-train recipe`

---

## The one object

**Deployment shift geometry** — ways inputs can change at site B vs site A **without changing the label**.

(In the paper this is written with “nuisance” \(n\) and \(\Sigma_{\mathrm{task}}\); in code you pick a **shift type** with `nuisance="domain_shift"` etc. **Plain English:** [WHAT_IS_DEPLOYMENT_SHIFT.md](WHAT_IS_DEPLOYMENT_SHIFT.md).)

You **estimate** that geometry once, then **train** with a capped penalty on representation sensitivity along it.

---

## Flow

```mermaid
flowchart LR
  S0[0 Scope] --> S1[1 Identify A_k]
  S1 --> S2[2 Estimate]
  S2 --> S3[3 Apply PMH]
  S3 --> S4[4 Cap protocol]
  S4 --> S5[5 Evidence]
```

| Step | What you do | Python |
|------|-------------|--------|
| **0 Scope** | Same labels on A and B? Deploy story label-preserving? | `check_applicability` · `pmh.scope` |
| **1 Identify** | What changes at deploy? → `nuisance=` shift type | `suggest_nuisance` · `pmh-train shifts` · [plain guide](WHAT_IS_DEPLOYMENT_SHIFT.md) |
| **2 Estimate** | \(\hat\Sigma_{\mathrm{task}}\) + eigengap preflight | `estimate_from_config` · `PMHTrainer.estimate` · `recipe.step_estimate` |
| **3 Apply** | Mode **A** (Jacobian) or **B** (projection) | below · `pmh.apply` · [INTEGRATE](INTEGRATE.md) |
| **4 Protocol** | Cap PMH vs task loss; hybrid = sum of capped terms | `PMHConfig.balanced()` · [PMH_PARAMETERS](PMH_PARAMETERS.md) |
| **5 Evidence** | matched / wrong-\(W\) / isotropic (+ signal-\(W\)); geometry ≠ accuracy | `compare_arms` · `pmh.evidence` |

---

<a id="step-3-mode-a-vs-b"></a>

## Step 3 — Mode A vs B

One \(\hat\Sigma_{\mathrm{task}}\), two application operators:

| Mode | When | API |
|------|------|-----|
| **A — Jacobian** | Train/fine-tune; hook \(h=\phi(x)\) | `robust_fit`, `PMHTrainer`, `PMHLoss` |
| **B — projection** | Frozen features / sklearn | `PMHMatcher`, `compare_arms_sklearn` |

```python
from pmh.recipe import recommended_application_mode, step_identify
shift = step_identify(has_target_domain=True, has_source_labels=True)
print(recommended_application_mode(shift, stack="pytorch"))  # or "sklearn"
```

Stack guides: [INTEGRATE.md](INTEGRATE.md) · code templates: [GOLDEN_PATHS.md](GOLDEN_PATHS.md).

---

## Quick start (bundled steps)

**PyTorch (Mode A)** — steps 0–3 in one call:

```python
from pmh import check_applicability, robust_fit

print(check_applicability(stack="pytorch", n_source=500, n_target=400).summary())
out = robust_fit(
    model, train_loader,
    source_batches=src, target_batches=tgt,
    hook="auto", epochs=20,
)
```

**sklearn (Mode B)** — frozen features:

```python
from pmh import evaluate_baseline_vs_pmh

report = evaluate_baseline_vs_pmh(x_source, y_source, x_target, y_target)  # Step 5 arms by default
print(report.summary())  # matched / wrong-W / isotropic on deploy holdout
```

**Explicit steps:**

```python
from pmh.recipe import step_scope, step_identify, format_five_step_guide

print(format_five_step_guide(stack="pytorch"))
scope = step_scope(stack="pytorch", n_source=500, n_target=400)
shift = step_identify(has_target_domain=True, has_source_labels=True)
print(scope.summary(), shift.nuisance, shift.assumption)
```

CLI:

```bash
pmh-train recipe
pmh-train recipe --stack sklearn --identify --target-domain
pmh-train recipe --task pose_or_keypoints
```

---

## After the recipe works

| Next | Page |
|------|------|
| Finder + `route --task` | [APPLICATIONS.md](APPLICATIONS.md) |
| Install, CLI, stacks | [INTEGRATE.md](INTEGRATE.md) |
| Copy one code path | [GOLDEN_PATHS.md](GOLDEN_PATHS.md) |
| Falsification (Step 5 detail) | [walkthroughs/08-falsification-controls.md](walkthroughs/08-falsification-controls.md) |
| Paper block examples only | **Evidence** tab · [walkthroughs/index.md](walkthroughs/index.md) |

**Do not skip Step 5** for production claims—matched-only improvements are inconclusive (paper §7).

---

## Symptom → assumption (Table 4)

| You notice | \(A_k\) | `nuisance` |
|------------|---------|------------|
| Site / camera / cohort look, same labels | \(A_4\) | `domain_shift` |
| Same classes, different feature geometry | \(A_1\) | `subspace` |
| Known aug modes (blur, crop, …) | \(A_3\) | `augmentation` |
| Only some coordinates of \(h\) move | \(A_5\) | `compositional` |
| Sequence drift, label constant | \(A_6\) | `temporal` |
| LLM surface / PGD deltas | \(A_7\) | `style` / alignment |
| No preferred direction (sensor noise) | \(A_2\) | `isotropic` |

Full table: [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md) · `pmh-train list-methods`
