# Design philosophy

**matching-pmh** is built the way we would ship an internal lab toolkit: small surface area, explicit phases, falsifiable defaults.

---

## Principles

### 1. Architecture-agnostic

The library never owns your model class. It needs one differentiable representation $h=\phi_\theta(x)$ and a story for label-preserving deployment variation. ResNet, ViT, GNN, Whisper encoder, or causal LM hidden states are all the same interface.

### 2. Two phases, frozen contract

| Phase | Responsibility |
|-------|----------------|
| **Estimate** | Turn data + nuisance assumption into $\hat\Sigma_{\mathrm{task}}$ (`SigmaTaskEstimate`) |
| **Train** | Add `PMHLoss` on $h$; leave task loss untouched |

Mixing estimation into every training step obscures identification and breaks reproducibility. We keep them separate.

### 3. One hook tensor

Pick a layer. Use it in Phase A and Phase B. Document it in your config. Re-estimate if you move the hook.

### 4. Estimators are pluggable (D1–D7)

CORAL-like methods are not competing training objectives—they are different ways to estimate the **same** $\Sigma_{\mathrm{task}}$ under different structural assumptions $A_k$. The training term is always matched PMH with $\Sigma' \approx \hat\Sigma$.

### 5. Falsification is not optional

Matched PMH without **wrong-W** and **isotropic** controls does not support a claim about the matching principle—only about regularization. The API exposes `mode=` for this reason.

### 6. Stable hyperparameters

`cap_ratio` caps the PMH term relative to task loss so $\lambda$ does not dominate tuning. `warmup_epochs` lets the task loss find a basin before geometry regularization ramps up.

### 7. Artifacts are first-class

`artifact.save()` / `load()` makes HPC jobs, team handoffs, and multi-stage pipelines (estimate on cluster → train on GPU farm) straightforward.

### 8. Examples are templates, not dependencies

Every script under `examples/` is copy-paste fodder. None are imported by the core package. CI smoke-tests them so they stay runnable.

---

## What we deliberately do not do

- Replace Hugging Face, Lightning, or sklearn— we integrate via thin wrappers.
- Auto-detect nuisance type from data (you name the deployment story).
- Hide failure modes: `preflight` and eigengap report weak identification.
- Promise SOTA on every benchmark without controls.

---

## For maintainers and contributors

- **API changes** require a CHANGELOG entry and a walkthrough update if user-facing.
- **New estimator** → implement Dk, catalog entry, nuisance_types.md, one walkthrough + example.
- **Tests** cover math invariants; **examples** cover integrator experience.

See [CONTRIBUTING.md](../CONTRIBUTING.md).
