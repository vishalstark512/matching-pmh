# Theory and the five-step recipe

This page summarizes the **matching principle** as in the Grand Unification manuscript. The library implements the **practice**; the paper proves the **claims**. You do not need the paper codebase to use `matching-pmh` on a new task.

---

## The one object

**Deployment nuisance** = ways the input can change at test time **without changing the label**.

Stack that variation into a random vector \(n\) with law \(Q_n\). The population matrix is

\[
\Sigma_{\mathrm{task}} = \mathrm{Cov}_{Q_n}(n).
\]

**Deployment drift** of an encoder \(\phi\) is how much representations move under that nuisance. In the local-linear regime,

\[
\tilde D_Q(\phi) = \mathbb{E}_x\left[\mathrm{Tr}\left(J_\phi(x)^\top J_\phi(x)\,\Sigma_{\mathrm{task}}\right)\right].
\]

**Matched PMH** adds a penalty that discourages Jacobian energy along a matrix \(\Sigma'\) whose **column space covers** \(\mathrm{range}(\Sigma_{\mathrm{task}})\):

\[
\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \lambda\,\mathbb{E}_x\left[\mathrm{Tr}\left(J_\phi^\top J_\phi\,\Sigma'\right)\right].
\]

**Architecture-agnostic:** \(\phi\) can be any differentiable map you train (CNN, ViT, GNN, Transformer, MLP, speech encoder, causal LM with LoRA). The library only needs representations \(h=\phi(x)\) (or hidden states) and a way to estimate \(\Sigma_{\mathrm{task}}\) for your nuisance story.

---

## What the library is for

| Paper | Library |
|-------|---------|
| Proves range matching, necessity, controls | Estimates \(\hat\Sigma_{\mathrm{task}}\) (D1–D7) |
| Thirteen **example** task blocks | **Your** task + **your** architecture |
| Theory of when \(A_k\) holds | `preflight` eigengap + falsification arms |

The thirteen blocks in the paper (ViT, ResNet, QM9, Whisper, Qwen DPO, …) are **evidence**, not a closed list. If you can state label-preserving deployment variation, you can apply the recipe.

---

## Five-step recipe (always the same)

1. **Name the nuisance family** \(A_k\) — what changes at deployment without changing \(y\)?
2. **Estimate** \(\hat\Sigma_{\mathrm{task}}\) with the matching Lemma **D1–D7** estimator.
3. **Preflight** — eigengap \(\gamma_r\); marginal gap means the estimator is ill-conditioned (Office-31 pattern).
4. **Train** with task loss + capped `PMHLoss` on \(h=\phi(x)\).
5. **Report controls** — matched, **wrong-W**, **isotropic**, and **signal-W** when applicable.

A gain on the task metric **without** wrong-W and signal-W does not show the principle is responsible.

---

## Seven nuisance families (structural assumptions)

| \(A_k\) | When it applies | Estimator | Library |
|---------|-----------------|-----------|---------|
| **D1** | Low-rank subspace shift; signal ⊥ nuisance subspace | Cross-domain SVD | `for_subspace` |
| **D2** | Rotation-invariant / isotropic input noise | \(\sigma^2 I\) | `for_isotropic` |
| **D3** | Finite photometric / occlusion modes | Augmentation Gram | `for_augmentation` |
| **D4** | Domain shift, \(P(y|x)\) stable | Domain feature Gram | `for_domain` |
| **D5** | Known nuisance coordinates (atoms, tokens) | Block on indices | `for_compositional` |
| **D6** | Label constant along a sequence | Temporal residual cov | `for_temporal` |
| **D7** | Style / format vs fixed semantics (LLM) | Style-pair Gram | `for_alignment` |

Hybrid nuisances → estimate separate \(\Sigma\) and **add penalties** (paper §5, additive composition).

---

## Three predictions (falsification)

| Arm | \(\Sigma'\) | Theory says |
|-----|-------------|-------------|
| **Matched** | \(\hat\Sigma_{\mathrm{task}}\) from correct \(A_k\) | Best reduction of \(\tilde D_Q\) / deployment metrics |
| **Wrong-W** | Random rank-\(r\) subspace | Like isotropic PMH (Lemma C) |
| **Isotropic** | \(\sigma^2 I\) | Uniform Jacobian shrinkage (Paper 1) |
| **Signal-W** | Task / signal directions | **Hurts** task metric (Cor. E) |

```python
PMHLoss(artifact, config)                    # matched
PMHLoss(artifact, config, mode="wrong_w")
PMHLoss(artifact, config, mode="isotropic")
```

---

## When **not** to use this framework

- **Causal / spurious correlation** where the “nuisance” is entangled with \(y\) (e.g. Colored MNIST, Waterbirds) — label-preservation fails.
- **No deployment story** — if you cannot describe what varies at test time without label change, \(\Sigma_{\mathrm{task}}\) is undefined.
- **Matched-only reporting** — without controls, improvements may be generic regularization.

---

## Mechanical core (~12 lines of PyTorch)

Once \(\hat\Sigma\) is a `(d, d)` PSD matrix and `encoder` maps `x` → `h`:

```python
from pmh import PMHLoss, PMHConfig

pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3))
task_loss = your_loss(h, y)
total, pmh_term = pmh.capped_total(task_loss, h)
(total).backward()
```

The paper uses Hutchinson / paired-view / multi-scale surrogates for the trace; `PMHLoss` implements a practical finite-difference-style penalty on representations.

---

## Further reading

- [Getting started](getting-started.md) — symptom → method
- [Nuisance cookbook](nuisance_types.md) — CLI and data formats
- Grand Unification manuscript (theory, thirteen blocks, proofs)
