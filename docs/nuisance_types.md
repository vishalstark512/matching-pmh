# Nuisance types D1–D7

Use `pmh-train list-methods` for a terminal summary. Each method estimates a deployment nuisance covariance $\Sigma_{\mathrm{task}}$; training uses `PMHLoss` on representations $h=\phi(x)$.

## D1 — Subspace (cross-domain SVD)

**Inputs:** source/target feature matrices (and labels for strict D1, or paired domains).

```python
from pmh import SigmaTaskConfig, estimate_from_config
artifact = estimate_from_config(SigmaTaskConfig.for_subspace(rank=32), h_src, h_tgt, y_src, y_tgt)
```

**CLI:** `source_npy`, `target_npy`, optional `source_labels` / `target_labels`, `rank`.

## D2 — Isotropic

**Inputs:** representation dimension and noise level only.

```bash
pmh-train estimate --config <(echo '{"estimator":{"method":"D2","dim":64,"noise_level":0.1},"data":{"dim":64},"output":"out/d2"}')
```

## D3 — Augmentation modes

**Inputs:** stack of augmentation-induced deltas `[M, d]`.

## D4 — Domain Gram

**Inputs:** unlabeled (or pooled) source and target features.

See `examples/configs/d4_estimate.json`.

## D5 — Compositional

**Inputs:** features `[N, d]` and `nuisance_indices` listing nuisance coordinates.

## D6 — Temporal

**Inputs:** sequences `[T, N, d]` or `[N, T, d]` per `estimate_d6` convention in examples.

## D7 — Style / alignment (LLM)

**Inputs:** `style_pairs.jsonl` with `prompt`, `content_fixed`, `style_variants` (dict).

```bash
pmh-train estimate --config examples/configs/d7_style_estimate.json
python examples/11_dpo_lora_style_pmh.py --train
```

**DPO training:** `preference_pairs.jsonl` with `chosen` / `rejected` and list `style_variants`; see `examples/configs/dpo_train_job.json` and `pmh-train run`.

## Adapting to your task

1. Pick the lemma row that matches your nuisance (table above).
2. Run `pmh-train estimate` or `estimate_from_config` → save `.pt`.
3. Check `pmh-train preflight artifacts/....pt` (eigengap $\gamma_r$).
4. Attach `PMHLoss(artifact)` in your trainer (PyTorch, HF `PMHTrainer`, or Lightning).

Falsification arms: `PMHLoss(..., mode="wrong_w"|"isotropic")` — see `examples/04_falsification_controls.py`.
