# Troubleshooting

Common issues when adapting PMH to **your** pipeline. Symptom → cause → fix.

---

## Estimation (Phase A)

### `ValueError: domain_shift (D4) requires target_batches`

**Cause:** D4 needs features from two deployments.  
**Fix:** Pass `target_batches=` to `PMHTrainer.estimate()` or `PMHTrainer.fit()`, or use `PMHMatcher.fit(x_source, x_target)`.

### `ValueError: Artifact dim X != hook dim Y`

**Cause:** Phase A and Phase B use different layers or shapes.  
**Fix:** Use the **same** `hook` / `encoder` for estimate and train. Re-run Phase A after moving the hook.

### `preflight=fail` or weak eigengap

**Cause:** Nuisance subspace poorly identified (too little data, wrong Dk, or rank too high).  
**Fix:**

1. More batches in Phase A (`max_batches=100+`)
2. Lower `rank`
3. Try D1 if you have **labels on both domains**
4. See [nuisance_types.md](nuisance_types.md)

### `encoder must return [B, d], got shape ...`

**Cause:** Hook returns 4D feature maps or 3D tokens without pooling.  
**Fix:** Use `pool_spatial=True` (default), pick a pooled layer, or see [hooks.md](hooks.md).

### D7 / HF `ImportError: transformers`

**Fix:** `pip install "matching-pmh[hf]"`

---

## Training (Phase B)

### PMH loss is always zero

**Cause:** `warmup_epochs` not finished, or `weight=0`.  
**Fix:** Check `pmh.set_epoch(epoch)` or use `PMHConfig(warmup_epochs=0)` for debugging.

### PMH loss dominates task loss

**Fix:** `PMHConfig.conservative()` or lower `weight` / `cap_ratio`.

### `rank must be in 1..d, got 32` (wrong-W)

**Cause:** Wrong-W rank larger than representation dim.  
**Fix:** Fixed in v1.1+ (auto-capped). Upgrade, or set smaller `wrong_rank` on `PMHLoss`.

### Training slower than before

**Expected:** PMH adds a penalty with probes. Reduce `n_probes` in `PMHConfig` for speed.

---

## sklearn / PMHMatcher

### `fit(X_source, y_source)` confused with D4

**Cause:** Second positional arg is `y`, not `X_target`.  
**Fix:** `matcher.fit(x_source, x_target=xt)` or `fit(xs, None, xt)`.

### Pipeline `fit` fails on PMHMatcher

**Cause:** `PMHMatcher.fit` needs domain pair, not single `(X, y)`.  
**Fix:** `matcher.fit(...)` **before** pipeline; use `matcher.transform(X)` inside a custom step, or train clf on transformed features manually (see [Gallery: tabular](gallery/tabular.md)).

---

## Credible evaluation

### Matched ≈ isotropic ≈ wrong-W

**Cause:** Σ̂ weak, hook too deep/shallow, or metric insensitive.  
**Fix:** Check preflight; try different hook; use deployment-relevant val set.

### Matched worse than B0

**Not always a bug** — PMH trades off in-distribution fit for deployment geometry. Report all arms; tune `cap_ratio` / `weight`.

---

## Still stuck?

1. Run minimal example: `python examples/01_domain_shift_d4.py`  
2. Compare your code to closest [walkthrough](walkthroughs/index.md)  
3. Open a GitHub issue with: nuisance sentence, hook shape, `preflight`, and error traceback  

---

## Quick reference

| Check | Command / field |
|-------|-----------------|
| Version | `import pmh; print(pmh.__version__)` |
| Method | `artifact.method` → D1–D7 |
| Preflight | `artifact.preflight` |
| Hook dim | `h.shape[-1]` vs `artifact.sigma.shape[0]` |
