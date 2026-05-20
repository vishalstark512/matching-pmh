# Troubleshooting

Common issues when adapting PMH to **your** pipeline. Symptom → cause → fix.

**New to PMH?** Read [What is deployment shift?](WHAT_IS_DEPLOYMENT_SHIFT.md), then the [glossary](#plain-language-glossary) below.

---

## Plain-language glossary

| You see | Plain English | What to do |
|---------|---------------|------------|
| `preflight=pass` | Geometry estimate looks identifiable | Proceed; still run controls before big claims |
| `preflight=marginal` | Weak signal — shift may be too small or too little data | More source/target batches in estimate; try lower `rank`; see [NUISANCE_SUBTYPES](NUISANCE_SUBTYPES.md) |
| `preflight=fail` | Estimate not usable as-is | Do not trust matched PMH yet; fix data/hook/rank |
| **Deployment shift** | How deploy can change inputs **without** changing the label | [WHAT_IS_DEPLOYMENT_SHIFT](WHAT_IS_DEPLOYMENT_SHIFT.md) |
| `nuisance=` (API) | **Shift type** string in code — not “bad data” | `pmh-train shifts` · `explain_nuisance_key("domain_shift")` |
| `eigengap` | How separated shift directions are from the rest | Low → treat like marginal; collect more data |
| `nuisance="domain_shift"` | Site A vs B **look**, same labels (default) | `source_batches` + `target_batches` |
| `nuisance="subspace"` | **Labels on both** sites, class geometry moves | `fit(x_src, y_src, x_tgt, y_tgt)` or labeled loaders |
| Phase A / estimate | One-time step: learn what differs between deploy and train | `trainer.fit(...)` runs this automatically |
| Phase B / PMH loss | Extra training penalty on representation `h` | Same `hook` layer as Phase A |
| `hook` | Layer where features `h` live (`[batch, d]`) | Pick one backbone layer; keep it fixed |
| `artifact` / `.pt` file | Saved geometry from Phase A | Reuse path in `artifact_path=` |
| `PMHTrainer` | Estimate + train in one object (PyTorch) | Default path for deep models |
| `PMHMatcher` | Adapt frozen NumPy/sklearn features | `fit(x_source, x_target)` then `transform` |
| `compare_arms` / `compare_arms_sklearn` | Falsification table (matched vs wrong controls) | **After** basic integration works |
| `wrong_w` arm | Random directions ⊥ matched W (sanity check) | Should not beat matched on **both** accuracy and geometry |
| `isotropic` (sklearn benchmark) | Unmatched domain directions (control), not “D2 noise” | See [CORRECT_USAGE](CORRECT_USAGE.md) |
| `trace_iso` (training) | Training-time control arm name | Not the same as sklearn `isotropic` |
| D1–D7 | Evidence estimator IDs | Ignore until [estimators](estimators/index.md); start with `domain_shift` |
| `t1_office31_sklearn` preset | Paper benchmark protocol for Office-31 | Researchers only; use [INTEGRATE](INTEGRATE.md) first |
| PMH loss = 0 | Warmup not finished or weight is zero | `PMHConfig(warmup_epochs=0)` to debug; check epoch callback |
| Matched worse than baseline | Can happen — tradeoff | Report both; tune `weight` / `cap_ratio`; not always a bug |

---

## If you see this error (copy-paste)

| Error snippet | Likely cause | Fix |
|---------------|--------------|-----|
| `ModuleNotFoundError: No module named 'sklearn'` | sklearn extra not installed | `pip install "matching-pmh[sklearn]"` |
| `ModuleNotFoundError: No module named 'torch'` | Core dep missing | `pip install matching-pmh torch` |
| `domain_shift (D4) requires target_batches` | PyTorch estimate without target data | `trainer.fit(..., target_batches=tgt_loader)` |
| `Artifact dim X != hook dim Y` | Different layer in Phase A vs B | Same `hook` for estimate and train |
| `encoder must return [B, d], got shape` | 4D feature map without pooling | `pool_spatial=True` or pooled hook ([hooks.md](hooks.md)) |
| `rank must be in 1..d` | Rank larger than representation dim | Lower `rank` / upgrade package (auto-cap in recent versions) |
| `PMHMatcher.fit` / wrong arity | Passed `y` where `x_target` expected | `matcher.fit(x_src, x_target=x_tgt)` |
| `ImportError: transformers` | D7 / HF path without extra | `pip install "matching-pmh[hf]"` |
| `KeyError: unknown preset` | Typo in preset name | `pmh-train list-presets` |
| `preflight=fail` after estimate | Too little data or wrong story | More batches, lower rank, check [glossary](#plain-language-glossary) |

---

## Estimation (Phase A)

### `ValueError: domain_shift (D4) requires target_batches`

**Cause:** D4 needs features from two deployments.  
**Fix:** Pass `target_batches=` to `PMHTrainer.estimate()` or `PMHTrainer.fit()`, or use `PMHMatcher.fit(x_source, x_target)`.

### `ValueError: Artifact dim X != hook dim Y`

**Cause:** Phase A and Phase B use different layers or shapes.  
**Fix:** Use the **same** `hook` / `encoder` for estimate and train. Re-run Phase A after moving the hook.

### `preflight=fail` or weak eigengap

**Plain English:** The library could not find a clear “deployment shift” direction in your features.

**Cause:** Too little data, hook too shallow/deep, wrong estimator, or rank too high.  
**Fix:**

1. More batches in Phase A (`max_batches=100+`)
2. Lower `rank`
3. If you have **labels on both sites**, try `nuisance="subspace"` (labeled cross-domain)
4. See [glossary](#plain-language-glossary) and [NUISANCE_SUBTYPES.md](NUISANCE_SUBTYPES.md)

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
