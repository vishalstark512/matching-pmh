# 60-second pipeline script (demo / video)

Use with `examples/00_first_run_domain_shift.py` (PyTorch) and `examples/02_g2_office31_style_demo.py` (sklearn).

---

## Hook (10s)

> Train on site A, deploy on site B, same labels.  
> Estimate deployment geometry once, train with capped PMH, then prove it on a **deploy holdout** with falsification arms.

---

## PyTorch — `00_first_run` or CLI (25s)

```bash
pip install matching-pmh torch
python examples/00_first_run_domain_shift.py
pmh-train evaluate --demo --stack pytorch --epochs 5
```

**Say while it runs:**

1. Synthetic Hospital A / B loaders — same labels, different shift.  
2. Baseline ERM vs PMH on B holdout.  
3. `preflight` on the geometry estimate.  
4. Step 5: `evaluate_robust_fit(..., include_falsification=True)`.

---

## sklearn — `02_g2` (25s)

```bash
pip install "matching-pmh[sklearn]"
pmh-train evaluate --demo
python examples/02_g2_office31_style_demo.py
```

**Say:** Office-31-style synthetic embeddings → one `report.summary()` with matched / wrong-W / isotropic.

---

## Close (10s)

```bash
pmh-train doctor
pmh-train recipe
```

Parameters: [PARAMETERS_CHEATSHEET.md](PARAMETERS_CHEATSHEET.md)
