# Quickstart (10 minutes)

**New users:** [What is PMH?](WHAT_IS_PMH.md) → [First hour](FIRST_HOUR.md) → this page.

---

## 1. Install

```bash
pip install matching-pmh torch
```

---

## 2. Run the first-run demo (readable metrics)

```bash
git clone https://github.com/vishalstark512/matching-pmh.git
cd matching-pmh
python examples/00_first_run_domain_shift.py
```

You should see **baseline vs PMH target accuracy**. Then:

```bash
python examples/01_domain_shift_d4.py
```

for the minimal `PMHTrainer` training loop.

---

## 3. Two phases (the only contract)

| Phase | What | When |
|-------|------|------|
| **A. Estimate** | Compute Σ̂_task from deployment data | Once per nuisance story |
| **B. Train** | Add matched PMH on hook `h` | Every step |

**Same hook `h` in both phases** — same layer, same dimension `[B, d]`.

---

## 4. Copy into your project

### PyTorch (recommended)

```python
from pmh import PMHTrainer, PMHConfig

trainer = PMHTrainer(
    model,
    hook=backbone,
    head=head,
    nuisance="domain_shift",
    pmh_config=PMHConfig.balanced(),
    artifact_path="artifacts/sigma.pt",
)
trainer.fit(
    train_loader,
    source_batches=source_loader,
    target_batches=target_loader,
    epochs=20,
)
```

### sklearn / frozen features

```python
from pmh import PMHMatcher

matcher = PMHMatcher(nuisance="domain_shift", rank=16).fit(x_source, x_target)
artifact = matcher.artifact_   # use with PMHLoss in torch, or matcher.transform(X)
```

### Manual loop (full control)

```python
from pmh import PMHConfig, PMHLoss, collect_features, estimate_from_config, SigmaTaskConfig

# Phase A
h_src = collect_features(encoder, source_loader, max_batches=50)
h_tgt = collect_features(encoder, target_loader, max_batches=50)
artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=32), h_src, h_tgt)

# Phase B
pmh = PMHLoss(artifact, PMHConfig.balanced())
for epoch in range(epochs):
    pmh.set_epoch(epoch)
    for x, y in train_loader:
        h = encoder(x)
        task = loss_fn(head(h), y)
        (total, _) = pmh.capped_total(task, h)
        total.backward()
        optimizer.step()
```

---

## 5. Pick your stack

→ [Choose your setup](CHOOSE_YOUR_SETUP.md) (decision table)  
→ [Gallery](gallery/README.md) (vision / tabular / NLP templates)

---

## 6. Credible evaluation

Train **matched**, **wrong-W**, and **isotropic** — not only “with vs without PMH.”

→ [Walkthrough 8 — Controls](walkthroughs/08-falsification-controls.md)  
→ `compare_arms` / `compare_arms_sklearn` in [Getting started](GETTING_STARTED.md)

---

## Next

| Doc | Purpose |
|-----|---------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | **Main adoption guide** |
| [ADAPT_YOUR_PIPELINE.md](ADAPT_YOUR_PIPELINE.md) | Checklist + estimator table |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Errors and fixes |
| [hooks.md](hooks.md) | ResNet, timm, HF hooks |
