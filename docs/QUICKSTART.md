# Quickstart (10 minutes)

Get from `pip install` to a trained model with matched PMH—without replacing your stack.

---

## 1. Install

```bash
pip install matching-pmh torch
```

Optional: `pip install "matching-pmh[vision]"` for ResNet/ViT examples.

---

## 2. Run the canonical example

```bash
git clone https://github.com/vishalstark512/matching-pmh.git
cd matching-pmh
python examples/01_domain_shift_d4.py
```

You should see `preflight=...` then epoch logs with `task` and `pmh` losses.

---

## 3. Understand the two phases

| Phase | Code | When |
|-------|------|------|
| **A. Estimate** | `estimate_from_config(...)` | Once per nuisance story (or when data distribution shifts) |
| **B. Train** | `pmh.capped_total(task_loss, h)` | Every training step |

The only contract: **`h` must be the same tensor** (same layer, same dim) in both phases.

---

## 4. Copy into your project (minimal)

```python
from pmh import SigmaTaskConfig, PMHConfig, PMHLoss, collect_features, estimate_from_config

# --- your existing model ---
def encode(x):
    return model.backbone(x)  # [B, d]

# Phase A
model.eval()
h_src = collect_features(encode, source_loader, max_batches=50)
h_tgt = collect_features(encode, target_loader, max_batches=50)
artifact = estimate_from_config(SigmaTaskConfig.for_domain(rank=32), h_src, h_tgt)
artifact.save("checkpoints/sigma_task")

# Phase B
pmh = PMHLoss(artifact, PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=2))
model.train()
for epoch in range(epochs):
    pmh.set_epoch(epoch)
    for x, y in train_loader:
        opt.zero_grad()
        h = encode(x)
        task = loss_fn(head(h), y)
        (total, _) = pmh.capped_total(task, h)
        total.backward()
        opt.step()
```

---

## 5. Pick your walkthrough

| If you use… | Open |
|-------------|------|
| Any PyTorch model | [Walkthrough 1](walkthroughs/01-pytorch-domain-d4.md) |
| ResNet / ViT | [2](walkthroughs/02-resnet-vision-d4.md) or [12](walkthroughs/12-vit-cls-d4.md) |
| Hugging Face LM | [6](walkthroughs/06-llm-style-d7.md), [7](walkthroughs/07-hf-trainer-d7-dpo.md) |
| Molecules / graphs | [14](walkthroughs/14-qm9-molecule-d5.md) |
| Code / tokens | [15](walkthroughs/15-codebert-tokens-d5.md) |
| Speech | [13](walkthroughs/13-speech-whisper-d4.md) |

Full list: [walkthroughs/index.md](walkthroughs/index.md).

---

## 6. Credible evaluation (required)

Train three arms and compare **deployment** metrics:

1. **Matched** — default `PMHLoss`
2. **Wrong-W** — `mode="wrong_w"`
3. **Isotropic** — `mode="isotropic"`

→ [Walkthrough 8 — Falsification](walkthroughs/08-falsification-controls.md)

---

## Next

- [THEORY.md](THEORY.md) — mathematics and scope  
- [ARCHITECTURES.md](ARCHITECTURES.md) — integration patterns  
- [PHILOSOPHY.md](PHILOSOPHY.md) — why the API looks like this
