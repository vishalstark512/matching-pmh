# Data layout for estimate (folders and `.npy`)

## Two folders (recommended)

```text
data/site_a/
  features.npy    # [N, d] float32
  labels.npy      # optional; required for D1

data/site_b/
  features.npy
  labels.npy
```

```bash
pmh-train estimate --source-dir data/site_a --target-dir data/site_b \
  --method D4 --rank 32 -o artifacts/site_b_shift
```

Accepted feature filenames: `features.npy`, `embeddings.npy`, `x.npy`, `representations.npy`, or the **only** `.npy` in the folder.

## Explicit `.npy` paths

```bash
pmh-train estimate --source-npy feats/train_hospital_a.npy \
  --target-npy feats/deploy_hospital_b.npy \
  --method D4 --rank 32 -o artifacts/d4
```

## JSON job (HPC / CI)

```bash
pmh-train estimate --config examples/configs/d4_estimate.json
```

Edit paths in `data.source_npy` / `data.target_npy`.

## Python

```python
from pmh.data_adapters import load_domain_dirs, batch_iterators

xs, ys, xt, yt = load_domain_dirs("data/site_a", "data/site_b")
src_it, tgt_it = batch_iterators(xs, xt, batch_size=32)
```

---

Next: [Golden paths](GOLDEN_PATHS.md) · [Integrate your project](GETTING_STARTED.md)
