# matching-pmh starter template

Copy this folder into your project. Three minimal scripts — pick one golden path.

```bash
pip install matching-pmh torch
pip install "matching-pmh[sklearn]"   # for sklearn_minimal.py
pip install "matching-pmh[hf]"        # for hf_minimal.py
```

| Script | When |
|--------|------|
| `pytorch_minimal.py` | Train PyTorch model, source + target loaders |
| `sklearn_minimal.py` | Frozen embeddings + sklearn |
| `hf_minimal.py` | Two text lists + toy HF model |

```bash
pmh-train wizard
```

Docs: https://github.com/vishalstark512/matching-pmh/blob/main/docs/GOLDEN_PATHS.md
