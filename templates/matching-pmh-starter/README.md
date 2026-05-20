# matching-pmh starter template

Copy this folder into your project. Three minimal scripts — pick one golden path.

```bash
pip install matching-pmh torch
pip install "matching-pmh[sklearn]"   # for sklearn_minimal.py
pip install "matching-pmh[hf]"        # for hf_minimal.py
```

| Script | Golden path |
|--------|-------------|
| `pytorch_minimal.py` | **G1** — PyTorch, source + target loaders |
| `lightning_g1b_minimal.py` | **G1b** — Lightning `training_step` |
| `sklearn_minimal.py` | **G2** — frozen embeddings + sklearn |
| `hf_minimal.py` | **G3** — `robust_fit_text_domains` |
| `hf_trainer_g3b_minimal.py` | **G3b** — `transformers.Trainer` + `get_pmh_trainer()` |

```bash
pmh-train doctor
pmh-train evaluate --demo          # G2 Step 5 (sklearn)
python pytorch_minimal.py          # G1 + Step 5
```

Parameters: see [task docs](https://github.com/vishalstark512/matching-pmh/blob/main/docs/tasks/index.md) and `pmh-train route --list`.  
Docs: [13 tasks](https://github.com/vishalstark512/matching-pmh/blob/main/docs/tasks/index.md) · [Quickstart](https://github.com/vishalstark512/matching-pmh/blob/main/docs/QUICKSTART.md)
