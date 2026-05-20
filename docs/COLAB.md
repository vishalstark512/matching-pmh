# Run in Google Colab (no downloads)

**Hospital A → Hospital B** — synthetic domain shift, baseline vs PMH target accuracy in ~5 minutes on CPU or GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/domain_shift_first_run.ipynb)

---

## What you get

- `pip install matching-pmh torch`
- Side-by-side **target accuracy** (deploy site) for standard training vs PMH
- Copy-paste snippet for your own loaders

No datasets, no API keys, no paper background.

---

## Local equivalent

```bash
pip install matching-pmh torch
python examples/00_first_run_domain_shift.py
```

---

## Frozen features + sklearn (no PyTorch train loop)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/sklearn_frozen_features_first_run.ipynb)

- `pip install "matching-pmh[sklearn]"`
- Synthetic embeddings, baseline vs `PMHMatcher` + `Pipeline`
- [Gallery: tabular](gallery/tabular.md) · [Walkthrough 3](walkthroughs/03-office31-sklearn-d1.md)

Local: `pip install "matching-pmh[sklearn]"` then `python examples/06_office31_sklearn.py`

---

## LLM format shift

| You have | Open |
|----------|------|
| Style-pair JSONL | [Gallery: NLP](gallery/nlp.md) · `examples/08_hf_style_d7.py` |

---

## Expected terminal output

[Demo output reference](DEMO_OUTPUT.md) (what success looks like before you record a GIF).
