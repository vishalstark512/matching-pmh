# Three golden paths (AI developers)

Ignore the full walkthrough grid until one of these works. Everything else is **Advanced**.

---

## G1 — PyTorch, two domains

**You have:** `model`, train loader, source + target loaders (target labels optional).

```python
from pmh import check_applicability, robust_fit, suggest_hook

print(check_applicability(stack="pytorch", n_source=500, n_target=400).summary())

out = robust_fit(
    model,
    train_loader,
    source_batches=source_loader,
    target_batches=target_loader,
    hook="auto",
    head=classifier,
    epochs=20,
)
print(out.preflight_message)
```

- Colab: [domain_shift_first_run.ipynb](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/domain_shift_first_run.ipynb)
- HF `Trainer`: [integrations-hf-trainer.md](integrations-hf-trainer.md) + `get_pmh_trainer()`

---

## G2 — Frozen features + sklearn

**You have:** `x_source`, `y_source`, `x_target` (embeddings).

```python
from pmh import check_applicability, evaluate_baseline_vs_pmh

print(check_applicability(stack="sklearn", n_source=len(x_source), n_target=len(x_target)).summary())

report = evaluate_baseline_vs_pmh(
    x_source=x_source, y_source=y_source,
    x_target=x_target, y_target=y_target,
    compare_to=("coral",),
)
print(report.summary())
```

- Colab: [sklearn_frozen_features_first_run.ipynb](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/sklearn_frozen_features_first_run.ipynb)
- Pipeline: [gallery/tabular.md](gallery/tabular.md)

---

## G3 — LLM / two text corpora (same task)

**You have:** texts from corpus A and B, same label semantics; fine-tuning with HF.

```python
from pmh import robust_fit_text_domains

out = robust_fit_text_domains(
    model, tokenizer, train_loader,
    source_texts=texts_a,
    target_texts=texts_b,
    epochs=3,
)
```

- Colab: [hf_two_corpora_first_run.ipynb](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/hf_two_corpora_first_run.ipynb)
- Style JSONL (different problem): [walkthrough 6](walkthroughs/06-llm-style-d7.md)

---

## Before production

1. `check_applicability(...)` — go / marginal / no-go  
2. Target holdout metric — `evaluate_baseline_vs_pmh` or your own val loop  
3. [Falsification controls](walkthroughs/08-falsification-controls.md) — optional, advanced
