# Hugging Face Trainer

**Golden path G3 (two corpora):** [GOLDEN_PATHS.md](GOLDEN_PATHS.md) · `robust_fit_text_domains` · [Colab notebook](https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/hf_two_corpora_first_run.ipynb)

Install: `pip install "matching-pmh[hf]"`

```python
from transformers import TrainingArguments
from pmh.integrations.hf_trainer import get_pmh_trainer
from pmh import estimate_from_config, SigmaTaskConfig, PMHConfig

PMHTrainer = get_pmh_trainer()
artifact = estimate_from_config(SigmaTaskConfig.for_alignment(rank=128), deltas)

trainer = PMHTrainer.from_artifact(
    artifact,
    PMHConfig(weight=0.3, cap_ratio=0.3),
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

`PMHTrainer` subclasses `transformers.Trainer` and overrides `compute_loss` to add capped PMH on representations (default: last hidden state).

Custom representation:

```python
def my_rep(model, inputs):
    return model.get_input_embeddings()(inputs["input_ids"])

trainer = PMHTrainer(..., representation_fn=my_rep)
```

Example: `examples/10_hf_trainer.py`
