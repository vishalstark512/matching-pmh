# Gallery: NLP — same answer, different format

**You have:** pairs of text with the **same factual content** but different style (bullets vs prose, formal vs casual, JSON wrapper vs plain text).

**You do:** estimate a style geometry from hidden-state differences, add `PMHLoss` on your LM hook during fine-tuning.

```python
from pmh import SigmaTaskConfig, PMHConfig, PMHLoss

# pip install "matching-pmh[hf]"
from pmh.integrations.huggingface import load_style_pairs_jsonl, estimate_style_sigma

pairs = load_style_pairs_jsonl("YOUR/style_pairs.jsonl")
artifact = estimate_style_sigma(
    pairs,
    model_id="YOUR/HF_MODEL",
    config=SigmaTaskConfig.for_alignment(rank=32),
)
artifact.save("artifacts/style_sigma")

pmh = PMHLoss(artifact, PMHConfig.finetune_llm())
# In training step: total, _ = pmh.capped_total(task_loss, h)
# h = fixed pooling of last hidden state — same rule in estimate and train
```

**Try first:** `examples/data/style_pairs_sample.jsonl` + `examples/08_hf_style_d7.py`

**Walkthroughs:** [LLM style](../walkthroughs/06-llm-style-d7.md) · [HF Trainer + DPO](../walkthroughs/07-hf-trainer-d7-dpo.md)

**Not this path:** sentence embeddings from two **corpora** (no style pairs) → use `PMHMatcher(nuisance="domain_shift")` on frozen encoder outputs instead.
