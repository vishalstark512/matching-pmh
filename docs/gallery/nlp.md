# Gallery: LLM style / format (D7)

Semantics fixed; deployment varies by **style** (tone, format, language register).

```python
from pmh import SigmaTaskConfig, estimate_from_config, PMHConfig, PMHLoss
# HF helpers (optional): pip install "matching-pmh[hf]"
from pmh.integrations.huggingface import load_style_pairs_jsonl, estimate_style_sigma

pairs = load_style_pairs_jsonl("YOUR/style_pairs.jsonl")
artifact = estimate_style_sigma(
    pairs,
    model_id="YOUR/HF_MODEL",  # or precomputed deltas
    config=SigmaTaskConfig.for_alignment(rank=32),
)
artifact.save("artifacts/style_sigma")

# Training: hook = last hidden state; see pmh.hooks.encoder_hf_hidden_states
pmh = PMHLoss(artifact, PMHConfig.finetune_llm())
# total, _ = pmh.capped_total(task_loss, h)
```

Walkthroughs: [LLM style D7](../walkthroughs/06-llm-style-d7.md) · [HF Trainer](../walkthroughs/07-hf-trainer-d7-dpo.md)

For domain shift on sentence **embeddings** (not style JSONL), use `PMHMatcher(nuisance="domain_shift")` on frozen encoder outputs instead.
