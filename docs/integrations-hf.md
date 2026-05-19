# Hugging Face (Lemma D7)

Install: `pip install "matching-pmh[hf]"`

## Load JSONL

```python
from pmh.integrations.huggingface import load_style_pairs_jsonl

pairs = load_style_pairs_jsonl("data/style_pairs.jsonl", max_pairs=500)
```

Schema per line:

```json
{"id": "ex1", "prompt": "...", "content_fixed": "...", "style_variants": {"bulleted": "..."}}
```

## Estimate style Sigma

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pmh.integrations.huggingface import estimate_style_sigma

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
artifact = estimate_style_sigma(pairs, model, tokenizer, rank=128)
artifact.save("checkpoints/style_sigma")
```

## Lower-level API

```python
from pmh.integrations.huggingface import encode_style_deltas, make_encoder

deltas = encode_style_deltas(pairs, model, tokenizer)
encoder = make_encoder(model, tokenizer)
```

Example: `examples/08_hf_style_d7.py`
