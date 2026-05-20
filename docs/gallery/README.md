# Gallery — copy-paste templates

**You have two environments and the same labels?** Pick one row, copy the block, swap in your model/data.

| You have… | Do this | Page |
|-----------|---------|------|
| PyTorch images/signals, site A + site B loaders | `PMHTrainer` + `nuisance="domain_shift"` | [Vision](vision.md) |
| Precomputed embeddings (`.npy`), sklearn classifier | `PMHMatcher` in a `Pipeline` | [Tabular](tabular.md) |
| LLM answers with same facts, different formatting | Style pairs JSONL + HF hook | [NLP](nlp.md) |

**Before you start:** [Docs home](../index.md) · [D1–D7 subtypes](../NUISANCE_SUBTYPES.md) · [First hour](../FIRST_HOUR.md)

After it runs → [Controls walkthrough](../walkthroughs/08-falsification-controls.md) · [Troubleshooting glossary](../TROUBLESHOOTING.md#plain-language-glossary)
