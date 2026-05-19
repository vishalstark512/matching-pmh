# Walkthrough 15: Code / token blocks (D5) — full guide

**At a glance**

| | |
|---|---|
| **Estimator** | D5 on token-type blocks |
| **Script** | `examples/17_code_tokens_d5.py` |

[Walkthrough 5](05-compositional-d5.md)

---

## Who this is for

Code models where **token groups** (comments, imports, identifiers) partition nuisance vs task.

---

## Your nuisance sentence

*“Style tokens / imports drift; bug class label unchanged.”*

---

## Step-by-step

```bash
python examples/17_code_tokens_d5.py
```

Map token-type indices → `nuisance_indices` in **your** tokenizer layout.

---

## Adaptation worksheet

| Example | Your CodeBERT / LM |
|---------|-------------------|
| Block indices | Your AST feature groups |

---

## Next steps

- [6 — LLM style D7](06-llm-style-d7.md) for format shift
