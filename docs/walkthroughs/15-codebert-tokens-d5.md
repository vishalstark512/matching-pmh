# Walkthrough 15: Code / token blocks (D5) — full guide


!!! tip "Adopt PMH first"
    **Start:** [ADOPT.md](../../ADOPT.md) → [Golden path G1–G4](../GOLDEN_PATHS.md#g4) · **Route:** `pmh-train route --task compositional_coordinates` · **Step 5:** paper token D5
    This walkthrough is **evidence / depth** — not your first page.

> **API note:** `nuisance=` is the **deployment shift type** (D1–D7 API key), not “bad data.” [What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)


**At a glance**

| | |
|---|---|
| **Estimator** | D5 on token-type blocks |
| **Script** | `examples/17_code_tokens_d5.py` |

[Walkthrough 5](05-compositional-d5.md)

---

## Who this is for

Code models where **token groups** (comments, imports, identifiers) partition shift-related vs task token groups.

---

## Your deployment shift sentence

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
