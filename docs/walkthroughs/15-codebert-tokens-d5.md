# Walkthrough 15: Code / token embeddings + D5

**Paper block:** T5B (BigCloneBench / CodeBERT) — **identifier names** are nuisance; keywords/operators carry clone signal.

**Goal:** D5 block on identifier dimensions; clone detection with PMH; falsify with signal-partition PMH.

**Script:** `examples/17_code_tokens_d5.py`

---

## Partition

```python
# After projecting tokens to h in R^d:
nuisance_idx = identifier_dims   # e.g. first k dims tied to renameable tokens
signal_idx   = keyword_dims      # do NOT put in D5 for matched arm
```

Build `h` the same way CodeBERT mean-pools subword tokens to a fixed-size vector (or use `[CLS]`).

---

## Phase A — estimate on rename-augmented code

Collect $h$ from code pairs that differ only in identifier renaming; covariance on `nuisance_idx`.

---

## Phase B — train clone classifier

```python
task = cross_entropy(head(h), clone_label)
total, _ = pmh.capped_total(task, h)
```

---

## Required falsification (T5B)

| Arm | Partition | Expected |
|-----|-----------|----------|
| Matched | identifiers | Best rename robustness |
| **E1S wrong** | keywords (signal) | **Below baseline** |

Re-run example with `nuisance_idx` on signal dims to see the negative.

---

## Run

```bash
python examples/17_code_tokens_d5.py
```

Production: use your tokenizer + CodeBERT `forward` hidden state; export $h$ to `.npy` for CLI jobs if needed.
