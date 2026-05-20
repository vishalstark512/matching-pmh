# Walkthrough 13: Speech encoder + D4 — full guide


!!! tip "Adopt PMH first"
    **Start:** [ADOPT.md](../../ADOPT.md) → [Golden path G1–G4](../GOLDEN_PATHS.md#g1) · **Route:** `pmh-train route --task vision_classification` · **Step 5:** compare_arms on mic-shift holdout
    This walkthrough is **evidence / depth** — not your first page.

> **API note:** `nuisance=` is the **deployment shift type** (D1–D7 API key), not “bad data.” [What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)


**At a glance**

| | |
|---|---|
| **Estimator** | D4 on encoder hidden states |
| **Script** | `examples/15_speech_encoder_d4.py` |
| **Metrics** | WER + geometry (TDI when implemented on your hook) |

[Walkthrough 11](11-temporal-d6.md) for temporal drift within utterances

---

## Who this is for

Speech models (Whisper-style encoders) with **accent / channel / mic** shift; transcript label unchanged.

---

## Your deployment shift sentence

*“New microphone or accent distribution; word labels still correct.”*

---

## Step-by-step

1. Hook: last encoder hidden state or your pooling rule `[B, d]`.
2. Source vs target audio loaders (different corpora).
3. `PMHTrainer` + `compare_arms` with **WER** on target.

```bash
python examples/15_speech_encoder_d4.py
```

---

## Adaptation worksheet

| Example | Your project |
|---------|--------------|
| Mel + toy encoder | Whisper / Wav2Vec2 |
| Val metric | WER / CER |

---

## Next steps

- [11 — D6 temporal](11-temporal-d6.md)
- [8 — Controls](08-falsification-controls.md)
