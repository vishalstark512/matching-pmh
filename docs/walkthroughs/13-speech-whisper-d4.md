# Walkthrough 13: Speech encoder + D4 — full guide

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

## Your nuisance sentence

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
