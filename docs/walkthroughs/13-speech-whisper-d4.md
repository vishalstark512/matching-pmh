# Walkthrough 13: Speech encoder + D4 (Whisper template)

**Paper block:** T6A (Whisper ASR) — accent / channel variation with fixed transcript semantics.

**Goal:** Estimate domain Gram on **encoder embeddings** from source vs target acoustic conditions; train with PMH on the same hidden states.

**Script:** `examples/15_speech_encoder_d4.py`

---

## Hook point

```python
# mel: [B, 1, n_mels, T] or Whisper feature extractor output
h = encoder(mel)  # [B, d] pooled hidden state
```

For Whisper: use encoder output after conv stem + Transformer block (mean pool or last frame)—**not** logits.

---

## Nuisance story

| Valid D4 | Prefer D6 / custom |
|----------|-------------------|
| Studio vs accented speech (same text) | Adjacent-frame deltas in full attention (contain phoneme signal) |

Paper T6A uses a **content-residual** estimator for geometry; this walkthrough uses **D4** as the simplest integration path when you have clear source/target corpora.

---

## Run

```bash
python examples/15_speech_encoder_d4.py
```

---

## Production checklist

1. Source corpus (studio) vs target (deployment mic / accent).
2. Freeze or warm Whisper encoder for Phase A.
3. `estimate_from_config(SigmaTaskConfig.for_domain(rank=…), h_studio, h_deploy)`.
4. Fine-tune with PMH on `h`; report WER **and** geometry metrics (TDI / drift).
5. Note: WER alone may not separate matched from wrong-W (paper §6A dissociation).

---

## Adapt

Replace `MelEncoder` with `whisper.load_model(...).encoder` forward pass on your mel batches.
