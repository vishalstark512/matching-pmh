# T6A — Speech / ASR — mic & room shift

**Paper evidence:** [main.pdf](../../main.pdf) · [Block findings](../findings.html)

**Lemma:** D6 · **Stack:** pytorch
**Nuisance key:** `temporal`

**Production change:** Microphone, room, codec; **transcript / word label** fixed.

**Notebook (Run All, built-in demo):** [t06a-speech-whisper.ipynb](../../notebooks/tasks/t06a-speech-whisper.ipynb)

```bash
pip install matching-pmh torch
# Open the notebook and Run All
```

## What this task achieved (headline)

> pmh_content_residual: Libri **other-WER 14.63%** (−8.6 pp vs baseline 23.26%); TDI **0.381**.

| baseline WER | pmh_content_residual |
|--------------|----------------------|
| 23.26% | **14.63%** |

**Note:** Paper uses **content-residual W** (`pmh.calibrate.content_residual_subspace`). Demo uses `temporal` on sequence embeddings.

## Subtasks (paper)

<a id="t6a-run"></a>

### LibriSpeech four arms + WER

pmh_content_residual WER 14.63%.

<a id="t6a-collect-w"></a>

### Content-residual subspace

Only matched W fixes geometry.

<a id="t6a-export"></a>

### Strengthening analysis JSON



## Run with matching-pmh

```python
from pmh import PMHTrainer, evaluate_robust_fit
# nuisance="temporal"
```

## Do not use PMH when

Language or vocabulary change at deploy.

## Replace demo data with yours

Swap demo loaders for your `train_loader`, `source_batches`, `target_batches`, and deploy holdout. Hook the backbone before your task head.

[← All 13 tasks](index.md) · [Quickstart](../QUICKSTART.md)

<a id="t06a-speech-whisper"></a>
