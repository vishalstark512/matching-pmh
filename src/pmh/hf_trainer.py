"""Hugging Face–aware PMHTrainer (D7 + hidden-state hooks)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from pmh.hooks import encoder_hf_hidden_states
from pmh.integrations.huggingface import make_encoder
from pmh.trainer import PMHTrainer


class HFPMHTrainer(PMHTrainer):
    """PMHTrainer for causal / encoder HF models.

  - Training batches: ``(input_ids,)`` or ``(input_ids, attention_mask, labels)``
  - Phase A D7: ``estimate(style_jsonl=..., hf_model=..., hf_tokenizer=...)``
  - Phase A D4 on text: use ``make_encoder`` as hook via ``text_encoder`` mode

    Parameters
    ----------
    hf_model, hf_tokenizer : Transformers model and tokenizer
    pool : ``last`` | ``mean`` | ``cls`` for hidden-state pooling
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        pool: str = "last",
        layer: int = -1,
        nuisance: str = "domain_shift",
        **kwargs: Any,
    ) -> None:
        enc = encoder_hf_hidden_states(model, layer=layer, pool=pool)  # type: ignore[arg-type]
        super().__init__(
            model,
            hook=enc,
            nuisance=nuisance,
            has_style_pairs=(nuisance in ("style", "alignment", "D7", "auto")),
            **kwargs,
        )
        self.tokenizer = tokenizer
        self.hf_model = model
        self._text_encoder = make_encoder(model, tokenizer, pool=pool)

    def estimate_text_domains(
        self,
        source_texts: list[str],
        target_texts: list[str],
        *,
        max_samples: int = 500,
    ) -> None:
        """D4 on sentence embeddings (no fine-tune required for estimate)."""
        src = source_texts[:max_samples]
        tgt = target_texts[:max_samples]
        h_src = self._text_encoder(src)
        h_tgt = self._text_encoder(tgt)
        from pmh.estimate import estimate_from_config
        from pmh.config import SigmaTaskConfig

        cfg = SigmaTaskConfig.for_domain(rank=self.rank or 32, shrinkage=self.shrinkage)
        self.artifact_ = estimate_from_config(cfg, h_src, h_tgt)
        self._bind_pmh_loss()

    def estimate_style(
        self,
        style_jsonl: str | Path,
        *,
        save: bool = True,
    ) -> None:
        """D7 from style-pair JSONL."""
        self.estimate(
            style_jsonl=style_jsonl,
            hf_model=self.hf_model,
            hf_tokenizer=self.tokenizer,
            save=save,
        )
