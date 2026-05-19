"""Hugging Face helpers for Lemma D7 (style-pair Gram).

Requires optional install: ``pip install "matching-pmh[hf]"``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pmh.artifact import SigmaTaskEstimate
from pmh.config import SigmaTaskConfig
from pmh.estimate import estimate_from_config
from pmh.estimators.d7_alignment import _embed_style_deltas


def _require_transformers() -> Any:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'Hugging Face integration requires transformers. Install with: '
            'pip install "matching-pmh[hf]"'
        ) from exc
    return transformers


@dataclass
class StylePairRecord:
    """One content-fixed example with multiple style rewrites."""

    item_id: str
    prompt: str
    content_fixed: str
    style_variants: dict[str, str]
    semantic_control: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.item_id,
            "prompt": self.prompt,
            "content_fixed": self.content_fixed,
            "style_variants": self.style_variants,
        }
        if self.semantic_control is not None:
            d["semantic_control"] = self.semantic_control
        return d


@dataclass
class PreferencePairRecord:
    """DPO-style row (T7A ``preference_pairs.jsonl`` schema)."""

    item_id: str
    prompt: str
    chosen: str
    rejected: str
    style_variants: list[str]
    rejected_style_variants: list[str] = field(default_factory=list)
    semantic_control: str | None = None


def _variants_list(raw: object) -> list[str]:
    if isinstance(raw, dict):
        return [str(v) for v in raw.values()]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def load_preference_pairs_jsonl(
    path: str | Path,
    *,
    max_pairs: int | None = None,
) -> list[PreferencePairRecord]:
    """Load preference JSONL (T7A: prompt, chosen, rejected, style_variants)."""
    path = Path(path)
    records: list[PreferencePairRecord] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chosen_vars = _variants_list(row.get("style_variants"))
            if not chosen_vars:
                raise ValueError(f"{path}:{line_no}: missing style_variants")
            records.append(
                PreferencePairRecord(
                    item_id=str(row.get("id", f"row_{line_no}")),
                    prompt=str(row["prompt"]),
                    chosen=str(row["chosen"]),
                    rejected=str(row["rejected"]),
                    style_variants=chosen_vars,
                    rejected_style_variants=_variants_list(row.get("rejected_style_variants")),
                    semantic_control=(
                        str(row["semantic_control"])
                        if row.get("semantic_control") is not None
                        else None
                    ),
                )
            )
            if max_pairs is not None and len(records) >= max_pairs:
                break
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def load_style_pairs_jsonl(
    path: str | Path,
    *,
    max_pairs: int | None = None,
) -> list[StylePairRecord]:
    """Load style-pair JSONL (T7A schema: prompt, content_fixed, style_variants)."""
    path = Path(path)
    records: list[StylePairRecord] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            variants = row.get("style_variants")
            if not isinstance(variants, dict) or not variants:
                raise ValueError(f"{path}:{line_no}: missing non-empty style_variants")
            records.append(
                StylePairRecord(
                    item_id=str(row.get("id", f"row_{line_no}")),
                    prompt=str(row["prompt"]),
                    content_fixed=str(row["content_fixed"]),
                    style_variants={str(k): str(v) for k, v in variants.items()},
                    semantic_control=(
                        str(row["semantic_control"])
                        if row.get("semantic_control") is not None
                        else None
                    ),
                )
            )
            if max_pairs is not None and len(records) >= max_pairs:
                break
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def format_chat(
    prompt: str,
    response: str,
    tokenizer: Any,
    *,
    use_chat_template: bool = True,
) -> str:
    """Format user/assistant turn for encoding."""
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return f"User: {prompt}\nAssistant: {response}"


@torch.no_grad()
def encode_texts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 8,
    max_length: int = 512,
    pool: str = "last",
    device: torch.device | str | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Encode texts to ``[N, d]`` hidden-state vectors.

    Parameters
    ----------
    pool : str
        ``'last'`` — last non-pad token (causal LM); ``'mean'`` — masked mean.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        try:
            out = model(**enc, output_hidden_states=True)
            hidden = out.hidden_states[-1].float()
        except TypeError:
            out = model(**enc)
            hidden = out.hidden_states[-1].float() if hasattr(out, "hidden_states") else out.float()
        att = enc["attention_mask"]
        if pool == "last":
            last_pos = att.sum(dim=1).clamp(min=1) - 1
            vec = hidden[torch.arange(hidden.size(0), device=device), last_pos]
        elif pool == "mean":
            mask = att.unsqueeze(-1).float()
            vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            raise ValueError("pool must be 'last' or 'mean'")
        if normalize:
            vec = torch.nn.functional.normalize(vec, dim=1)
        rows.append(vec.detach())
    return torch.cat(rows, dim=0)


def make_encoder(
    model: Any,
    tokenizer: Any,
    *,
    batch_size: int = 8,
    max_length: int = 512,
    pool: str = "last",
    use_chat_template: bool = True,
    device: torch.device | str | None = None,
) -> Callable[[Sequence[str]], torch.Tensor]:
    """Build ``encoder(texts) -> [B, d]`` for :func:`estimate_from_config` D7."""

    def _enc(texts: Sequence[str]) -> torch.Tensor:
        return encode_texts(
            model,
            tokenizer,
            texts,
            batch_size=batch_size,
            max_length=max_length,
            pool=pool,
            device=device,
            normalize=True,
        )

    return _enc


@torch.no_grad()
def encode_style_deltas(
    pairs: Sequence[StylePairRecord] | Sequence[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    batch_size: int = 8,
    max_length: int = 512,
    use_chat_template: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Embedding deltas ``h(style) - h(base)`` for each style variant (Lemma D7 inputs)."""
    dicts = [p.to_dict() if isinstance(p, StylePairRecord) else p for p in pairs]
    encoder = make_encoder(
        model,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        use_chat_template=use_chat_template,
        device=device,
    )

    def _format_encoder(texts: Sequence[str]) -> torch.Tensor:
        return encoder(texts)

    return _embed_style_deltas(dicts, _format_encoder)


def estimate_style_sigma(
    pairs: Sequence[StylePairRecord] | Sequence[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    rank: int = 128,
    shrinkage: float = 1e-6,
    config: SigmaTaskConfig | None = None,
    **encode_kw: Any,
) -> SigmaTaskEstimate:
    """End-to-end D7: encode style pairs with HF model, return :class:`SigmaTaskEstimate`."""
    deltas = encode_style_deltas(pairs, model, tokenizer, **encode_kw)
    cfg = config or SigmaTaskConfig.for_alignment(rank=rank, shrinkage=shrinkage)
    cfg.encoder = None  # already encoded
    return estimate_from_config(cfg, deltas)
