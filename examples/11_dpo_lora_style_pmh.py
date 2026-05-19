#!/usr/bin/env python3
"""T7A-style D7 estimate + optional LoRA fine-tune with matched PMH (Qwen JSONL).

Schema (paper Task 7A):

- ``style_pairs.jsonl``: ``prompt``, ``content_fixed``, ``style_variants`` (dict)
- ``preference_pairs.jsonl``: ``prompt``, ``chosen``, ``rejected``, ``style_variants`` (list)

Bundled samples: ``examples/data/``. CLI equivalent::

    pmh-train estimate --config examples/configs/d7_style_estimate.json
    pmh-train run --config examples/configs/dpo_train_job.json

CPU / CI (no GPU model)::

    python examples/11_dpo_lora_style_pmh.py

GPU + Qwen::

    pip install -e ".[hf-lora]"
    python examples/11_dpo_lora_style_pmh.py --model-id Qwen/Qwen2.5-0.5B-Instruct --train --lora
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STYLE = ROOT / "examples" / "data" / "style_pairs_sample.jsonl"
DEFAULT_PREF = ROOT / "examples" / "data" / "preference_pairs_sample.jsonl"


class HashEncoder(nn.Module):
    """Deterministic toy encoder for CPU / CI (same idea as ``08_hf_style_d7.py``)."""

    def __init__(self, dim: int = 64, n_vocab: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.n_vocab = n_vocab
        self.lm_head = nn.Linear(dim, n_vocab)
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        b, t = input_ids.shape
        h = torch.zeros(b, t, self.dim, device=input_ids.device)
        for i in range(b):
            h[i] = F.one_hot(input_ids[i] % self.dim, self.dim).float().mean(0)
        logits = self.lm_head(h)
        return type("Out", (), {"logits": logits, "hidden_states": (h,)})()


class ToyTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    chat_template = None

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=128):
        rows = [[hash(w) % 997 for w in t.split()[:max_length]] or [0] for t in texts]
        max_len = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), max_len, dtype=torch.long)
        mask = torch.zeros(len(rows), max_len, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            mask[i, : len(r)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


class PreferenceDataset(Dataset):
    def __init__(self, records) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        return {"prompt": r.prompt, "chosen": r.chosen, "rejected": r.rejected}


def collate_tokenize(batch, tokenizer, max_length: int) -> dict[str, torch.Tensor]:
    def _fmt(prompt: str, response: str) -> str:
        if getattr(tokenizer, "chat_template", None):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        return f"User: {prompt}\nAssistant: {response}"

    texts = [_fmt(b["prompt"], b["chosen"]) for b in batch]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    labels = enc["input_ids"].clone()
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    return {**enc, "labels": labels}


def dpo_style_loss(model, batch_chosen, batch_rejected, beta: float = 0.1) -> torch.Tensor:
    def logp(b):
        out = model(**b)
        logits = out.logits.float()
        labels = b["labels"]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        mask = labels.ne(-100).float()
        return (token_lp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return -F.logsigmoid(beta * (logp(batch_chosen) - logp(batch_rejected))).mean()


def load_model_and_tokenizer(model_id: str | None, *, use_lora: bool):
    if not model_id:
        print("Using toy HashEncoder (pass --model-id for a real causal LM).")
        return HashEncoder(64), ToyTokenizer()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError('LoRA requires: pip install "matching-pmh[hf-lora]"') from exc
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style-jsonl", type=Path, default=DEFAULT_STYLE)
    parser.add_argument("--preference-jsonl", type=Path, default=DEFAULT_PREF)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--artifact-out", type=Path, default=ROOT / "artifacts" / "d7_style")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")

    from pmh import PMHConfig
    from pmh.integrations.hf_trainer import compute_pmh_training_loss
    from pmh.integrations.huggingface import (
        estimate_style_sigma,
        load_preference_pairs_jsonl,
        load_style_pairs_jsonl,
    )
    from pmh.training import PMHLoss

    style_pairs = load_style_pairs_jsonl(args.style_jsonl, max_pairs=args.max_pairs)
    model, tokenizer = load_model_and_tokenizer(args.model_id, use_lora=False)

    print(f"Estimating D7 from {len(style_pairs)} style records ...")
    artifact = estimate_style_sigma(
        style_pairs,
        model,
        tokenizer,
        rank=args.rank,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    pt = artifact.save(args.artifact_out)
    print(f"Saved {pt}  preflight={artifact.preflight}  dim={artifact.dim}")

    if not args.train:
        print("Estimate done. Pass --train for a short DPO+PMH loop.")
        return

    pref = load_preference_pairs_jsonl(args.preference_jsonl, max_pairs=args.max_pairs)
    if args.lora and args.model_id:
        model, tokenizer = load_model_and_tokenizer(args.model_id, use_lora=True)

    pmh_mod = PMHLoss(artifact, PMHConfig(weight=0.2, cap_ratio=0.3))
    loader = DataLoader(PreferenceDataset(pref), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    model.train()

    for step, batch_raw in enumerate(loader):
        if step >= args.max_steps:
            break
        batch_c = collate_tokenize(batch_raw, tokenizer, args.max_length)
        batch_r = collate_tokenize(
            [{"prompt": b["prompt"], "chosen": b["rejected"]} for b in batch_raw],
            tokenizer,
            args.max_length,
        )
        device = next(model.parameters()).device
        batch_c = {k: v.to(device) for k, v in batch_c.items()}
        batch_r = {k: v.to(device) for k, v in batch_r.items()}

        dpo_loss = dpo_style_loss(model, batch_c, batch_r, beta=args.beta)
        _, _, pmh_term = compute_pmh_training_loss(model, batch_c, pmh_mod)
        loss = dpo_loss + pmh_term
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step={step}  dpo={dpo_loss.item():.4f}  pmh={pmh_term.item():.4f}")

    print("Training demo finished.")


if __name__ == "__main__":
    main()
