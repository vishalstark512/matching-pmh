#!/usr/bin/env python3
"""Lemma D7 with Hugging Face: load JSONL, encode style deltas, estimate Sigma.

Without a GPU model, runs a **toy** encoder (hash embeddings) to demonstrate the API.
Pass ``--model-id`` for a real causal LM (requires ``pip install "matching-pmh[hf]"``).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch


def _write_demo_jsonl(path: Path) -> None:
    rows = [
        {
            "id": "ex1",
            "prompt": "Summarize the paper.",
            "content_fixed": "The method matches deployment nuisance covariance.",
            "style_variants": {
                "bulleted": "- Matches Sigma_task\n- Adds PMH penalty",
                "verbose": "In this detailed response, we explain matching at length.",
            },
        },
        {
            "id": "ex2",
            "prompt": "What is PMH?",
            "content_fixed": "A Jacobian regularizer along Sigma'.",
            "style_variants": {
                "formal": "PMH denotes a penalty on representation sensitivity.",
                "casual": "It's basically a fancy robustness loss.",
            },
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class HashEncoder(torch.nn.Module):
    """Deterministic toy encoder for CI / CPU-only demos."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.dummy = torch.nn.Parameter(torch.zeros(1))  # for .parameters() / device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: object) -> object:
        del attention_mask
        # fake hidden states from token ids
        b, t = input_ids.shape
        h = torch.zeros(b, t, self.dim)
        for i in range(b):
            h[i] = torch.nn.functional.one_hot(
                input_ids[i] % self.dim, self.dim
            ).float().mean(0)
        return type("Out", (), {"hidden_states": (h,)})()


class ToyTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    chat_template = None

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=128):
        rows = []
        for t in texts:
            ids = [hash(w) % 997 for w in t.split()[:max_length]]
            if not ids:
                ids = [0]
            rows.append(ids)
        max_len = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), max_len, dtype=torch.long)
        mask = torch.zeros(len(rows), max_len, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            mask[i, : len(r)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    from pmh.integrations.huggingface import (
        estimate_style_sigma,
        load_style_pairs_jsonl,
    )

    if args.jsonl is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        path = Path(tmp.name)
        _write_demo_jsonl(path)
        print(f"Wrote demo JSONL to {path}")
    else:
        path = args.jsonl

    pairs = load_style_pairs_jsonl(path)

    if args.model_id:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        print("Using toy HashEncoder (pass --model-id for a real LM).")
        model = HashEncoder(64)
        tokenizer = ToyTokenizer()

    artifact = estimate_style_sigma(pairs, model, tokenizer, rank=args.rank, batch_size=4)
    print(f"method={artifact.method}  dim={artifact.dim}  preflight={artifact.preflight}")
    print(f"Sigma trace={artifact.sigma.trace().item():.4f}")


if __name__ == "__main__":
    main()
