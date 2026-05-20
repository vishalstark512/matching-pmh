#!/usr/bin/env python3
"""Hugging Face Trainer + PMH artifact (sketch).

Full DPO / margin PMH: ``paper_code/T7/task7B/``.
Library estimate: ``estimate_style_sigma`` or ``robust_fit_text_domains``.

This script shows the **estimate-once, attach artifact** pattern without
requiring a large model download.
"""

from __future__ import annotations

# Pattern for your HF training loop:
#
#   from pmh import PMHConfig, PMHTrainer
#   from pmh.integrations.huggingface import estimate_style_sigma, load_style_pairs_jsonl
#
#   pairs = load_style_pairs_jsonl("style_pairs.jsonl")
#   artifact = estimate_style_sigma(pairs, model, tokenizer, rank=32)
#
#   trainer = PMHTrainer.from_artifact(
#       model, artifact, hook="last_hidden_state", pmh_config=PMHConfig.golden_path(),
#   )
#
#   # In Trainer.compute_loss or manual loop:
#   #   loss, step = trainer.callback.training_step(batch)
#
#   # PMH is capped to pmh_max_task_ratio × task loss (default 25--30% max).

def main() -> None:
    print(__doc__)
    print("See notebooks/tasks/t07a-llm-style.ipynb and docs/MIGRATE.md (HF section).")


if __name__ == "__main__":
    main()
