#!/usr/bin/env python3
"""Add adoption banner, API note, plain-language fixes, and link repairs in walkthroughs."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "docs" / "walkthroughs"

LINK_FIXES = {
    "../ADAPTATION_WORKBOOK.md": "../FIVE_STEP_RECIPE.md",
    "../CHOOSE_YOUR_SETUP.md": "../APPLICATIONS.md",
    "../CORRECT_USAGE.md": "../PAPER_ALIGNMENT.md",
    "../BENCHMARKS.md": "../PAPER_ALIGNMENT.md",
    "../gallery/tabular.md": "../GOLDEN_PATHS.md#g2",
    "../gallery/nlp.md": "../GOLDEN_PATHS.md#g3",
    "../gallery/vision.md": "../GOLDEN_PATHS.md#g1",
    "../gallery/README.md": "../GOLDEN_PATHS.md",
    "../sklearn.md": "../GOLDEN_PATHS.md#g2",
    "../hooks.md": "../INTEGRATE.md",
    "../integrations-lightning.md": "../GOLDEN_PATHS.md#g1b",
    "../integrations-hf.md": "../GOLDEN_PATHS.md#g3",
    "../integrations-hf-trainer.md": "../GOLDEN_PATHS.md#g3b",
    "../integrations.md": "../GOLDEN_PATHS.md",
    "../ADAPT_YOUR_PIPELINE.md": "../INTEGRATE.md",
    "../DATA_POLICY.md": "../DOCS_GUIDE.md",
}

API_NOTE = (
    "> **API note:** `nuisance=` is the **deployment shift type** (D1–D7 API key), not “bad data.” "
    "[What is deployment shift?](../WHAT_IS_DEPLOYMENT_SHIFT.md)\n\n"
)

# Prose only — never change `nuisance=` in code fences
PROSE_REPLACEMENTS = [
    ("estimated the **right** nuisance geometry", "estimated the **right** deployment-shift geometry"),
    ("Val metric may not reflect nuisance", "Val metric may not reflect deployment shift"),
    ("check `preflight`, nuisance, Dk", "check `preflight`, shift type (`nuisance=`), Dk"),
    ("PMH nuisance geometry", "PMH deployment-shift geometry"),
    ("not PMH nuisance geometry", "not PMH deployment-shift geometry"),
    ("unless nuisance IS the transform", "unless the **shift** is the transform (D3)"),
    ("where nuisance lives in **early layers**", "where deployment shift lives in **early layers**"),
    ("separate from task coordinates", "separate from task coordinates (D5)"),
    ("partition nuisance vs task", "partition shift-related vs task token groups"),
    ("**nuisance coordinates**", "**shift-related coordinates**"),
    ("pure nuisance", "pure shift-related coords"),
    ("nuisance type, rank", "shift type (`nuisance=`), rank"),
]

# (filename, golden_path anchor, route task or None, step5 note)
BANNERS: dict[str, tuple[str, str | None, str]] = {
    "01-pytorch-domain-d4.md": ("#g1", "vision_classification", "evaluate_robust_fit or compare_arms"),
    "02-resnet-vision-d4.md": ("#g1", "vision_classification", "evaluate_robust_fit"),
    "03-office31-sklearn-d1.md": ("#g2", "frozen_embeddings_sklearn", "evaluate_baseline_vs_pmh (default Step 5)"),
    "04-multilayer-convnet.md": ("#g1", "vision_classification", "compare_arms — paper multilayer"),
    "05-compositional-d5.md": ("#g4", "compositional_coordinates", "compare_arms"),
    "06-llm-style-d7.md": ("#g3", "llm_style_or_format", "geometry + task metric separately"),
    "07-hf-trainer-d7-dpo.md": ("#g3b", "llm_style_or_format", "HF eval holdout + walkthrough 08"),
    "08-falsification-controls.md": ("", None, "evaluate_* / compare_arms — required before claims"),
    "09-cli-json-jobs.md": ("", None, "compare_arms_sklearn on saved benchmark"),
    "10-lightning.md": ("#g1b", "pytorch_lightning", "compare_arms after Lightning integrate"),
    "11-temporal-d6.md": ("#g1", "temporal_drift", "compare_arms on sequence holdout"),
    "12-vit-cls-d4.md": ("#g1", "vision_classification", "compare_arms"),
    "13-speech-whisper-d4.md": ("#g1", "vision_classification", "compare_arms on mic-shift holdout"),
    "14-qm9-molecule-d5.md": ("#g4", "compositional_coordinates", "paper T5A controls"),
    "15-codebert-tokens-d5.md": ("#g4", "compositional_coordinates", "paper token D5"),
    "16-augmentation-d3.md": ("#g1", "augmentation_robustness", "wrong-W vs matched on aug holdout"),
    "17-compare-arms-your-pipeline.md": ("", None, "examples/20_compare_training_arms.py"),
    "18-pmh-trainer-quickstart.md": ("#g1", None, "evaluate_robust_fit after fit"),
    "19-office31-real-data.md": ("#g2", "frozen_embeddings_sklearn", "compare_arms_sklearn preset t1"),
}

# After "## Who this is for" block (before next ##), insert if missing
SHIFT_SENTENCES: dict[str, str] = {
    "07-hf-trainer-d7-dpo.md": (
        "## Your deployment shift sentence\n\n"
        '*Same task, different writing style or template at deploy.* -> **D7** + HF Trainer.\n\n---\n\n'
    ),
    "08-falsification-controls.md": (
        "## Your deployment shift sentence\n\n"
        '*We fixed site/camera/style shift; gains must beat wrong-W and isotropic controls.* -> Step 5 on **deploy holdout**.\n\n---\n\n'
    ),
    "09-cli-json-jobs.md": (
        "## Your deployment shift sentence\n\n"
        '*Batch jobs estimate Sigma_task for site A vs B; same labels, reproducible JSON configs.* -> any Dk your config names.\n\n---\n\n'
    ),
    "10-lightning.md": (
        "## Your deployment shift sentence\n\n"
        '*Lightning project: train hospital A, deploy hospital B; hook on layer4 or CLS.* -> **D4** typical.\n\n---\n\n'
    ),
    "17-compare-arms-your-pipeline.md": (
        "## Your deployment shift sentence\n\n"
        '*Before we ship, matched PMH must beat B0, wrong-W, and isotropic on our deploy split.* -> required evidence.\n\n---\n\n'
    ),
    "18-pmh-trainer-quickstart.md": (
        "## Your deployment shift sentence\n\n"
        '*Quick PMHTrainer loop: name shift with `nuisance=` or `auto`, then falsify on target.* -> see Step 2 below.\n\n---\n\n'
    ),
    "19-office31-real-data.md": (
        "## Your deployment shift sentence\n\n"
        '*"Amazon vs DSLR vs webcam - same 31 classes, different imaging domain."* -> **D1** subspace on frozen features.\n\n---\n\n'
    ),
}


def _banner(gp: str, route: str | None, step5: str) -> str:
    gp_link = f"[Golden path G1–G4](../GOLDEN_PATHS.md{gp})" if gp else "[Golden paths](../GOLDEN_PATHS.md)"
    route_line = f" · **Route:** `pmh-train route --task {route}`" if route else ""
    return (
        "!!! tip \"Adopt PMH first\"\n"
        "    **Start:** [ADOPT.md](../../ADOPT.md) -> "
        f"{gp_link}{route_line} · **Step 5:** {step5}\n"
        "    This walkthrough is **evidence / depth** — not your first page.\n\n"
    )


def _apply_prose_outside_fences(text: str) -> str:
    parts = re.split(r"(```[\s\S]*?```)", text)
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        for old, new in PROSE_REPLACEMENTS:
            chunk = chunk.replace(old, new)
        parts[i] = chunk
    return "".join(parts)


def _insert_api_note(text: str) -> str:
    if "API note:** `nuisance=`" in text:
        return text
    marker = "This walkthrough is **evidence / depth** — not your first page.\n\n"
    if marker in text:
        return text.replace(marker, marker + API_NOTE, 1)
    return text


def _insert_shift_sentence(text: str, name: str) -> str:
    if "## Your deployment shift sentence" in text:
        return text
    block = SHIFT_SENTENCES.get(name)
    if not block:
        return text
    # After first --- following "Who this is for" section, or before "## Step"
    m = re.search(r"(## Who this is for\n[\s\S]*?)\n---\n", text)
    if m:
        return text[: m.end()] + "\n" + block + text[m.end() :]
    m2 = re.search(r"(## Prerequisites\n)", text)
    if m2:
        return text[: m2.start()] + block + text[m2.start() :]
    return text


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    for old, new in LINK_FIXES.items():
        text = text.replace(old, new)
    text = text.replace("## Your nuisance sentence", "## Your deployment shift sentence")
    text = text.replace(
        "## Your nuisance sentence (write this first)",
        "## Your deployment shift sentence (write this first)",
    )
    text = text.replace(
        "## Step 2 — `nuisance=\"auto\"`",
        "## Step 2 — Pick shift type (`nuisance=\"auto\"`)",
    )
    text = _apply_prose_outside_fences(text)
    name = path.name
    if name in BANNERS and "Adopt PMH first" not in text:
        gp, route, step5 = BANNERS[name]
        banner = _banner(gp, route, step5)
        lines = text.splitlines(keepends=True)
        out: list[str] = []
        inserted = False
        i = 0
        while i < len(lines):
            out.append(lines[i])
            if not inserted and lines[i].startswith("# ") and i + 1 < len(lines):
                j = i + 1
                if j < len(lines) and lines[j].strip() == "":
                    out.append(lines[j])
                    j += 1
                if j < len(lines) and not lines[j].startswith("!!! tip"):
                    out.append("\n")
                    out.append(banner)
                    inserted = True
            i += 1
        if not inserted:
            out.insert(2, "\n" + banner)
        text = "".join(out)
    text = _insert_api_note(text)
    text = _insert_shift_sentence(text, name)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    n = 0
    for path in sorted(ROOT.glob("*.md")):
        if path.name in ("index.md", "GUIDE_FORMAT.md", "DAILY_AI_USE.md", "paper-presets-by-block.md"):
            continue
        if patch_file(path):
            print("patched", path.name)
            n += 1
    print(f"done: {n} files updated")


if __name__ == "__main__":
    main()
