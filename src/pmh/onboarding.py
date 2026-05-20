"""Developer onboarding helpers (no paper vocabulary required)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Stack = Literal["pytorch", "sklearn", "hf"]

_PREFLIGHT_HINTS = {
    "pass": "Geometry estimate looks usable. Proceed; run controls before large claims.",
    "marginal": "Weak shift signal. Use more source/target batches or lower rank (see Troubleshooting glossary).",
    "fail": "Estimate not reliable yet. Fix hook/data/rank before trusting PMH.",
}


def preflight_plain_english(status: str | None) -> str:
    """Map artifact preflight status to a short developer message."""
    if status is None:
        return "No preflight run yet."
    key = str(status).strip().lower()
    return _PREFLIGHT_HINTS.get(key, f"Status {status!r}. See docs/WHEN_PMH_HELPS.md.")


@dataclass(frozen=True)
class SetupRecommendation:
    """Plain-language setup path for integrators."""

    stack: Stack
    nuisance: str
    lemma: str
    title: str
    summary: str
    install_extra: str
    example_script: str
    doc_link: str
    subtype_doc: str
    snippet: str


def recommend_setup(
    *,
    stack: Stack = "pytorch",
    has_target_domain: bool = True,
    has_target_labels: bool = False,
    has_frozen_features: bool = False,
    has_style_pairs: bool = False,
    has_augmentation_modes: bool = False,
    has_temporal_sequences: bool = False,
    has_nuisance_indices: bool = False,
    noise_level_known: bool = False,
    lemma: str | None = None,
) -> SetupRecommendation:
    """Return the default integration path for common developer situations."""
    from pmh.suggest import suggest_nuisance

    sug = suggest_nuisance(
        has_source_labels=True,
        has_target_labels=has_target_labels,
        has_target_domain=has_target_domain,
        has_augmentation_modes=has_augmentation_modes,
        has_style_pairs=has_style_pairs,
        has_temporal_sequences=has_temporal_sequences,
        has_nuisance_indices=has_nuisance_indices,
        noise_level_known=noise_level_known,
    )
    from pmh.subtypes import get_subtype

    method = lemma or sug.method
    subtype_doc = f"docs/{get_subtype(method).doc_anchor}"

    if has_style_pairs or stack == "hf":
        return SetupRecommendation(
            stack="hf",
            nuisance="style",
            lemma="D7",
            title="LLM style / format shift",
            summary="Same factual content, different formatting — style-pair JSONL + HF hidden states.",
            install_extra='pip install "matching-pmh[hf]"',
            example_script="notebooks/tasks/t07a-llm-style.ipynb",
            doc_link="docs/tasks/t07a-llm-style.md",
            subtype_doc=subtype_doc,
            snippet=(
                "from pmh import PMHTrainer, PMHConfig\n"
                "trainer = PMHTrainer(model, hook=hook, nuisance='style', pmh_config=PMHConfig.balanced())\n"
                "trainer.estimate(style_jsonl='YOUR_PAIRS.jsonl', model_id='YOUR_MODEL')\n"
                "trainer.fit(train_loader, epochs=YOUR_EPOCHS)"
            ),
        )

    if has_frozen_features or stack == "sklearn":
        nui = "subspace" if has_target_labels else "domain_shift"
        lem = "D1" if has_target_labels else "D4"
        return SetupRecommendation(
            stack="sklearn",
            nuisance=nui,
            lemma=lem,
            title="Frozen features + sklearn",
            summary="You already have embeddings; adapt source features using target domain geometry.",
            install_extra='pip install "matching-pmh[sklearn]"',
            example_script="scripts/demos/office31_sklearn.py",
            doc_link="docs/tasks/t01-classical.md",
            subtype_doc=f"docs/{get_subtype(lem).doc_anchor}",
            snippet=(
                "from pmh import PMHMatcher\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "pipe = Pipeline([\n"
                f"    ('adapt', PMHMatcher(nuisance='{nui}').fit(x_source, x_target)),\n"
                "    ('clf', LogisticRegression(max_iter=500)),\n"
                "])\n"
                "pipe.fit(x_source, y_source)"
            ),
        )

    nuisance = sug.nuisance
    title = f"Subtype {sug.method}: {nuisance}"
    summary = sug.reason

    return SetupRecommendation(
        stack="pytorch",
        nuisance=nuisance,
        lemma=sug.method,
        title=title,
        summary=summary,
        install_extra="pip install matching-pmh torch",
        example_script="scripts/demos/first_run_domain_shift.py",
        doc_link="docs/tasks/t04a-vision-domain.md",
        subtype_doc=subtype_doc,
        snippet=(
            "from pmh import PMHTrainer, PMHConfig\n"
            "trainer = PMHTrainer(\n"
            "    model, hook=backbone, nuisance='"
            + nuisance
            + "',\n"
            "    pmh_config=PMHConfig.golden_path(),  # PMH capped at ~25% of task loss\n"
            ")\n"
            "trainer.fit(train_loader, source_batches=src, target_batches=tgt, epochs=20)"
        ),
    )


def format_setup_guide(rec: SetupRecommendation) -> str:
    lines = [
        f"Recommended: {rec.title}",
        f"  {rec.summary}",
        f"  subtype={rec.lemma}  nuisance={rec.nuisance!r}  stack={rec.stack}",
        f"  Subtype guide: {rec.subtype_doc}",
        f"  Install: {rec.install_extra}",
        f"  Example: {rec.example_script}",
        f"  Doc: {rec.doc_link}",
    ]
    if rec.stack == "pytorch":
        lines.append(
            "  Colab: https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/tasks/t04a-vision-domain.ipynb"
        )
    if rec.stack == "sklearn":
        lines.append(
            "  Colab: https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/tasks/t01-classical.ipynb"
        )
    lines.extend(
        [
            "",
            "Loss scale: PMH term is hard-capped to 5--30% of task loss (PMHConfig.pmh_max_task_ratio).",
            "  Quick try: pmh-train try --quick",
            "  Docs: docs/LOSS_SCALING.md",
            "",
            "Snippet:",
            rec.snippet,
        ]
    )
    return "\n".join(lines)


def print_setup_guide(**kwargs) -> SetupRecommendation:
    rec = recommend_setup(**kwargs)
    print(format_setup_guide(rec))
    return rec


def _ask_choice(prompt: str, choices: dict[str, str], input_fn: Callable[[str], str]) -> str:
    print(prompt)
    keys = list(choices.keys())
    for k in keys:
        print(f"  [{k}] {choices[k]}")
    while True:
        raw = input_fn("> ").strip().lower()
        if raw in choices:
            return raw
        print(f"  Choose one of: {', '.join(keys)}")


def _ask_yes_no(prompt: str, default: bool, input_fn: Callable[[str], str]) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input_fn(f"{prompt} [{hint}] ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer y or n.")


def run_wizard(
    *,
    stack: Stack | None = None,
    task_id: str | None = None,
    has_target_domain: bool | None = None,
    has_target_labels: bool | None = None,
    has_frozen_features: bool | None = None,
    has_style_pairs: bool | None = None,
    interactive: bool = True,
    input_fn: Callable[[str], str] = input,
) -> SetupRecommendation:
    """Interactive or flag-driven setup recommendation."""
    from pmh.task_router import TaskRoute, explain_task, format_task_menu, get_task, route_from_wizard_choice

    selected_task: TaskRoute | None = None

    def _apply_task(tr: TaskRoute) -> None:
        nonlocal stack, has_frozen_features, has_style_pairs, has_target_labels, selected_task
        selected_task = tr
        stack = tr.stack
        if stack == "sklearn":
            has_frozen_features = True
        if stack == "hf":
            has_style_pairs = True
        if tr.lemma == "D1":
            has_target_labels = True

    if task_id:
        selected_task = get_task(task_id)
        print(explain_task(task_id))
        print()
        _apply_task(selected_task)

    if interactive and stack is None and selected_task is None:
        from pmh.adoption import format_recipe_banner

        print("matching-pmh setup wizard")
        print(format_recipe_banner(trailing="Guide: docs/START.md\n"))
        if not _ask_yes_no(
            "Same label semantics on train (site A) and deploy (site B)?",
            default=True,
            input_fn=input_fn,
        ):
            print("  PMH is for label-preserving deploy shift only. See docs/WHEN_PMH_HELPS.md")
        print(format_task_menu())
        from pmh.task_router import TASK_IDS

        n = len(TASK_IDS)
        tchoice = _ask_choice(
            "Pick your task (or skip to stack-only setup)",
            {str(i): f"Task {i}" for i in range(1, n + 1)}
            | {"0": "Skip — I only know my stack (PyTorch / sklearn / HF)"},
            input_fn,
        )
        if tchoice != "0":
            tid = route_from_wizard_choice(tchoice)
            if tid:
                print(explain_task(tid))
                print()
                _apply_task(get_task(tid))

    if interactive and stack is None:
        choice = _ask_choice(
            "What are you integrating with?",
            {
                "1": "PyTorch model (CNN, ViT, custom nn.Module)",
                "2": "Frozen feature matrices + sklearn",
                "3": "Hugging Face LLM (style / format shift)",
            },
            input_fn,
        )
        stack = {"1": "pytorch", "2": "sklearn", "3": "hf"}[choice]
        if stack == "sklearn":
            has_frozen_features = True
        if stack == "hf":
            has_style_pairs = True

    if stack is None:
        stack = "pytorch"

    if stack == "hf":
        has_style_pairs = True if has_style_pairs is None else has_style_pairs
        has_target_domain = has_target_domain if has_target_domain is not None else True
        has_target_labels = False
        has_frozen_features = False
    elif stack == "sklearn":
        has_frozen_features = True
        has_target_domain = has_target_domain if has_target_domain is not None else True
        has_target_labels = has_target_labels if has_target_labels is not None else True
        has_style_pairs = False
    else:
        has_frozen_features = False
        has_style_pairs = False
        if interactive and has_target_domain is None:
            has_target_domain = _ask_yes_no(
                "Do you have batches from the deployment site (target domain)?",
                default=True,
                input_fn=input_fn,
            )
        if has_target_domain is None:
            has_target_domain = True
        if interactive and has_target_labels is None and has_target_domain:
            has_target_labels = _ask_yes_no(
                "Do you have class labels on the deployment site?",
                default=False,
                input_fn=input_fn,
            )
        if has_target_labels is None:
            has_target_labels = False

    if not interactive:
        if has_target_domain is None:
            has_target_domain = True
        if has_target_labels is None:
            has_target_labels = False

    subtype_flags: dict[str, bool] = {}
    preset_lemma = selected_task.lemma if selected_task is not None else None
    if interactive and stack == "pytorch" and has_target_domain and preset_lemma is None:
        print()
        st = _ask_choice(
            "What best describes the deployment shift? (nuisance subtype)",
            {
                "1": "D1 — different site/sensor, labels on BOTH train and deploy",
                "2": "D4 — different site/sensor, deploy labels unknown",
                "3": "D3 — known augmentations / sensitivities you can enumerate",
                "4": "D6 — temporal or sequence drift (same label over time)",
                "5": "D5 — nuisance in specific coordinates (positions, tokens, …)",
                "6": "D7 — format/style only (same facts, different surface)",
                "7": "D2 — generic isotropic noise (no domain pair)",
            },
            input_fn,
        )
        from pmh.subtypes import apply_wizard_subtype_choice

        subtype_flags = apply_wizard_subtype_choice(st)

    rec = recommend_setup(
        stack=stack,
        has_target_domain=subtype_flags.get("has_target_domain", has_target_domain),
        has_target_labels=subtype_flags.get("has_target_labels", has_target_labels),
        has_frozen_features=bool(has_frozen_features),
        has_style_pairs=subtype_flags.get("has_style_pairs", bool(has_style_pairs)),
        has_augmentation_modes=subtype_flags.get("has_augmentation_modes", False),
        has_temporal_sequences=subtype_flags.get("has_temporal_sequences", False),
        has_nuisance_indices=subtype_flags.get("has_nuisance_indices", False),
        noise_level_known=subtype_flags.get("noise_level_known", False),
        lemma=preset_lemma or subtype_flags.get("lemma"),
    )

    print()
    print(format_setup_guide(rec))
    print()
    print("Next steps:")
    if selected_task is not None:
        for i, step in enumerate(selected_task.walkthrough, 1):
            print(f"  {i}. {step}")
        print(f"  Read: {selected_task.doc_one_pager}")
    else:
        print(f"  1. {rec.install_extra}")
        print(f"  2. python {rec.example_script}")
        print("  3. docs/tasks/index.md")
    print("  Golden path: pmh-train try --quick")
    if rec.stack == "pytorch":
        print(
            "  Colab T4A: https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/tasks/t04a-vision-domain.ipynb"
        )
        print(
            "  Colab T4B: https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/tasks/t04b-multilayer-vision.ipynb"
        )
    if rec.stack == "sklearn":
        print(
            "  Colab T1: https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/tasks/t01-classical.ipynb"
        )
    print("  Loss scale (5--30% of task): docs/LOSS_SCALING.md")
    print("  Expectations: docs/WHEN_PMH_HELPS.md")
    print("  Start: docs/START.md")
    return rec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Suggest a matching-pmh integration path (developer onboarding).",
    )
    p.add_argument(
        "--stack",
        choices=("pytorch", "sklearn", "hf"),
        default="pytorch",
    )
    p.add_argument("--target-domain", action="store_true", default=True)
    p.add_argument("--no-target-domain", action="store_false", dest="target_domain")
    p.add_argument("--target-labels", action="store_true")
    p.add_argument("--frozen-features", action="store_true")
    p.add_argument("--style-pairs", action="store_true")
    p.add_argument(
        "--wizard",
        action="store_true",
        help="Interactive questionnaire (same as pmh-train route --wizard)",
    )
    args = p.parse_args(argv)
    if args.wizard:
        run_wizard(interactive=True)
    else:
        print_setup_guide(
            stack=args.stack,
            has_target_domain=args.target_domain,
            has_target_labels=args.target_labels,
            has_frozen_features=args.frozen_features,
            has_style_pairs=args.style_pairs,
        )
    print("\nDocs: https://github.com/vishalstark512/matching-pmh/blob/main/docs/WHAT_IS_PMH.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
