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
    return _PREFLIGHT_HINTS.get(key, f"Status {status!r}. See docs/TROUBLESHOOTING.md#glossary.")


@dataclass(frozen=True)
class SetupRecommendation:
    """Plain-language setup path for integrators."""

    stack: Stack
    nuisance: str
    title: str
    summary: str
    install_extra: str
    example_script: str
    doc_link: str
    snippet: str


def recommend_setup(
    *,
    stack: Stack = "pytorch",
    has_target_domain: bool = True,
    has_target_labels: bool = False,
    has_frozen_features: bool = False,
    has_style_pairs: bool = False,
) -> SetupRecommendation:
    """Return the default integration path for common developer situations."""
    if has_style_pairs or stack == "hf":
        return SetupRecommendation(
            stack="hf",
            nuisance="style",
            title="LLM style / format shift",
            summary="Same factual content, different formatting — style-pair JSONL + HF hidden states.",
            install_extra='pip install "matching-pmh[hf]"',
            example_script="examples/08_hf_style_d7.py",
            doc_link="docs/walkthroughs/06-llm-style-d7.md",
            snippet=(
                "from pmh import PMHTrainer, PMHConfig\n"
                "trainer = PMHTrainer(model, hook=hook, nuisance='style', pmh_config=PMHConfig.balanced())\n"
                "trainer.estimate(style_jsonl='YOUR_PAIRS.jsonl', model_id='YOUR_MODEL')\n"
                "trainer.fit(train_loader, epochs=YOUR_EPOCHS)"
            ),
        )

    if has_frozen_features or stack == "sklearn":
        return SetupRecommendation(
            stack="sklearn",
            nuisance="domain_shift",
            title="Frozen features + sklearn",
            summary="You already have embeddings; adapt source features using target domain geometry.",
            install_extra='pip install "matching-pmh[sklearn]"',
            example_script="examples/06_office31_sklearn.py",
            doc_link="docs/COLAB.md (sklearn notebook) / docs/walkthroughs/03-office31-sklearn-d1.md",
            snippet=(
                "from pmh import PMHMatcher\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "pipe = Pipeline([\n"
                "    ('adapt', PMHMatcher(nuisance='domain_shift').fit(x_source, x_target)),\n"
                "    ('clf', LogisticRegression(max_iter=500)),\n"
                "])\n"
                "pipe.fit(x_source, y_source)"
            ),
        )

    if has_target_labels and has_target_domain:
        nuisance = "subspace"
        title = "Labeled source and target"
        summary = "Class labels on both domains — stronger subspace estimate than domain-only.",
    else:
        nuisance = "domain_shift"
        title = "Domain shift (default)"
        summary = "Source vs target batches; target labels not required."

    return SetupRecommendation(
        stack="pytorch",
        nuisance=nuisance,
        title=title,
        summary=summary,
        install_extra="pip install matching-pmh torch",
        example_script="examples/00_first_run_domain_shift.py",
        doc_link="docs/COLAB.md (or docs/FIRST_HOUR.md)",
        snippet=(
            "from pmh import PMHTrainer, PMHConfig\n"
            "trainer = PMHTrainer(\n"
            "    model, hook=backbone, nuisance='"
            + nuisance
            + "',\n"
            "    pmh_config=PMHConfig.balanced(),\n"
            ")\n"
            "trainer.fit(train_loader, source_batches=src, target_batches=tgt, epochs=20)"
        ),
    )


def format_setup_guide(rec: SetupRecommendation) -> str:
    lines = [
        f"Recommended: {rec.title}",
        f"  {rec.summary}",
        f"  nuisance={rec.nuisance!r}  stack={rec.stack}",
        f"  Install: {rec.install_extra}",
        f"  Example: {rec.example_script}",
        f"  Doc: {rec.doc_link}",
    ]
    if rec.stack == "pytorch":
        lines.append(
            "  Colab: https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/domain_shift_first_run.ipynb"
        )
    if rec.stack == "sklearn":
        lines.append(
            "  Colab: https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/sklearn_frozen_features_first_run.ipynb"
        )
    lines.extend(["", "Snippet:", rec.snippet])
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
    has_target_domain: bool | None = None,
    has_target_labels: bool | None = None,
    has_frozen_features: bool | None = None,
    has_style_pairs: bool | None = None,
    interactive: bool = True,
    input_fn: Callable[[str], str] = input,
) -> SetupRecommendation:
    """Interactive or flag-driven setup recommendation."""
    if interactive and stack is None:
        print("matching-pmh setup wizard")
        print("Train on one environment, deploy on another — same labels.\n")
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

    rec = recommend_setup(
        stack=stack,
        has_target_domain=has_target_domain,
        has_target_labels=has_target_labels,
        has_frozen_features=bool(has_frozen_features),
        has_style_pairs=bool(has_style_pairs),
    )

    print()
    print(format_setup_guide(rec))
    print()
    print("Next steps:")
    print(f"  1. {rec.install_extra}")
    print(f"  2. python {rec.example_script}")
    print("  3. docs/FIRST_HOUR.md")
    if rec.stack == "pytorch":
        print(
            "  Colab: https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/domain_shift_first_run.ipynb"
        )
    if rec.stack == "sklearn":
        print(
            "  Colab: https://colab.research.google.com/github/vishalstark512/matching-pmh/blob/main/notebooks/sklearn_frozen_features_first_run.ipynb"
        )
    print("  Glossary: docs/TROUBLESHOOTING.md#plain-language-glossary")
    print("  Demo output: docs/DEMO_OUTPUT.md")
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
        help="Interactive questionnaire (same as pmh-train wizard)",
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
