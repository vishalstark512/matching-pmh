"""Remove consolidated doc files (redirects live in mkdocs.yml only)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "docs"

# Same keys as former stubs — files deleted, not replaced.
TO_DELETE = [
    "START_HERE.md",
    "MAP.md",
    "WHAT_IS_PMH.md",
    "MODES.md",
    "FIRST_HOUR.md",
    "GETTING_STARTED.md",
    "COLAB.md",
    "DEMO_OUTPUT.md",
    "QUICKSTART.md",
    "CHOOSE_YOUR_SETUP.md",
    "getting-started.md",
    "ADAPT_YOUR_PIPELINE.md",
    "ARCHITECTURES.md",
    "ARCHITECTURE.md",
    "hooks.md",
    "sklearn.md",
    "integrations.md",
    "integrations-lightning.md",
    "integrations-hf.md",
    "integrations-hf-trainer.md",
    "gallery/README.md",
    "gallery/vision.md",
    "gallery/tabular.md",
    "gallery/nlp.md",
    "ADAPTATION_WORKBOOK.md",
    "FIDELITY_BY_SUBTYPE.md",
    "BENCHMARKS.md",
    "CORRECT_USAGE.md",
    "training.md",
    "api/tier0.md",
    "api/recipe.md",
    "api/protocol.md",
    "api/developer.md",
    "api/pmh-trainer.md",
    "api/subtypes.md",
    "api/custom.md",
    "api/deployment.md",
    "DEVELOPER_ONBOARDING_PLAN.md",
    "NUISANCE_SUBTYPE_PLAN.md",
    "ROADMAP.md",
    "PHILOSOPHY.md",
    "HYBRID_NUISANCE.md",
    "datasets.md",
    "DATA_POLICY.md",
    "DATA_LAYOUT.md",
    "DEPLOYMENT.md",
    "cli.md",
    "GITHUB_PAGES_SETUP.md",
    "nuisance_types.md",
    "estimators/d1.md",
    "estimators/d2.md",
    "estimators/d3.md",
    "estimators/d4.md",
    "estimators/d5.md",
    "estimators/d6.md",
    "estimators/d7.md",
    "recipes/t1-office31-d1.md",
    "recipes/t2a-vit-isotropic.md",
    "recipes/t4-domain-d4.md",
    "recipes/t7a-style-d7.md",
    "benchmarks/office31_amazon_to_dslr.md",
    "benchmarks/office31_synthetic_reference.md",
]

EXTRA_REDIRECTS = {
    "ARCHITECTURE.md": "META_STRUCTURE.md",
    "GITHUB_PAGES_SETUP.md": "DOCS_GUIDE.md",
    "estimators/d1.md": "estimators/index.md",
    "estimators/d2.md": "estimators/index.md",
    "estimators/d3.md": "estimators/index.md",
    "estimators/d4.md": "estimators/index.md",
    "estimators/d5.md": "estimators/index.md",
    "estimators/d6.md": "estimators/index.md",
    "estimators/d7.md": "estimators/index.md",
    "recipes/t1-office31-d1.md": "PAPER_ALIGNMENT.md",
    "recipes/t2a-vit-isotropic.md": "PAPER_ALIGNMENT.md",
    "recipes/t4-domain-d4.md": "PAPER_ALIGNMENT.md",
    "recipes/t7a-style-d7.md": "PAPER_ALIGNMENT.md",
    "recipes/README.md": "walkthroughs/index.md",
    "benchmarks/office31_amazon_to_dslr.md": "walkthroughs/index.md",
    "benchmarks/office31_synthetic_reference.md": "walkthroughs/index.md",
    "FIDELITY_BY_SUBTYPE.md": "PAPER_ALIGNMENT.md",
    "CORRECT_USAGE.md": "PAPER_ALIGNMENT.md",
    "BENCHMARKS.md": "walkthroughs/index.md",
}


def main() -> None:
    removed = 0
    for rel in TO_DELETE:
        path = ROOT / rel
        if path.is_file():
            path.unlink()
            removed += 1
            print("deleted", rel)
    # recipes README if still stub
    for empty_dir in [ROOT / "gallery", ROOT / "benchmarks"]:
        if empty_dir.is_dir() and not any(empty_dir.iterdir()):
            empty_dir.rmdir()
            print("rmdir", empty_dir.relative_to(ROOT))
    print(f"removed {removed} files")
    print("Add EXTRA_REDIRECTS to mkdocs.yml (see script source)")


if __name__ == "__main__":
    main()
