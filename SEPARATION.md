# Repository separation

This project **must** live in its own git repository, separate from the paper LaTeX / experiment codebase.

## Do

- Version, release, and license **here** only.
- Keep dependencies minimal (`torch`, `numpy`).
- Add integrations (Hugging Face, Lightning) in this repo or optional extras.

## Do not

- Import from `Paper2/T1`, `T7`, or `submission_grand_unification/`.
- Copy frozen task JSON or appendix tables into this tree.
- Block paper submission on library polish.

## Paper ↔ library link

In the manuscript: one sentence in the reproducibility paragraph, e.g.  
*“Reference implementation: [matching-pmh](https://github.com/…).”*

In this README: paper title + “cite when using the package.”

No shared CI, no monorepo requirement.
