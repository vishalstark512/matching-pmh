# API reference

!!! tip "Adoption path"
    New here? [Find your application](../APPLICATIONS.md) → [Golden paths](../GOLDEN_PATHS.md).  
    This page is **lookup** after you know your task.

| Page | When to read |
|------|----------------|
| [Developer API](developer.md) | `robust_fit`, `check_applicability`, `evaluate_*` |
| [PMHTrainer](pmh-trainer.md) | PyTorch estimate + train |
| [Subtypes](subtypes.md) | `suggest_subtype`, D1–D7 registry |
| [Custom geometry](custom.md) | Your deltas / W / saved artifact |
| [Deployment](deployment.md) | Ship `sigma_task` bundle |
| [Training primitives](../training.md) | `PMHLoss`, `PMHConfig` |
| [CLI](../cli.md) | `pmh-train route`, `wizard`, `validate` |

**Paper contracts:** [CORRECT_USAGE.md](../CORRECT_USAGE.md) (Research tab).

---

## PMHMatcher (sklearn)

::: pmh.matcher.PMHMatcher
    options:
      members:
        - __init__
        - fit
        - transform
        - predict
        - predict_proba
      show_root_heading: true

## TDI and geometry

::: pmh.tdi
    options:
      members:
        - TDIReport
        - tdi_cls
        - tdi_layout
        - tdi_feature_isotropic
        - trajectory_tdi_layerwise
        - trajectory_tdi_encoder
        - directional_drift_numpy
        - geometry_report
      show_root_heading: false

## Benchmarks

::: pmh.compare.compare_arms_sklearn

::: pmh.benchmark.report.benchmark_to_markdown

::: pmh.sklearn_pipeline.make_pmh_pipeline
