"""Multi-arm comparison (B0 / matched / wrong-W / isotropic) for your pipeline."""

from pmh.benchmark.arms import (
    ARM_SPECS,
    STANDARD_ARMS,
    ArmSpec,
    normalize_arm,
    resolve_arms,
)
from pmh.benchmark.protocol import (
    ArmRunResult,
    BenchmarkResult,
    run_benchmark_protocol,
    train_one_arm,
)
from pmh.benchmark.report import (
    benchmark_to_markdown,
    load_benchmark_report,
    write_benchmark_report,
)
from pmh.benchmark.presets import (
    BlockPreset,
    SUBTYPE_TO_BLOCK_PRESET,
    get_preset,
    get_subtype_preset,
    list_presets,
)
from pmh.benchmark.validate import ValidationReport, validate_falsification
from pmh.benchmark.protocol import (
    collect_val_embeddings,
    default_geometry_metric,
)
from pmh.benchmark.sklearn_protocol import (
    run_sklearn_benchmark,
    run_sklearn_benchmark_multi_seed,
)

__all__ = [
    "ARM_SPECS",
    "STANDARD_ARMS",
    "ArmSpec",
    "normalize_arm",
    "resolve_arms",
    "ArmRunResult",
    "BenchmarkResult",
    "run_benchmark_protocol",
    "train_one_arm",
    "benchmark_to_markdown",
    "load_benchmark_report",
    "write_benchmark_report",
    "run_sklearn_benchmark",
    "run_sklearn_benchmark_multi_seed",
    "get_preset",
    "get_subtype_preset",
    "list_presets",
    "SUBTYPE_TO_BLOCK_PRESET",
    "BlockPreset",
    "validate_falsification",
    "ValidationReport",
    "collect_val_embeddings",
    "default_geometry_metric",
]
