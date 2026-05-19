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
from pmh.benchmark.sklearn_protocol import run_sklearn_benchmark

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
]
