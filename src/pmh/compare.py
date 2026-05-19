"""High-level arm comparison (B0 / matched / wrong-W / isotropic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pmh.benchmark.protocol import BenchmarkResult, run_benchmark_protocol
from pmh.benchmark.report import benchmark_to_markdown, write_benchmark_report
from pmh.benchmark.sklearn_protocol import run_sklearn_benchmark

__all__ = [
    "compare_arms",
    "compare_arms_sklearn",
    "benchmark_to_markdown",
    "write_benchmark_report",
]


def compare_arms(
    artifact,
    model_factory,
    setup_model=None,
    train_loader=None,
    val_loader=None,
    *,
    setup_fn=None,
    epochs: int = 10,
    pmh_config=None,
    arms=None,
    device=None,
    max_steps_per_epoch: int | None = None,
    report_dir: str | Path | None = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """Run standard falsification arms on **your** PyTorch model.

    Wraps :func:`pmh.benchmark.run_benchmark_protocol`. If ``report_dir`` is set,
    writes ``benchmark.json`` and ``benchmark.md``.
    """
    setup = setup_model or setup_fn
    if setup is None:
        raise ValueError("Pass setup_model= (encoder, head, optimizer factory)")
    result = run_benchmark_protocol(
        artifact,
        model_factory,
        setup,
        train_loader,
        val_loader,
        epochs=epochs,
        pmh_config=pmh_config,
        arms=arms,
        device=device,
        max_steps_per_epoch=max_steps_per_epoch,
        **kwargs,
    )
    if report_dir is not None:
        write_benchmark_report(result, report_dir)
    return result


def compare_arms_sklearn(
    x_source,
    y_source,
    x_target,
    y_target,
    *,
    rank: int = 16,
    include_coral: bool = True,
    report_dir: str | Path | None = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """Four-arm compare on frozen NumPy features (optional CORAL)."""
    result = run_sklearn_benchmark(
        x_source,
        y_source,
        x_target,
        y_target,
        rank=rank,
        include_coral=include_coral,
        **kwargs,
    )
    if report_dir is not None:
        write_benchmark_report(result, report_dir)
    return result
