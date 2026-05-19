"""High-level arm comparison (B0 / matched / wrong-W / isotropic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pmh.benchmark.presets import get_preset
from pmh.benchmark.protocol import BenchmarkResult, run_benchmark_protocol
from pmh.benchmark.report import benchmark_to_markdown, write_benchmark_report
from pmh.benchmark.sklearn_protocol import (
    run_sklearn_benchmark,
    run_sklearn_benchmark_multi_seed,
)

__all__ = [
    "compare_arms",
    "compare_arms_sklearn",
    "benchmark_to_markdown",
    "write_benchmark_report",
]


def _apply_preset(
    preset: str | None,
    *,
    rank: int | None,
    pmh_config: Any,
    arms: Any,
    kwargs: dict[str, Any],
) -> tuple[int | None, Any, Any, dict[str, Any]]:
    if preset is None:
        return rank, pmh_config, arms, kwargs
    p = get_preset(preset)
    kw = {**p.sklearn_benchmark, **kwargs}
    r = rank if rank is not None else kw.pop("rank", p.default_rank)
    if pmh_config is None:
        pmh_config = p.pmh_config
    if arms is None:
        arms = p.arms
    return r, pmh_config, arms, kw


def compare_arms(
    artifact,
    model_factory,
    setup_model=None,
    train_loader=None,
    val_loader=None,
    *,
    setup_fn=None,
    preset: str | None = None,
    epochs: int = 10,
    pmh_config=None,
    arms=None,
    device=None,
    max_steps_per_epoch: int | None = None,
    report_dir: str | Path | None = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """Run standard falsification arms on **your** PyTorch model.

    Pass ``preset='t4_domain_d4'`` (etc.) for paper block defaults — see
    :mod:`pmh.benchmark.presets` and :doc:`CORRECT_USAGE`.
    """
    setup = setup_model or setup_fn
    if setup is None:
        raise ValueError("Pass setup_model= (encoder, head, optimizer factory)")
    if preset is not None:
        p = get_preset(preset)
        if pmh_config is None:
            pmh_config = p.pmh_config
        if arms is None:
            arms = list(p.arms)
        kwargs.setdefault("wrong_rank", p.wrong_rank or p.default_rank)
        kwargs.setdefault("wrong_seed", p.wrong_seed)
        kwargs = {**p.pytorch_benchmark, **kwargs}
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
    if preset is not None:
        result.notes.insert(0, f"Preset: {preset} — {get_preset(preset).description}")
    if report_dir is not None:
        write_benchmark_report(result, report_dir)
    return result


def compare_arms_sklearn(
    x_source,
    y_source,
    x_target,
    y_target,
    *,
    preset: str | None = None,
    rank: int | None = None,
    include_coral: bool = True,
    include_geometry: bool = True,
    seeds: list[int] | tuple[int, ...] | None = None,
    report_dir: str | Path | None = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """Four-arm compare on frozen NumPy features (optional CORAL + TDI / D_N/D_S).

    Use ``preset='t1_office31_sklearn'`` for T1 pool/test protocol (rank 32, no test leakage).
    Use ``seeds=[0, 42, 142]`` for mean accuracy over seeds (paper ``n_seeds`` style).
    """
    rank, _, _, kw = _apply_preset(preset, rank=rank, pmh_config=None, arms=None, kwargs=kwargs)
    if preset is not None and "include_coral" not in kwargs:
        include_coral = get_preset(preset).sklearn_benchmark.get("include_coral", include_coral)

    run_kw = dict(
        rank=rank or 16,
        include_coral=include_coral,
        include_geometry=include_geometry,
        **kw,
    )
    if seeds is not None and len(seeds) > 1:
        result = run_sklearn_benchmark_multi_seed(
            x_source, y_source, x_target, y_target, seeds=seeds, **run_kw
        )
    else:
        seed = int(seeds[0]) if seeds else int(run_kw.get("seed", 0))
        run_kw["seed"] = seed
        result = run_sklearn_benchmark(
            x_source, y_source, x_target, y_target, **run_kw
        )
    if preset is not None:
        result.notes.insert(0, f"Preset: {preset} — {get_preset(preset).description}")
    if report_dir is not None:
        write_benchmark_report(result, report_dir)
    return result
