"""Typed configuration for estimators and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Method = Literal["D1", "D2", "D3", "D4", "D5", "D6", "D7"]


def _norm_method(method: str) -> Method:
    m = method.upper()
    if m not in ("D1", "D2", "D3", "D4", "D5", "D6", "D7"):
        raise ValueError(f"method must be D1--D7, got {method!r}")
    return m  # type: ignore[return-value]


@dataclass
class SigmaTaskConfig:
    """Estimator hyperparameters (Lemma D1--D7).

    Use factory helpers ``for_domain()``, ``for_isotropic()``, etc., or pass
    ``method=`` directly.  Call ``.estimate(...)`` or ``estimate_sigma_task(config, ...)``.
    """

    method: Method = "D4"
    rank: int | None = None
    shrinkage: float = 1e-6
    # D2
    dim: int | None = None
    noise_level: float | None = None
    # D5
    nuisance_indices: list[int] | None = None
    # D7 / generic
    encoder: Any = None  # callable, not serialized in artifacts
    device: str | None = None
    dtype: str = "float32"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.method = _norm_method(self.method)
        if self.method == "D1" and self.rank is None:
            raise ValueError("D1 requires rank=")
        if self.method == "D2":
            if self.dim is None:
                raise ValueError("D2 requires dim=")
            if self.noise_level is None:
                self.noise_level = 0.1

    @classmethod
    def for_subspace(cls, rank: int, **kwargs: Any) -> SigmaTaskConfig:
        return cls(method="D1", rank=rank, **kwargs)

    @classmethod
    def for_isotropic(cls, dim: int, noise_level: float = 0.1, **kwargs: Any) -> SigmaTaskConfig:
        return cls(method="D2", dim=dim, noise_level=noise_level, **kwargs)

    @classmethod
    def for_augmentation(cls, **kwargs: Any) -> SigmaTaskConfig:
        return cls(method="D3", **kwargs)

    @classmethod
    def for_domain(cls, rank: int | None = None, **kwargs: Any) -> SigmaTaskConfig:
        return cls(method="D4", rank=rank, **kwargs)

    @classmethod
    def for_compositional(cls, nuisance_indices: list[int], **kwargs: Any) -> SigmaTaskConfig:
        return cls(method="D5", nuisance_indices=nuisance_indices, **kwargs)

    @classmethod
    def for_temporal(cls, **kwargs: Any) -> SigmaTaskConfig:
        return cls(method="D6", **kwargs)

    @classmethod
    def for_alignment(cls, rank: int = 128, **kwargs: Any) -> SigmaTaskConfig:
        return cls(method="D7", rank=rank, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "rank": self.rank,
            "shrinkage": self.shrinkage,
            "dim": self.dim,
            "noise_level": self.noise_level,
            "nuisance_indices": self.nuisance_indices,
            "dtype": self.dtype,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SigmaTaskConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        d = {k: v for k, v in data.items() if k in known and k != "encoder"}
        extra = data.get("extra", {})
        if "encoder" in data:
            d["encoder"] = data["encoder"]
        cfg = cls(**d)
        cfg.extra = dict(extra)
        return cfg


@dataclass
class PMHConfig:
    """Training-time PMH hyperparameters.

    **Loss scale (practitioner default):** keep the PMH term between roughly
    ``pmh_min_task_ratio`` and ``pmh_max_task_ratio`` of the task loss (5--30%
    is the usual band). ``cap_ratio`` is synced to ``pmh_max_task_ratio``; training
    uses ``cap_basis='task'`` so PMH never dominates classification loss.
    """

    weight: float = 0.3
    cap_ratio: float = 0.25
    cap_basis: Literal["total", "task"] = "task"
    pmh_max_task_ratio: float = 0.30
    pmh_min_task_ratio: float = 0.05
    n_probes: int = 4
    shrinkage: float = 1e-6
    warmup_epochs: int = 2
    warmup_ramp_epochs: int = 10
    warn_underpowered_pmh: bool = True

    def pmh_weight_for_epoch(self, epoch: int) -> float:
        """Ramp multiplier in [0, 1] after warmup."""
        if epoch <= self.warmup_epochs:
            return 0.0
        t = epoch - self.warmup_epochs
        if self.warmup_ramp_epochs <= 0:
            return 1.0
        return min(1.0, t / self.warmup_ramp_epochs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight,
            "cap_ratio": self.cap_ratio,
            "cap_basis": self.cap_basis,
            "pmh_max_task_ratio": self.pmh_max_task_ratio,
            "pmh_min_task_ratio": self.pmh_min_task_ratio,
            "warn_underpowered_pmh": self.warn_underpowered_pmh,
            "n_probes": self.n_probes,
            "shrinkage": self.shrinkage,
            "warmup_epochs": self.warmup_epochs,
            "warmup_ramp_epochs": self.warmup_ramp_epochs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PMHConfig:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})  # type: ignore[attr-defined]

    @classmethod
    def conservative(cls) -> PMHConfig:
        """Small PMH influence (~5--15% of task); good first try."""
        return cls(
            weight=0.15,
            cap_ratio=0.15,
            cap_basis="task",
            pmh_max_task_ratio=0.15,
            pmh_min_task_ratio=0.05,
            warmup_epochs=3,
            warmup_ramp_epochs=5,
        )

    @classmethod
    def balanced(cls) -> PMHConfig:
        """Default: PMH capped at 25% of task loss (inside 5--30% band)."""
        return cls(
            weight=0.3,
            cap_ratio=0.25,
            cap_basis="task",
            pmh_max_task_ratio=0.25,
            pmh_min_task_ratio=0.05,
            warmup_epochs=2,
            warmup_ramp_epochs=10,
        )

    @classmethod
    def aggressive(cls) -> PMHConfig:
        """Stronger geometry (still capped at 30% of task)."""
        return cls(
            weight=0.5,
            cap_ratio=0.30,
            cap_basis="task",
            pmh_max_task_ratio=0.30,
            pmh_min_task_ratio=0.05,
            warmup_epochs=1,
            warmup_ramp_epochs=5,
        )

    @classmethod
    def golden_path(cls) -> PMHConfig:
        """``try_pmh`` / ``pmh-train try`` default — strict task-ratio cap."""
        return cls.balanced()

    @classmethod
    def finetune_llm(cls) -> PMHConfig:
        """LoRA / LLM style fine-tunes: long warmup, gentle ramp."""
        return cls(weight=0.2, cap_ratio=0.25, warmup_epochs=5, warmup_ramp_epochs=15)


@dataclass
class PreflightConfig:
    """Eigengap thresholds (paper Section 5.1 practitioner guide)."""

    pass_ratio: float = 1.5
    fail_ratio: float = 1.1
    rank: int = 1
