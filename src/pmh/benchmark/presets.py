"""Paper block presets: nuisance, rank, tuning, and benchmark protocol per T1–T7."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pmh.benchmark.arms import STANDARD_ARMS
from pmh.config import PMHConfig, SigmaTaskConfig

BlockId = Literal[
    "t1_office31_sklearn",
    "t1_synthetic_sklearn",
    "t2a_vit_isotropic",
    "t2b_chexpert_isotropic",
    "t3b_depth_d3",
    "t4_domain_d4",
    "t5_compositional_d5",
    "t6_temporal_d6",
    "t7a_style_d7",
    "t7b_pgd_d7",
]


@dataclass(frozen=True)
class BlockPreset:
    """Recommended Phase-A + falsification settings for one paper block."""

    block_id: str
    paper_type: str
    lemma: str
    nuisance: str
    description: str
    # Phase A
    sigma_method: str
    default_rank: int
    estimate_kwargs: dict[str, Any] = field(default_factory=dict)
    # Phase B (PyTorch PMHLoss)
    pmh_config: PMHConfig = field(default_factory=PMHConfig)
    wrong_rank: int | None = None
    wrong_seed: int = 0
    # sklearn frozen-feature benchmark
    sklearn_benchmark: dict[str, Any] = field(default_factory=dict)
    pytorch_benchmark: dict[str, Any] = field(default_factory=dict)
    arms: tuple[str, ...] = STANDARD_ARMS
    application_mode: Literal["jacobian", "projection", "both"] = "jacobian"
    paper_paths: str = ""
    notes: tuple[str, ...] = ()

    def sigma_config(self, rank: int | None = None) -> SigmaTaskConfig:
        r = rank if rank is not None else self.default_rank
        m = self.sigma_method
        if m == "D1":
            return SigmaTaskConfig.for_subspace(rank=r, **self.estimate_kwargs)
        if m == "D2":
            dim = self.estimate_kwargs.get("dim")
            nl = self.estimate_kwargs.get("noise_level", 0.1)
            if dim is None:
                raise ValueError("D2 preset requires estimate_kwargs['dim']")
            return SigmaTaskConfig.for_isotropic(int(dim), float(nl))
        if m == "D3":
            return SigmaTaskConfig.for_augmentation(**self.estimate_kwargs)
        if m == "D4":
            return SigmaTaskConfig.for_domain(rank=r, **self.estimate_kwargs)
        if m == "D5":
            return SigmaTaskConfig.for_compositional(
                nuisance_indices=self.estimate_kwargs["nuisance_indices"],
                rank=r,
            )
        if m == "D6":
            return SigmaTaskConfig.for_temporal(rank=r, **self.estimate_kwargs)
        if m == "D7":
            return SigmaTaskConfig.for_alignment(rank=r, **self.estimate_kwargs)
        raise ValueError(f"unknown sigma_method {m!r}")


PRESETS: dict[str, BlockPreset] = {
    "t1_office31_sklearn": BlockPreset(
        block_id="t1_office31_sklearn",
        paper_type="T1",
        lemma="D1",
        nuisance="subspace",
        description="Office-31 Amazon→DSLR, ResNet-18 frozen features, T1 pool/test protocol.",
        sigma_method="D1",
        default_rank=32,
        estimate_kwargs={"n_pairs_per_class": 40},
        pmh_config=PMHConfig(weight=0.3, cap_ratio=0.3),
        wrong_rank=32,
        sklearn_benchmark={
            "paper_protocol": True,
            "n_train_src": 1500,
            "n_target_pool": 200,
            "n_test": 250,
            "n_pairs_per_class": 40,
            "rank": 32,
            "include_coral": True,
        },
        arms=("b0", "matched", "wrong_w", "isotropic", "coral"),
        application_mode="projection",
        paper_paths="Paper2/T1/classical_pmh/office31_pmh.py",
        notes=(
            "Sklearn isotropic arm = D4 domain Gram (unmatched), not D2.",
            "Paper Office-31 loop does not train wrong-W; library table adds Lemma C arms.",
            "Expect CORAL ≥ matched PMH on linear head (D1 eigengap ~1.05).",
        ),
    ),
    "t1_synthetic_sklearn": BlockPreset(
        block_id="t1_synthetic_sklearn",
        paper_type="T1",
        lemma="D1",
        nuisance="subspace",
        description="Synthetic subspace shift for quick sklearn falsification.",
        sigma_method="D1",
        default_rank=16,
        sklearn_benchmark={"paper_protocol": False, "rank": 16},
        arms=STANDARD_ARMS,
        application_mode="projection",
    ),
    "t2a_vit_isotropic": BlockPreset(
        block_id="t2a_vit_isotropic",
        paper_type="T2A",
        lemma="D2",
        nuisance="isotropic",
        description="ImageNet ViT CLS, isotropic input noise (σ=0.10); no wrong-W arm in paper.",
        sigma_method="D2",
        default_rank=0,
        estimate_kwargs={"noise_level": 0.10},
        pmh_config=PMHConfig(weight=0.3, cap_ratio=0.3, warmup_epochs=0),
        arms=("b0", "matched"),
        application_mode="jacobian",
        paper_paths="Paper2/T2/Task2A/",
        notes=("D2: matched PMH ≡ isotropic penalty on h; wrong-W not used.",),
    ),
    "t2b_chexpert_isotropic": BlockPreset(
        block_id="t2b_chexpert_isotropic",
        paper_type="T2B",
        lemma="D2",
        nuisance="isotropic",
        description="Chest X-ray embeddings, σ=0.08, warmup 5.",
        sigma_method="D2",
        default_rank=0,
        estimate_kwargs={"noise_level": 0.08},
        pmh_config=PMHConfig(weight=0.5, cap_ratio=0.5, warmup_epochs=5),
        arms=("b0", "matched"),
        application_mode="jacobian",
        paper_paths="Paper2/T2/Task2B/train.py",
    ),
    "t3b_depth_d3": BlockPreset(
        block_id="t3b_depth_d3",
        paper_type="T3B",
        lemma="D3",
        nuisance="augmentation",
        description="NYU depth photometric aug-delta Gram, rank 32.",
        sigma_method="D3",
        default_rank=32,
        pmh_config=PMHConfig(weight=0.3, cap_ratio=0.3),
        wrong_rank=32,
        arms=STANDARD_ARMS,
        application_mode="jacobian",
        paper_paths="Paper2/T3/Task3B/calibrate_subspace.py",
        notes=("Task3A uses gradient-SVD in paper code — use pmh.calibrate.gradient_subspace.",),
    ),
    "t4_domain_d4": BlockPreset(
        block_id="t4_domain_d4",
        paper_type="T4",
        lemma="D4",
        nuisance="domain_shift",
        description="Domain Gram on hook features; multilayer in paper uses per-layer rank 64.",
        sigma_method="D4",
        default_rank=64,
        pmh_config=PMHConfig(weight=0.5, cap_ratio=0.3, warmup_epochs=2),
        wrong_rank=64,
        pytorch_benchmark={"epochs": 15},
        arms=("b0", "matched", "wrong_w", "isotropic"),
        application_mode="jacobian",
        paper_paths="Paper2/T4/Task4A/, Paper2/T4/Task4B/",
        notes=("Paper E1 iso-pixel arm is separate from trace_iso training control.",),
    ),
    "t5_compositional_d5": BlockPreset(
        block_id="t5_compositional_d5",
        paper_type="T5",
        lemma="D5",
        nuisance="compositional",
        description="Block-coordinate nuisance (QM9 / code tokens); set nuisance_indices per task.",
        sigma_method="D5",
        default_rank=16,
        estimate_kwargs={"nuisance_indices": (0, 1, 2)},
        pmh_config=PMHConfig(weight=0.5, cap_ratio=0.3, warmup_epochs=2),
        wrong_rank=16,
        application_mode="jacobian",
        paper_paths="Paper2/T5/",
        notes=("Replace nuisance_indices with your coordinate map.",),
    ),
    "t6_temporal_d6": BlockPreset(
        block_id="t6_temporal_d6",
        paper_type="T6",
        lemma="D6",
        nuisance="temporal",
        description="Temporal / sequence scatter (HAR aug-deltas); Whisper uses content-residual calibrator.",
        sigma_method="D6",
        default_rank=48,
        pmh_config=PMHConfig(weight=0.03, cap_ratio=0.3),
        wrong_rank=48,
        application_mode="jacobian",
        paper_paths="Paper2/T6/task6B/, Paper2/T6/task6A/",
        notes=(
            "6A paper code: content-residual W — see pmh.calibrate.content_residual_subspace.",
        ),
    ),
    "t7a_style_d7": BlockPreset(
        block_id="t7a_style_d7",
        paper_type="T7A",
        lemma="D7",
        nuisance="style",
        description="LLM style-pair hidden-state Gram, rank 128, shrinkage 0.1 in eval.",
        sigma_method="D7",
        default_rank=128,
        estimate_kwargs={"shrinkage": 0.1},
        pmh_config=PMHConfig(weight=0.7, cap_ratio=0.3, warmup_epochs=5),
        wrong_rank=128,
        arms=STANDARD_ARMS,
        application_mode="jacobian",
        paper_paths="Paper2/T7/task7A/",
        notes=("Wrong arm in paper uses content/semantic Σ, not random QR.",),
    ),
    "t7b_pgd_d7": BlockPreset(
        block_id="t7b_pgd_d7",
        paper_type="T7B",
        lemma="D7",
        nuisance="style",
        description="PGD-δ subspace on normalized inputs (CIFAR ViT), rank 16.",
        sigma_method="D7",
        default_rank=16,
        pmh_config=PMHConfig(weight=0.5, cap_ratio=0.3),
        wrong_rank=16,
        application_mode="jacobian",
        paper_paths="Paper2/T7/task7B/collect_pgd_subspace.py",
        notes=("Collect W with pmh.calibrate.subspace_from_stacked_deltas on PGD rows.",),
    ),
}


# Default paper-backed preset per nuisance subtype (developer-facing names).
SUBTYPE_TO_BLOCK_PRESET: dict[str, str] = {
    "D1": "t1_office31_sklearn",
    "D2": "t2a_vit_isotropic",
    "D3": "t3b_depth_d3",
    "D4": "t4_domain_d4",
    "D5": "t5_compositional_d5",
    "D6": "t6_temporal_d6",
    "D7": "t7a_style_d7",
}


def get_subtype_preset(subtype: str) -> BlockPreset:
    """Map ``D1``–``D7`` (or ``d4``) to the recommended block preset."""
    key = subtype.strip().upper()
    if not key.startswith("D"):
        key = f"D{key}"
    pid = SUBTYPE_TO_BLOCK_PRESET.get(key)
    if pid is None:
        raise KeyError(
            f"no subtype preset for {subtype!r}; choose from {list(SUBTYPE_TO_BLOCK_PRESET)}"
        )
    return get_preset(pid)


def get_preset(block_id: str) -> BlockPreset:
    key = block_id.strip().lower().replace("-", "_")
    if key not in PRESETS:
        raise KeyError(f"unknown preset {block_id!r}; choose from {sorted(PRESETS)}")
    return PRESETS[key]


def list_presets() -> list[str]:
    return sorted(PRESETS.keys())
