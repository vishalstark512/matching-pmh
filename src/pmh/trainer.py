"""PMHTrainer: Phase A estimate + Phase B train in one object."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from pmh.artifact import SigmaTaskEstimate
from pmh.config import PMHConfig, SigmaTaskConfig
from pmh.data_context import DataContext
from pmh.estimate import estimate_from_config
from pmh.features import (
    AugFn,
    collect_augmentation_deltas,
    collect_features,
    collect_labeled_features,
    collect_sequence_features,
)
from pmh.hooks import resolve_hook
from pmh.integrations.torch import PMHCallback, train_epoch_with_pmh
from pmh.multi import MultiPMHLoss
from pmh.nuisance import config_from_nuisance, resolve_method
from pmh.numpy_api import estimate_sigma_task_numpy
from pmh.suggest import resolve_nuisance_arg
from pmh.training import PMHLoss

Encoder = Callable[[torch.Tensor], torch.Tensor]
Head = Callable[[torch.Tensor], torch.Tensor]


class PMHTrainer:
    """Estimate ``Sigma_task`` once, then train with matched PMH on hook ``h``.

    Supports D1–D7 on PyTorch batches (see ``estimate()`` kwargs per method).
    For hybrid nuisances pass ``artifacts=`` or call :meth:`add_artifact` and use
  ``multi_loss=True``.

    Set ``train_mode="feature_diff"`` with ``forward_features`` + ``layer_names`` for
    paper T4B-style per-layer Gram + feature-diff PMH (see :meth:`estimate_multilayer`).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        hook: str | nn.Module | Encoder | None = "backbone",
        head: Head | nn.Module | None = None,
        nuisance: str = "domain_shift",
        rank: int | None = None,
        shrinkage: float = 1e-6,
        pmh_config: PMHConfig | None = None,
        artifact_path: str | Path | None = None,
        artifacts: Sequence[SigmaTaskEstimate | str | Path] | None = None,
        device: torch.device | str | None = None,
        pool_spatial: bool = True,
        nuisance_indices: list[int] | None = None,
        noise_level: float = 0.1,
        data_context: DataContext | None = None,
        has_source_labels: bool = True,
        has_target_labels: bool = False,
        has_target_domain: bool = True,
        has_augmentation_modes: bool = False,
        has_style_pairs: bool = False,
        train_mode: Literal["jacobian", "feature_diff"] = "jacobian",
        forward_features: Callable[[torch.Tensor], dict[str, torch.Tensor]] | None = None,
        layer_names: Sequence[str] | None = None,
        head_layer: str | None = None,
        noise_std: float = 0.05,
        noise_rank: int = 64,
    ) -> None:
        self.model = model
        ctx = data_context or DataContext(
            has_source_labels=has_source_labels,
            has_target_labels=has_target_labels,
            has_target_domain=has_target_domain,
            has_augmentation_modes=has_augmentation_modes,
            has_style_pairs=has_style_pairs,
            has_nuisance_indices=nuisance_indices is not None,
        )
        auto_kw = ctx.to_auto_kwargs()
        self.nuisance = resolve_nuisance_arg(nuisance, **auto_kw)
        self.rank = rank
        self.shrinkage = shrinkage
        self.pmh_config = pmh_config or PMHConfig.balanced()
        self.artifact_path = Path(artifact_path) if artifact_path else None
        self.nuisance_indices = nuisance_indices
        self.noise_level = noise_level
        self.device = torch.device(device) if device is not None else None

        self.encoder: Encoder = resolve_hook(model, hook, pool_spatial=pool_spatial)
        if head is None:
            self.head: Head | None = None
        elif isinstance(head, nn.Module):
            mod = head

            def _head(h: torch.Tensor) -> torch.Tensor:
                return mod(h)

            self.head = _head
        else:
            self.head = head

        self.train_mode = train_mode
        self.forward_features = forward_features
        self.layer_names: tuple[str, ...] = tuple(layer_names or ())
        self.head_layer = head_layer
        self.noise_std = noise_std
        self.noise_rank = noise_rank
        self.layer_sigmas_: dict[str, torch.Tensor] | None = None
        self._feature_diff_callback: Any = None

        self.artifact_: SigmaTaskEstimate | None = None
        self._extra_artifacts: list[SigmaTaskEstimate] = []
        self.pmh_loss_: PMHLoss | MultiPMHLoss | None = None
        self._callback: PMHCallback | None = None

        if train_mode == "feature_diff" and (not forward_features or not layer_names):
            raise ValueError(
                "train_mode='feature_diff' requires forward_features= and layer_names="
            )

        if artifacts:
            for a in artifacts:
                self.add_artifact(a)

    @classmethod
    def from_artifact(
        cls,
        model: nn.Module,
        artifact: SigmaTaskEstimate | str | Path,
        *,
        hook: str | nn.Module | Encoder | None = "backbone",
        head: Head | nn.Module | None = None,
        pmh_config: PMHConfig | None = None,
        artifact_path: str | Path | None = None,
        device: torch.device | str | None = None,
        pool_spatial: bool = True,
    ) -> PMHTrainer:
        """Train with a pre-estimated Σ̂ (your deltas, calibrator, or saved ``.pt``)."""
        if isinstance(artifact, (str, Path)):
            loaded = SigmaTaskEstimate.load(artifact)
            path = Path(artifact)
        else:
            loaded = artifact
            path = artifact_path
        _method_to_nuisance = {
            "D1": "subspace",
            "D2": "isotropic",
            "D3": "augmentation",
            "D4": "domain_shift",
            "D5": "compositional",
            "D6": "temporal",
            "D7": "style",
        }
        nuisance = _method_to_nuisance.get(loaded.method, "domain_shift")
        trainer = cls(
            model,
            hook=hook,
            head=head,
            nuisance=nuisance,
            rank=loaded.config.rank,
            shrinkage=loaded.config.shrinkage,
            pmh_config=pmh_config,
            artifact_path=path,
            device=device,
            pool_spatial=pool_spatial,
            nuisance_indices=loaded.config.nuisance_indices,
            noise_level=loaded.config.noise_level,
        )
        trainer.artifact_ = loaded
        trainer._bind_pmh_loss()
        return trainer

    @property
    def method(self) -> str:
        return resolve_method(self.nuisance)

    def _sigma_config(self, dim: int, n_samples: int) -> SigmaTaskConfig:
        r = self.rank
        if r is None and self.method in ("D1", "D4", "D7"):
            r = min(32, max(1, dim // 4))
        return config_from_nuisance(
            self.nuisance,
            rank=r,
            shrinkage=self.shrinkage,
            dim=dim,
            noise_level=self.noise_level,
            nuisance_indices=self.nuisance_indices,
        )

    def add_artifact(self, artifact: SigmaTaskEstimate | str | Path) -> None:
        if isinstance(artifact, (str, Path)):
            artifact = SigmaTaskEstimate.load(artifact)
        if self.artifact_ is None:
            self.artifact_ = artifact
        else:
            self._extra_artifacts.append(artifact)

    def _all_artifacts(self) -> list[SigmaTaskEstimate]:
        out: list[SigmaTaskEstimate] = []
        if self.artifact_ is not None:
            out.append(self.artifact_)
        out.extend(self._extra_artifacts)
        return out

    @torch.no_grad()
    def estimate(
        self,
        source_batches: Iterable[Any] | None = None,
        target_batches: Iterable[Any] | None = None,
        *,
        max_batches: int = 50,
        save: bool = True,
        aug_deltas: torch.Tensor | None = None,
        augmentations: Sequence[AugFn] | None = None,
        sequences_batches: Iterable[Any] | None = None,
        style_jsonl: str | Path | None = None,
        hf_model: Any = None,
        hf_tokenizer: Any = None,
        d6_source: str = "content",
    ) -> SigmaTaskEstimate:
        """Phase A: estimate ``Sigma_task`` (D1–D7).

        Extra kwargs by method
        ----------------------
        D3 : ``aug_deltas`` ``[K,d]`` or ``[K,N,d]``, **or** ``augmentations`` + ``source_batches``
        D6 : ``sequences_batches`` (encoder returns ``[B,T,d]``)
        D7 : ``style_jsonl`` + ``hf_model`` / ``hf_tokenizer`` (Transformers)
        D5 : set ``nuisance_indices=`` on trainer; ``source_batches`` only
        D1 : labeled ``(x,y)`` in source and target loaders
        D4 : ``source_batches`` + ``target_batches`` (class-aligned Gram when ``(x,y)`` batches)
        D6 : ``d6_source='content'`` (default, paper 6A) or ``'temporal'`` for consecutive diffs
        D2 : ``source_batches`` only (dim from ``h``)
        """
        if self.train_mode == "feature_diff":
            raise ValueError(
                "train_mode='feature_diff': call estimate_multilayer() instead of estimate()"
            )

        if self.artifact_path and self.artifact_path.exists() and source_batches is None:
            self.artifact_ = SigmaTaskEstimate.load(self.artifact_path)
            self._bind_pmh_loss()
            return self.artifact_

        method = self.method
        self.model.eval()
        dev = self.device

        if method == "D7":
            self.artifact_ = self._estimate_d7(
                style_jsonl=style_jsonl,
                hf_model=hf_model,
                hf_tokenizer=hf_tokenizer,
            )
        elif method == "D3":
            self.artifact_ = self._estimate_d3(
                source_batches,
                aug_deltas=aug_deltas,
                augmentations=augmentations,
                max_batches=max_batches,
            )
        elif method == "D6":
            if sequences_batches is None:
                raise ValueError("temporal (D6) requires sequences_batches=")
            seq = collect_sequence_features(
                self.encoder, sequences_batches, max_batches=max_batches, device=dev
            )
            cfg = self._sigma_config(seq.shape[-1], seq.shape[0])
            if d6_source == "content":
                from pmh.calibrate.content_residual import content_residual_subspace

                _, self.artifact_ = content_residual_subspace(
                    seq.numpy(),
                    rank=int(cfg.rank or 32),
                    source="content",
                )
                self.artifact_.config = cfg
            else:
                self.artifact_ = estimate_from_config(cfg, seq)
        elif method == "D5":
            if self.nuisance_indices is None:
                raise ValueError("compositional (D5) requires nuisance_indices=")
            if source_batches is None:
                raise ValueError("D5 requires source_batches=")
            h = collect_features(self.encoder, source_batches, max_batches=max_batches, device=dev)
            cfg = self._sigma_config(h.shape[1], h.shape[0])
            self.artifact_ = estimate_from_config(cfg, h)
        elif method == "D2":
            if source_batches is None:
                raise ValueError("D2 requires source_batches=")
            h = collect_features(self.encoder, source_batches, max_batches=max_batches, device=dev)
            cfg = config_from_nuisance(
                "isotropic", dim=h.shape[1], shrinkage=self.shrinkage, noise_level=self.noise_level
            )
            self.artifact_ = estimate_from_config(cfg)
        elif method == "D1":
            if source_batches is None or target_batches is None:
                raise ValueError("subspace (D1) requires labeled source_batches and target_batches")
            h_s, y_s = collect_labeled_features(
                self.encoder, source_batches, max_batches=max_batches, device=dev
            )
            h_t, y_t = collect_labeled_features(
                self.encoder, target_batches, max_batches=max_batches, device=dev
            )
            cfg = self._sigma_config(h_s.shape[1], h_s.shape[0] + h_t.shape[0])
            self.artifact_ = estimate_sigma_task_numpy(
                h_s.numpy(),
                y_s.numpy().astype("int64"),
                h_t.numpy(),
                y_t.numpy().astype("int64"),
                config=cfg,
            )
        elif method == "D4":
            if source_batches is None or target_batches is None:
                raise ValueError("domain_shift (D4) requires source_batches and target_batches")
            from pmh.estimators.d4_domain import estimate_d4_from_paired_diffs
            from pmh.features import collect_domain_paired_diffs

            diff, aligned = collect_domain_paired_diffs(
                self.encoder,
                source_batches,
                target_batches,
                align_by_class=True,
                max_batches=max_batches,
                device=dev,
            )
            cfg = self._sigma_config(diff.shape[1], diff.shape[0])
            sigma = estimate_d4_from_paired_diffs(
                diff, rank=cfg.rank, shrinkage=cfg.shrinkage
            )
            meta = {"d4_class_aligned": aligned}
            cov = sigma
            preflight = None
            eigengap = None
            if cfg.rank is not None:
                from pmh.preflight import preflight_eigengap

                status, eigengap = preflight_eigengap(cov, cfg.rank)
                preflight = status.value
            self.artifact_ = SigmaTaskEstimate(
                sigma=sigma,
                method="D4",
                config=cfg,
                eigengap=eigengap,
                preflight=preflight,
                metadata=meta,
            )
        else:
            raise ValueError(f"Unknown method {method}")

        if method != "D7" and source_batches is not None and method not in ("D2",):
            h_probe = collect_features(
                self.encoder, source_batches, max_batches=1, device=dev
            )
            self._validate_artifact_dim(h_probe)

        self._bind_pmh_loss()
        if save and self.artifact_path:
            self.artifact_.save(self.artifact_path)
        return self.artifact_

    def _estimate_d3(
        self,
        source_batches: Iterable[Any] | None,
        *,
        aug_deltas: torch.Tensor | None,
        augmentations: Sequence[AugFn] | None,
        max_batches: int,
    ) -> SigmaTaskEstimate:
        if aug_deltas is None:
            if source_batches is None or not augmentations:
                raise ValueError("D3: pass aug_deltas= or source_batches= + augmentations=")
            aug_deltas = collect_augmentation_deltas(
                self.encoder,
                source_batches,
                augmentations,
                max_batches=max_batches,
                device=self.device,
            )
        cfg = SigmaTaskConfig.for_augmentation(shrinkage=self.shrinkage)
        return estimate_from_config(cfg, aug_deltas=aug_deltas)

    def _estimate_d7(
        self,
        *,
        style_jsonl: str | Path | None,
        hf_model: Any,
        hf_tokenizer: Any,
    ) -> SigmaTaskEstimate:
        if style_jsonl is None:
            raise ValueError("D7: pass style_jsonl= with hf_model= and hf_tokenizer=")
        from pmh.integrations.huggingface import estimate_style_sigma, load_style_pairs_jsonl

        pairs = load_style_pairs_jsonl(style_jsonl)
        if hf_model is None or hf_tokenizer is None:
            raise ValueError("D7: pass hf_model= and hf_tokenizer= (pip install matching-pmh[hf])")
        cfg = SigmaTaskConfig.for_alignment(rank=self.rank or 32, shrinkage=self.shrinkage)
        return estimate_style_sigma(pairs, hf_model, hf_tokenizer, config=cfg)

    def _validate_artifact_dim(self, h_sample: torch.Tensor) -> None:
        if self.artifact_ is None:
            return
        d = h_sample.shape[-1]
        sig_d = self.artifact_.sigma.shape[0]
        if sig_d != d:
            raise ValueError(
                f"Artifact dim {sig_d} != hook dim {d}. Re-estimate with the same hook layer."
            )

    @torch.no_grad()
    def estimate_multilayer(
        self,
        source_batches: Iterable[Any] | None,
        target_batches: Iterable[Any] | None,
        *,
        forward_features: Callable[[torch.Tensor], dict[str, torch.Tensor]] | None = None,
        layer_names: Sequence[str] | None = None,
        max_batches: int = 50,
        save: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Class-aligned D4 Gram per layer (paper E1_multiscale estimate phase)."""
        from pmh.vision.domain_multilayer import estimate_multilayer_domain_sigmas

        ff = forward_features or self.forward_features
        layers = tuple(layer_names or self.layer_names)
        if ff is None or not layers:
            raise ValueError("estimate_multilayer requires forward_features= and layer_names=")
        if source_batches is None or target_batches is None:
            raise ValueError("estimate_multilayer requires source_batches and target_batches")
        self.model.eval()
        self.layer_sigmas_ = estimate_multilayer_domain_sigmas(
            ff,
            source_batches,
            target_batches,
            layers,
            rank=int(self.rank or 32),
            max_batches=max_batches,
            device=self.device,
        )
        primary = layers[0]
        d = int(self.layer_sigmas_[primary].shape[0])
        cfg = self._sigma_config(d, max_batches * 16)
        self.artifact_ = SigmaTaskEstimate(
            sigma=self.layer_sigmas_[primary],
            method="D4",
            config=cfg,
            metadata={
                "multilayer": True,
                "layers": list(layers),
                "d4_class_aligned": True,
            },
        )
        if self.train_mode == "jacobian":
            self._bind_pmh_loss()
        else:
            self._bind_feature_diff()
        if save and self.artifact_path:
            self.artifact_.save(self.artifact_path)
        return self.layer_sigmas_

    def _bind_feature_diff(self) -> None:
        from pmh.vision.domain_multilayer import FeatureDiffCallback, build_multilayer_domain_trainer

        if self.forward_features is None or not self.layer_names:
            raise ValueError("feature_diff mode requires forward_features and layer_names")
        if not self.layer_sigmas_:
            raise RuntimeError("Call estimate_multilayer() first")
        ml_loss, noisy_forward = build_multilayer_domain_trainer(
            self.model,
            self.forward_features,
            self.layer_sigmas_,
            self.layer_names,
            pmh_config=self.pmh_config,
            noise_std=self.noise_std,
            noise_rank=self.noise_rank,
        )
        self._feature_diff_callback = FeatureDiffCallback(
            self.model,
            self.forward_features,
            noisy_forward,
            ml_loss,
            head=self.head,
            head_layer=self.head_layer,
            layer_names=self.layer_names,
        )

    def _bind_pmh_loss(self) -> None:
        if self.train_mode == "feature_diff":
            self._bind_feature_diff()
            return
        arts = self._all_artifacts()
        if not arts:
            raise RuntimeError("Call estimate() or load artifact first.")
        if len(arts) == 1:
            self.pmh_loss_ = PMHLoss(arts[0], self.pmh_config)
        else:
            self.pmh_loss_ = MultiPMHLoss(arts, self.pmh_config)
        assert self.pmh_loss_ is not None
        self._callback = PMHCallback(
            self.pmh_loss_,  # type: ignore[arg-type]
            self.encoder,
            head=self.head,
        )

    def load_artifact(self, path: str | Path) -> SigmaTaskEstimate:
        self.artifact_ = SigmaTaskEstimate.load(path)
        self._bind_pmh_loss()
        return self.artifact_

    @property
    def callback(self) -> PMHCallback:
        if self._callback is None:
            self._bind_pmh_loss()
        assert self._callback is not None
        return self._callback

    def fit(
        self,
        train_loader: Iterable[Any],
        *,
        source_batches: Iterable[Any] | None = None,
        target_batches: Iterable[Any] | None = None,
        sequences_batches: Iterable[Any] | None = None,
        val_loader: Iterable[Any] | None = None,
        epochs: int = 10,
        optimizer: torch.optim.Optimizer | None = None,
        max_batches_estimate: int = 50,
        max_steps_per_epoch: int | None = None,
        reestimate: bool = False,
        estimate_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Phase A (if needed) + Phase B training loop."""
        if self.train_mode == "feature_diff":
            return self._fit_feature_diff(
                train_loader,
                source_batches=source_batches,
                target_batches=target_batches,
                epochs=epochs,
                optimizer=optimizer,
                max_batches_estimate=max_batches_estimate,
                max_steps_per_epoch=max_steps_per_epoch,
                reestimate=reestimate,
                estimate_kwargs=estimate_kwargs,
            )

        est_kw = estimate_kwargs or {}
        if reestimate or self.artifact_ is None:
            if source_batches is None and sequences_batches is None and not est_kw.get("style_jsonl"):
                if self.artifact_path and self.artifact_path.exists() and not reestimate:
                    self.load_artifact(self.artifact_path)
                elif not self._all_artifacts():
                    raise ValueError(
                        "Pass source_batches=, sequences_batches=, style_jsonl=, or artifact_path="
                    )
            else:
                if reestimate and self.artifact_path and self.artifact_path.exists():
                    self.artifact_path.unlink(missing_ok=True)
                self.estimate(
                    source_batches,
                    target_batches,
                    sequences_batches=sequences_batches,
                    max_batches=max_batches_estimate,
                    **est_kw,
                )

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        last: dict[str, float] = {}
        for epoch in range(1, epochs + 1):
            last = train_epoch_with_pmh(
                self.model,
                self.callback,
                train_loader,
                optimizer,
                epoch=epoch,
                device=self.device,
                max_steps=max_steps_per_epoch,
            )
        return last

    def _fit_feature_diff(
        self,
        train_loader: Iterable[Any],
        *,
        source_batches: Iterable[Any] | None,
        target_batches: Iterable[Any] | None,
        epochs: int,
        optimizer: torch.optim.Optimizer | None,
        max_batches_estimate: int,
        max_steps_per_epoch: int | None,
        reestimate: bool,
        estimate_kwargs: dict[str, Any] | None,
    ) -> dict[str, float]:
        from pmh.vision.domain_multilayer import train_epoch_feature_diff

        est_kw = estimate_kwargs or {}
        if reestimate or not self.layer_sigmas_:
            if source_batches is None or target_batches is None:
                raise ValueError(
                    "feature_diff fit requires source_batches= and target_batches= "
                    "(or pre-call estimate_multilayer)"
                )
            self.estimate_multilayer(
                source_batches,
                target_batches,
                max_batches=max_batches_estimate,
                **{k: v for k, v in est_kw.items() if k in ("forward_features", "layer_names")},
            )
        elif self._feature_diff_callback is None:
            self._bind_feature_diff()

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        last: dict[str, float] = {}
        assert self._feature_diff_callback is not None
        for epoch in range(1, epochs + 1):
            last = train_epoch_feature_diff(
                self._feature_diff_callback,
                train_loader,
                optimizer,
                epoch=epoch,
                device=self.device,
                max_steps=max_steps_per_epoch,
            )
        return last

    def training_step(self, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
        if self.train_mode == "feature_diff":
            if self._feature_diff_callback is None:
                self._bind_feature_diff()
            loss, step = self._feature_diff_callback.training_step(batch)
            return loss, {
                "task_loss": step.task_loss,
                "pmh_loss": step.pmh_loss,
                "total_loss": step.total_loss,
            }
        loss, step = self.callback.training_step(batch)
        return loss, {
            "task_loss": step.task_loss,
            "pmh_loss": step.pmh_loss,
            "total_loss": step.total_loss,
        }

    @torch.no_grad()
    def measure_trajectory_tdi(
        self,
        probe_batches: Iterable[Any],
        *,
        sigma: float = 0.01,
        max_batches: int = 20,
        seed: int = 0,
    ) -> dict[str, float | list[float] | int]:
        """Label-free trajectory TDI on hook representations (isotropic input noise).

        Uses the same ``encoder`` / hook as training. Paper default: ``sigma=0.01``.
        For per-layer ViT probes, use :func:`pmh.tdi.trajectory_tdi_layerwise` on
        stacked layer features from your own hooks.

        Parameters
        ----------
        probe_batches
            Typically target-domain validation batches ``(x, y)`` or ``x`` only.
        sigma
            Gaussian perturbation scale on inputs.
        max_batches
            Cap for speed.

        Returns
        -------
        dict
            ``trajectory_tdi``, ``tdi_per_layer``, ``sigma``, ``n_samples``.
        """
        from pmh.tdi import trajectory_tdi_encoder

        self.model.eval()
        return trajectory_tdi_encoder(
            self.model,
            self.encoder,
            probe_batches,
            sigma=sigma,
            max_batches=max_batches,
            device=self.device,
            seed=seed,
        )


def build_hybrid_trainer(
    model: nn.Module,
    estimates: Sequence[SigmaTaskEstimate],
    **trainer_kw: Any,
) -> PMHTrainer:
    """Trainer with multiple pre-estimated artifacts (additive PMH)."""
    t = PMHTrainer(model, **trainer_kw)
    t.artifact_ = estimates[0]
    for a in estimates[1:]:
        t.add_artifact(a)
    t._bind_pmh_loss()
    return t
