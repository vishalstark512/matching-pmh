"""High-level developer API (no paper vocabulary)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

import numpy as np

from pmh.config import PMHConfig
from pmh.hooks import HOOK_REGISTRY, detect_model_family, resolve_hook
from pmh.onboarding import preflight_plain_english
from pmh.suggest import suggest_nuisance

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from pmh.trainer import PMHTrainer

Verdict = Literal["go", "marginal", "no_go"]


@dataclass
class HookSuggestion:
    """Suggested representation hook for a PyTorch model."""

    family: str
    hook: str | Any
    path: str
    repr_dim: int | None = None
    note: str = ""


@dataclass
class ApplicabilityReport:
    """Whether PMH is appropriate for your setup."""

    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    suggested_nuisance: str = "domain_shift"
    suggested_rank: int | None = 16
    can_proceed: bool = True

    def summary(self) -> str:
        lines = [f"Verdict: {self.verdict.upper()}", *self.reasons]
        if self.suggested_nuisance:
            lines.append(f"Suggested nuisance={self.suggested_nuisance!r}")
        return "\n".join(lines)


@dataclass
class DomainPair:
    """Source vs target domain data contract."""

    n_source: int | None = None
    n_target: int | None = None
    feature_dim: int | None = None
    has_target_labels: bool = False
    has_source_labels: bool = True

    @classmethod
    def from_arrays(
        cls,
        x_source: np.ndarray,
        x_target: np.ndarray,
        y_source: np.ndarray | None = None,
        y_target: np.ndarray | None = None,
    ) -> DomainPair:
        xs = np.asarray(x_source)
        xt = np.asarray(x_target)
        if xs.ndim != 2 or xt.ndim != 2:
            raise ValueError("x_source and x_target must be 2D [N, d]")
        if xs.shape[1] != xt.shape[1]:
            raise ValueError(
                f"feature dim mismatch: source d={xs.shape[1]}, target d={xt.shape[1]}"
            )
        pair = cls(
            n_source=int(xs.shape[0]),
            n_target=int(xt.shape[0]),
            feature_dim=int(xs.shape[1]),
            has_target_labels=y_target is not None,
            has_source_labels=y_source is not None,
        )
        pair.validate()
        return pair

    def validate(self) -> None:
        if self.n_source is not None and self.n_source < 8:
            raise ValueError("need at least ~8 source samples for a stable estimate")
        if self.n_target is not None and self.n_target < 8:
            raise ValueError("need at least ~8 target samples for a stable estimate")
        if self.feature_dim is not None and self.feature_dim < 2:
            raise ValueError("feature_dim must be >= 2")


@dataclass
class RobustFitResult:
    """Output of :func:`robust_fit`."""

    trainer: PMHTrainer
    stats: dict[str, float]
    applicability: ApplicabilityReport
    hook_used: str | Any
    preflight: str | None = None

    @property
    def preflight_message(self) -> str:
        return preflight_plain_english(self.preflight)

    def summary(self) -> str:
        return (
            f"train task_loss={self.stats.get('task_loss', 0):.4f}  "
            f"pmh_loss={self.stats.get('pmh_loss', 0):.4f}  "
            f"preflight={self.preflight} ({self.preflight_message})"
        )


@dataclass
class EvaluationReport:
    """Baseline vs PMH on a target holdout (developer-friendly)."""

    baseline_metric: float
    pmh_metric: float
    metric_name: str = "accuracy"
    preflight: str | None = None
    preflight_message: str = ""
    compare_baselines: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Target {self.metric_name}: baseline={self.baseline_metric:.3f}  pmh={self.pmh_metric:.3f}",
        ]
        if self.preflight:
            lines.append(f"Preflight: {self.preflight} - {self.preflight_message}")
        for k, v in self.compare_baselines.items():
            lines.append(f"  {k}: {v:.3f}")
        for n in self.notes:
            lines.append(f"Note: {n}")
        return "\n".join(lines)


def check_applicability(
    *,
    stack: Literal["pytorch", "sklearn", "hf"] = "pytorch",
    n_source: int | None = None,
    n_target: int | None = None,
    feature_dim: int | None = None,
    has_target_domain: bool = True,
    has_target_labels: bool = False,
    has_style_pairs: bool = False,
    new_classes_at_deploy: bool = False,
    min_samples: int = 16,
) -> ApplicabilityReport:
    """Plain-language go / no-go before you train."""
    reasons: list[str] = []
    verdict: Verdict = "go"

    if new_classes_at_deploy:
        return ApplicabilityReport(
            verdict="no_go",
            reasons=[
                "New classes at deployment are label shift, not PMH domain shift.",
                "Use methods for open-set / label shift instead.",
            ],
            can_proceed=False,
        )

    if not has_target_domain and not has_style_pairs:
        return ApplicabilityReport(
            verdict="no_go",
            reasons=["Need deployment/target data (or style pairs for LLM format shift)."],
            can_proceed=False,
        )

    if n_source is not None and n_source < min_samples:
        reasons.append(f"Few source samples ({n_source}); aim for {min_samples}+.")
        verdict = "marginal"
    if n_target is not None and n_target < min_samples:
        reasons.append(f"Few target samples ({n_target}); aim for {min_samples}+.")
        verdict = "marginal"

    if feature_dim is not None and feature_dim < 4:
        reasons.append(f"Very low feature dim ({feature_dim}); try rank <= dim/2.")
        verdict = "marginal"

    sug = suggest_nuisance(
        has_source_labels=True,
        has_target_labels=has_target_labels,
        has_target_domain=has_target_domain,
        has_style_pairs=has_style_pairs,
    )
    rank = 16
    if feature_dim is not None:
        rank = max(4, min(32, feature_dim // 2))

    if not reasons:
        reasons.append("Setup matches train-on-A / deploy-on-B with same labels.")

    return ApplicabilityReport(
        verdict=verdict,
        reasons=reasons,
        suggested_nuisance=sug.nuisance,
        suggested_rank=rank,
        can_proceed=verdict != "no_go",
    )


def suggest_hook(
    model: Any,
    *,
    probe_input: Any = None,
    alias: str = "backbone",
) -> HookSuggestion:
    """Pick a default hook layer for ``model`` (best-effort)."""
    import torch

    family = detect_model_family(model)
    registry = HOOK_REGISTRY.get(family, HOOK_REGISTRY["generic"])
    path = registry.get(alias, registry.get("default", ""))
    path_disp = path if path else "full model"
    note = f"Detected family={family!r}; using hook alias {alias!r} -> {path_disp!r}."
    dim: int | None = None
    hook_spec: str | Any = alias if alias in registry else (path or alias)

    if probe_input is not None:
        for candidate in (hook_spec, path, None):
            try:
                enc = resolve_hook(model, candidate)
                with torch.no_grad():
                    h = enc(probe_input)
                dim = int(h.shape[-1])
                hook_spec = candidate if candidate is not None else hook_spec
                note += f" Probe output shape {tuple(h.shape)}."
                break
            except (ValueError, AttributeError, TypeError):
                continue
        else:
            note += " Could not probe hook; pass hook= explicitly."
    else:
        note += " Pass probe_input= for a dry-run shape check."
    return HookSuggestion(family=family, hook=hook_spec, path=path, repr_dim=dim, note=note)


def robust_fit(
    model: Any,
    train_loader: Any,
    *,
    source_batches: Any,
    target_batches: Any | None = None,
    hook: str | Any = "auto",
    head: Any = None,
    nuisance: str | None = None,
    rank: int | None = None,
    pmh_config: PMHConfig | None = None,
    epochs: int = 10,
    applicability: ApplicabilityReport | None = None,
    strict_applicability: bool = False,
    artifact_path: str | None = None,
    **trainer_kw: Any,
) -> RobustFitResult:
    """Single entry: check applicability, resolve hook, estimate + train."""
    from pmh.trainer import PMHTrainer

    app = applicability or check_applicability(
        stack="pytorch",
        has_target_domain=target_batches is not None,
    )
    if strict_applicability and not app.can_proceed:
        raise ValueError(app.summary())
    if app.verdict == "marginal" and not strict_applicability:
        import warnings

        warnings.warn(f"PMH applicability marginal:\n{app.summary()}", stacklevel=2)

    hook_used: str | Any = hook
    if hook == "auto":
        sug = suggest_hook(model)
        hook_used = sug.hook

    nui = nuisance or app.suggested_nuisance
    r = rank if rank is not None else app.suggested_rank

    trainer = PMHTrainer(
        model,
        hook=hook_used,
        head=head,
        nuisance=nui,
        rank=r,
        pmh_config=pmh_config or PMHConfig.balanced(),
        artifact_path=artifact_path,
        **trainer_kw,
    )
    stats = trainer.fit(
        train_loader,
        source_batches=source_batches,
        target_batches=target_batches,
        epochs=epochs,
    )
    pf = trainer.artifact_.preflight if trainer.artifact_ else None
    if pf == "fail" and strict_applicability:
        raise ValueError(preflight_plain_english(pf))

    return RobustFitResult(
        trainer=trainer,
        stats=stats,
        applicability=app,
        hook_used=hook_used,
        preflight=pf,
    )


def evaluate_baseline_vs_pmh(
    *,
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    test_size: float = 0.35,
    rank: int = 16,
    seed: int = 0,
    compare_to: tuple[str, ...] = ("coral",),
) -> EvaluationReport:
    """Sklearn path: source train vs PMH adapt + clf; score on target holdout."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError('evaluate_baseline_vs_pmh requires sklearn: pip install "matching-pmh[sklearn]"') from exc
    from pmh import PMHMatcher

    pair = DomainPair.from_arrays(x_source, x_target, y_source, y_target)
    app = check_applicability(
        stack="sklearn",
        n_source=pair.n_source,
        n_target=pair.n_target,
        feature_dim=pair.feature_dim,
        has_target_labels=True,
    )

    x_pool, x_te, y_pool, y_te = train_test_split(
        x_target, y_target, test_size=test_size, random_state=seed, stratify=y_target
    )

    clf0 = LogisticRegression(max_iter=500)
    clf0.fit(x_source, y_source)
    acc_b0 = float(accuracy_score(y_te, clf0.predict(x_te)))

    matcher = PMHMatcher(nuisance="domain_shift", rank=rank, seed=seed)
    matcher.fit(x_source, y_source, x_pool, y_pool)
    x_ad = matcher.transform(x_source)
    clf1 = LogisticRegression(max_iter=500)
    clf1.fit(x_ad, y_source)
    acc_pmh = float(accuracy_score(y_te, clf1.predict(matcher.transform(x_te))))

    pf = matcher.artifact_.preflight
    extras: dict[str, float] = {}
    notes = [app.reasons[0] if app.reasons else ""]
    if compare_to:
        from pmh import compare_arms_sklearn

        res = compare_arms_sklearn(
            x_source,
            y_source,
            x_target,
            y_target,
            rank=rank,
            include_coral="coral" in compare_to,
            seed=seed,
        )
        for arm in compare_to:
            if arm in res.arms and res.arms[arm].val_metric is not None:
                extras[arm] = float(res.arms[arm].val_metric)
        notes.append("For falsification arms see compare_arms_sklearn (advanced).")

    return EvaluationReport(
        baseline_metric=acc_b0,
        pmh_metric=acc_pmh,
        preflight=pf,
        preflight_message=preflight_plain_english(pf),
        compare_baselines=extras,
        notes=[n for n in notes if n],
    )


def _accuracy_on_loader(
    model: Any,
    encoder: Any,
    head: Any | None,
    loader: Any,
    device: Any = None,
) -> float:
    """Labeled-loader accuracy for a PyTorch model + hook (+ optional head)."""
    import torch

    device = device or next(model.parameters()).device
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                xb, yb = batch[0], batch[1]
            elif isinstance(batch, dict):
                xb = batch.get("input_ids", batch.get("x"))
                yb = batch["labels"]
            else:
                raise TypeError("batch must be tuple or dict")
            xb = xb.to(device)
            yb = yb.to(device)
            logits = head(encoder(xb)) if head is not None else model(xb)
            if hasattr(logits, "logits"):
                logits = logits.logits
            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


def evaluate_trainer_on_loader(
    trainer: PMHTrainer,
    encoder: Any,
    head: Any,
    loader: Any,
    device: Any = None,
) -> float:
    """Accuracy of a trained model on a labeled loader."""
    return _accuracy_on_loader(trainer.model, encoder, head, loader, device=device)


def _train_erm_epochs(
    model: Any,
    encoder: Any,
    head: Any | None,
    train_loader: Any,
    *,
    epochs: int,
    lr: float = 1e-3,
    device: Any = None,
) -> None:
    """Task-loss-only training (ERM baseline) on ``train_loader``."""
    import torch
    import torch.nn as nn

    device = device or next(model.parameters()).device
    params = list(model.parameters())
    if head is not None and hasattr(head, "parameters"):
        params += list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    if head is not None and hasattr(head, "train"):
        head.train()
    for _ in range(epochs):
        for batch in train_loader:
            if isinstance(batch, (tuple, list)):
                xb, yb = batch[0], batch[1]
            elif isinstance(batch, dict):
                xb = batch.get("input_ids", batch.get("x"))
                yb = batch["labels"]
            else:
                raise TypeError("batch must be tuple or dict")
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = head(encoder(xb)) if head is not None else model(xb)
            if hasattr(logits, "logits"):
                logits = logits.logits
            crit(logits, yb).backward()
            opt.step()


def evaluate_robust_fit(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    *,
    source_batches: Any,
    target_batches: Any,
    hook: str | Any = "auto",
    head: Any = None,
    epochs: int = 5,
    rank: int | None = None,
    nuisance: str | None = None,
    pmh_result: RobustFitResult | None = None,
    seed: int = 0,
) -> EvaluationReport:
    """PyTorch path: ERM baseline vs PMH on a labeled target ``val_loader``.

    Trains two copies of ``model`` (ERM-only, then PMH via :func:`robust_fit`) unless
    ``pmh_result`` is already available. Returns the same :class:`EvaluationReport` shape
    as :func:`evaluate_baseline_vs_pmh`.
    """
    import copy

    import torch
    from pmh.hooks import resolve_hook

    torch.manual_seed(seed)
    app = check_applicability(
        stack="pytorch",
        has_target_domain=target_batches is not None,
    )

    hook_used: str | Any = hook
    if hook == "auto":
        hook_used = suggest_hook(model).hook

    model_erm = copy.deepcopy(model)
    encoder_erm = resolve_hook(model_erm, hook_used)
    _train_erm_epochs(model_erm, encoder_erm, head, train_loader, epochs=epochs)

    if pmh_result is not None:
        pmh_out = pmh_result
    else:
        model_pmh = copy.deepcopy(model)
        pmh_out = robust_fit(
            model_pmh,
            train_loader,
            source_batches=source_batches,
            target_batches=target_batches,
            hook=hook_used,
            head=head,
            nuisance=nuisance,
            rank=rank,
            epochs=epochs,
            applicability=app,
        )

    acc_b0 = _accuracy_on_loader(model_erm, encoder_erm, head, val_loader)
    acc_pmh = _accuracy_on_loader(
        pmh_out.trainer.model,
        pmh_out.trainer.encoder,
        head,
        val_loader,
    )

    notes = list(app.reasons[:1]) if app.reasons else []
    notes.append("Run compare_arms for matched / wrong_w / isotropic controls (advanced).")
    notes.append("See docs/WHEN_PMH_HELPS.md for when to expect gains.")

    return EvaluationReport(
        baseline_metric=acc_b0,
        pmh_metric=acc_pmh,
        preflight=pmh_out.preflight,
        preflight_message=preflight_plain_english(pmh_out.preflight),
        notes=[n for n in notes if n],
    )


def robust_fit_text_domains(
    model: Any,
    tokenizer: Any,
    train_loader: Any,
    source_texts: list[str],
    target_texts: list[str],
    *,
    epochs: int = 3,
    rank: int | None = 32,
    pmh_config: PMHConfig | None = None,
    **kwargs: Any,
) -> RobustFitResult:
    """HF / LLM path: estimate domain shift on text embeddings, then train.

    Use when you have two text corpora (same label semantics), not style-pair JSONL.
    For style formatting shift, use ``HFPMHTrainer.estimate_style`` instead.
    """
    from pmh.hf_trainer import HFPMHTrainer

    app = check_applicability(
        stack="hf",
        n_source=len(source_texts),
        n_target=len(target_texts),
        has_target_domain=True,
    )
    trainer = HFPMHTrainer(
        model,
        tokenizer,
        nuisance="domain_shift",
        rank=rank,
        pmh_config=pmh_config or PMHConfig.balanced(),
        **kwargs,
    )
    trainer.estimate_text_domains(source_texts, target_texts)
    stats = trainer.fit(train_loader, epochs=epochs, reestimate=False)
    pf = trainer.artifact_.preflight if trainer.artifact_ else None
    return RobustFitResult(
        trainer=trainer,
        stats=stats,
        applicability=app,
        hook_used="hf_hidden_states",
        preflight=pf,
    )
