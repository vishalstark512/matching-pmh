"""Train and evaluate standard PMH arms on a PyTorch model."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from pmh.artifact import SigmaTaskEstimate
from pmh.benchmark.arms import STANDARD_ARMS, ArmName, resolve_arms
from pmh.config import PMHConfig
from pmh.integrations.torch import PMHCallback, train_epoch_with_pmh
from pmh.training import PMHLoss

EncoderFn = Callable[[torch.Tensor], torch.Tensor]
HeadFn = Callable[[torch.Tensor], torch.Tensor]
TaskLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MetricFn = Callable[[nn.Module, EncoderFn, HeadFn, Any], float]


@dataclass
class ArmEpochRecord:
    epoch: int
    task_loss: float
    pmh_loss: float
    total_loss: float


@dataclass
class ArmRunResult:
    arm: str
    val_metric: float | None = None
    metric_name: str = "val_accuracy"
    epochs: list[ArmEpochRecord] = field(default_factory=list)
    final: dict[str, float] = field(default_factory=dict)
    geometry: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "arm": self.arm,
            "metric_name": self.metric_name,
            "val_metric": self.val_metric,
            "final": self.final,
            "epochs": [e.__dict__ for e in self.epochs],
        }
        if self.geometry:
            out["geometry"] = self.geometry
        return out


@dataclass
class BenchmarkResult:
    """Full multi-arm run on your model factory and dataloaders."""

    artifact_method: str
    artifact_preflight: str | None
    artifact_eigengap: float | None
    arms: dict[str, ArmRunResult] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": {
                "method": self.artifact_method,
                "preflight": self.artifact_preflight,
                "eigengap": self.artifact_eigengap,
            },
            "arms": {k: v.to_dict() for k, v in self.arms.items()},
            "notes": self.notes,
        }


def default_accuracy_metric(
    model: nn.Module,
    encoder: EncoderFn,
    head: HeadFn,
    val_loader: Any,
    *,
    device: torch.device | None = None,
) -> float:
    """Classification accuracy on ``(x, y)`` batches."""
    model.eval()
    correct = 0
    total = 0
    dev = device
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1]
            else:
                raise ValueError("val_loader must yield (x, y)")
            if dev is not None:
                x, y = x.to(dev), y.to(dev)
            logits = head(encoder(x))
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)


def collect_val_embeddings(
    encoder: EncoderFn,
    val_loader: Any,
    *,
    device: torch.device | None = None,
    max_batches: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack validation embeddings and labels for geometry probes."""
    parts: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            if max_batches is not None and n >= max_batches:
                break
            if isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1]
            else:
                raise ValueError("val_loader must yield (x, y)")
            if device is not None:
                x = x.to(device)
            parts.append(encoder(x).detach().float().cpu())
            labels.append(y.detach().cpu() if isinstance(y, torch.Tensor) else torch.as_tensor(y))
            n += 1
    if not parts:
        raise ValueError("val_loader empty")
    return torch.cat(parts, dim=0), torch.cat(labels, dim=0)


def default_geometry_metric(
    artifact: SigmaTaskEstimate,
    encoder: EncoderFn,
    val_loader: Any,
    *,
    device: torch.device | None = None,
    max_batches: int = 20,
    seed: int = 0,
) -> dict[str, float | None]:
    """TDI_cls and D_N/D_S on validation embeddings (paper §6)."""
    from pmh.tdi import geometry_report

    emb, y = collect_val_embeddings(
        encoder, val_loader, device=device, max_batches=max_batches
    )
    w = artifact.metadata.get("w")
    w_np = None
    if w is not None:
        w_np = w.cpu().numpy() if isinstance(w, torch.Tensor) else np.asarray(w)
    rep = geometry_report(
        emb.numpy(),
        y.numpy().reshape(-1),
        w=w_np,
        seed=seed,
    )
    return rep.to_dict()


def _pmh_for_arm(
    artifact: SigmaTaskEstimate,
    arm: ArmName,
    pmh_config: PMHConfig,
    *,
    wrong_rank: int,
    wrong_seed: int = 0,
) -> PMHLoss | None:
    if arm == "b0":
        return None
    from pmh.benchmark.arms import ARM_SPECS

    spec = ARM_SPECS[arm]
    if spec.pmh_mode is None:
        return None
    return PMHLoss(
        artifact,
        pmh_config,
        mode=spec.pmh_mode,  # type: ignore[arg-type]
        wrong_rank=wrong_rank,
        wrong_seed=wrong_seed,
    )


def train_one_arm(
    arm: ArmName,
    model: nn.Module,
    artifact: SigmaTaskEstimate,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    encoder: EncoderFn,
    head: HeadFn,
    *,
    epochs: int,
    pmh_config: PMHConfig | None = None,
    wrong_rank: int = 32,
    wrong_seed: int = 0,
    device: torch.device | str | None = None,
    task_loss_fn: TaskLossFn | None = None,
    max_steps_per_epoch: int | None = None,
) -> ArmRunResult:
    """Train a single arm; return epoch curves."""
    pmh_cfg = pmh_config or PMHConfig()
    pmh_mod = _pmh_for_arm(artifact, arm, pmh_cfg, wrong_rank=wrong_rank, wrong_seed=wrong_seed)
    result = ArmRunResult(arm=arm)
    task_fn = task_loss_fn or nn.functional.cross_entropy

    if pmh_mod is None:
        dev = torch.device(device) if device is not None else None
        model.train()
        for epoch in range(1, epochs + 1):
            stats = {"task_loss": 0.0, "pmh_loss": 0.0, "total_loss": 0.0, "n_steps": 0}
            steps = 0
            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    x, y = batch[0], batch[1]
                else:
                    raise ValueError("train_loader must yield (x, y)")
                if dev is not None:
                    x, y = x.to(dev), y.to(dev)
                optimizer.zero_grad(set_to_none=True)
                h = encoder(x)
                task = task_fn(head(h), y)
                task.backward()
                optimizer.step()
                stats["task_loss"] += float(task.detach())
                stats["total_loss"] += float(task.detach())
                stats["n_steps"] += 1
                steps += 1
                if max_steps_per_epoch is not None and steps >= max_steps_per_epoch:
                    break
            n = max(stats["n_steps"], 1)
            result.epochs.append(
                ArmEpochRecord(
                    epoch=epoch,
                    task_loss=stats["task_loss"] / n,
                    pmh_loss=0.0,
                    total_loss=stats["total_loss"] / n,
                )
            )
        if result.epochs:
            last = result.epochs[-1]
            result.final = {
                "task_loss": last.task_loss,
                "pmh_loss": last.pmh_loss,
                "total_loss": last.total_loss,
            }
        return result

    callback = PMHCallback(pmh_mod, encoder, head=head, task_loss_fn=task_fn)
    for epoch in range(1, epochs + 1):
        avgs = train_epoch_with_pmh(
            model,
            callback,
            train_loader,
            optimizer,
            epoch=epoch,
            device=device,
            max_steps=max_steps_per_epoch,
        )
        result.epochs.append(
            ArmEpochRecord(
                epoch=epoch,
                task_loss=avgs["task_loss"],
                pmh_loss=avgs["pmh_loss"],
                total_loss=avgs["total_loss"],
            )
        )
    if result.epochs:
        last = result.epochs[-1]
        result.final = {
            "task_loss": last.task_loss,
            "pmh_loss": last.pmh_loss,
            "total_loss": last.total_loss,
        }
    return result


def run_benchmark_protocol(
    artifact: SigmaTaskEstimate,
    model_factory: Callable[[], nn.Module],
    setup_model: Callable[[nn.Module], tuple[EncoderFn, HeadFn, torch.optim.Optimizer]],
    train_loader: Any,
    val_loader: Any,
    *,
    arms: list[str] | None = None,
    epochs: int = 15,
    pmh_config: PMHConfig | None = None,
    wrong_rank: int = 32,
    wrong_seed: int = 0,
    device: torch.device | str | None = None,
    metric_fn: MetricFn | None = None,
    include_geometry: bool = False,
    geometry_fn: Callable[..., dict[str, float | None]] | None = None,
    max_steps_per_epoch: int | None = None,
    shared_init: bool = True,
) -> BenchmarkResult:
    """Train B0 / matched / wrong-W / isotropic and evaluate on ``val_loader``.

    Parameters
    ----------
    shared_init :
        If True, clone initial weights across arms (fair architecture comparison).
    """
    arm_list = resolve_arms(arms)
    out = BenchmarkResult(
        artifact_method=artifact.method,
        artifact_preflight=artifact.preflight,
        artifact_eigengap=artifact.eigengap,
    )
    if "coral" in arm_list:
        out.notes.append("coral arm skipped in PyTorch protocol; use numpy/sklearn path (example 06).")

    pytorch_arms = [a for a in arm_list if a != "coral"]
    metric = metric_fn or (
        lambda m, enc, hd, loader: default_accuracy_metric(m, enc, hd, loader, device=device)
    )

    template_state = None
    if shared_init and pytorch_arms:
        m0 = model_factory()
        template_state = copy.deepcopy(m0.state_dict())

    for arm in pytorch_arms:
        model = model_factory()
        if template_state is not None:
            model.load_state_dict(template_state)
        encoder, head, optimizer = setup_model(model)
        run = train_one_arm(
            arm,
            model,
            artifact,
            train_loader,
            optimizer,
            encoder,
            head,
            epochs=epochs,
            pmh_config=pmh_config,
            wrong_rank=wrong_rank,
            wrong_seed=wrong_seed,
            device=device,
            max_steps_per_epoch=max_steps_per_epoch,
        )
        run.val_metric = metric(model, encoder, head, val_loader)
        run.metric_name = "val_accuracy"
        if include_geometry:
            gfn = geometry_fn or (
                lambda _m, enc, _hd, loader: default_geometry_metric(
                    artifact, enc, loader, device=device, seed=wrong_seed
                )
            )
            try:
                run.geometry = gfn(model, encoder, head, val_loader)
            except Exception as exc:  # noqa: BLE001
                out.notes.append(f"geometry failed for {arm}: {exc}")
        out.arms[arm] = run

    return out
