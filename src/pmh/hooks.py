"""Resolve encoder hooks on PyTorch models (any architecture)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn as nn

Encoder = Callable[[torch.Tensor], torch.Tensor]

# Common hook names → attribute path on known model families
HOOK_REGISTRY: dict[str, dict[str, str]] = {
    "torchvision_resnet": {
        "default": "avgpool",
        "avgpool": "avgpool",
        "layer4": "layer4",
        "backbone": "avgpool",
    },
    "torchvision_vit": {
        "default": "forward_features",
        "cls": "forward_features",
        "backbone": "forward_features",
    },
    "timm": {
        "default": "forward_features",
        "backbone": "forward_features",
        "features": "forward_features",
    },
    "hf_causal_lm": {
        "default": "model",
        "backbone": "model",
        "transformer": "model",
    },
    "hf_encoder": {
        "default": "",
        "backbone": "",
    },
    "sequential": {
        "default": "0",
        "backbone": "0",
    },
    "generic": {
        "default": "",
    },
}


def _get_attr(obj: Any, path: str) -> Any:
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def detect_model_family(model: nn.Module) -> str:
    """Best-effort model family tag for hook hints."""
    name = type(model).__name__.lower()
    module = (type(model).__module__ or "").lower()
    if "timm" in module:
        return "timm"
    if "resnet" in name or ("torchvision" in module and "resnet" in name):
        return "torchvision_resnet"
    if "vit" in name or "visiontransformer" in name:
        return "torchvision_vit"
    if "pretrainedmodel" in name or "bertmodel" in name or "roberta" in module:
        return "hf_encoder"
    if "causallm" in name or "forcausallm" in name or "gpt" in module:
        return "hf_causal_lm"
    if isinstance(model, nn.Sequential):
        return "sequential"
    return "generic"


def list_hook_families() -> dict[str, list[str]]:
    """Known families and alias names for documentation / CLI."""
    return {family: sorted(aliases.keys()) for family, aliases in HOOK_REGISTRY.items()}


def resolve_hook(
    model: nn.Module,
    hook: str | nn.Module | Callable[[torch.Tensor], torch.Tensor] | None = None,
    *,
    pool_spatial: bool = True,
) -> Encoder:
    """Return ``encoder(x) -> [B, d]`` from a model and hook spec.

    Parameters
    ----------
    model : nn.Module
        Full model or backbone submodule.
    hook : str, Module, callable, or None
        - **callable**: used as-is (must return [B, d]).
        - **Module**: ``hook(x)``.
        - **str**: attribute path (``"layer4"``, ``"net"``) or registry alias (``"backbone"``).
        - **None**: family default from :func:`detect_model_family`.
    pool_spatial : bool
        If output is [B,C,H,W], global average pool to [B,C].
    """
    if hook is not None and callable(hook) and not isinstance(hook, nn.Module):
        return _wrap_encoder(hook, pool_spatial=pool_spatial)

    if isinstance(hook, nn.Module):
        mod = hook

        def _enc_mod(x: torch.Tensor) -> torch.Tensor:
            return validate_representation(mod(x), pool_spatial=pool_spatial)

        return _enc_mod

    family = detect_model_family(model)
    registry = HOOK_REGISTRY.get(family, HOOK_REGISTRY["generic"])

    if hook is None or hook == "default":
        path = registry.get("default", "")
    elif hook in registry:
        path = registry[hook]
    else:
        path = str(hook)

    # timm / ViT: prefer forward_features when hook points at full model
    if family in ("timm", "torchvision_vit") and path in ("", "forward_features"):
        if hasattr(model, "forward_features"):

            def _enc_timm(x: torch.Tensor) -> torch.Tensor:
                return validate_representation(model.forward_features(x), pool_spatial=pool_spatial)

            return _enc_timm

    target = _get_attr(model, path) if path else model

    if isinstance(target, nn.Module):

        def _enc(x: torch.Tensor) -> torch.Tensor:
            out = target(x)
            return validate_representation(out, pool_spatial=pool_spatial)

        return _enc

    raise ValueError(
        f"Could not resolve hook={hook!r} on {type(model).__name__}. "
        f"Pass a callable, nn.Module, or path like 'layer4'. "
        f"Family={family}, known aliases: {list(registry.keys())}. "
        f"See docs/hooks.md or encoder_hf() / encoder_timm()."
    )


def encoder_timm(model: nn.Module, *, pool_spatial: bool = True) -> Encoder:
    """Hook for timm models via ``forward_features`` (optional dependency: timm)."""
    if not hasattr(model, "forward_features"):
        raise ValueError(f"{type(model).__name__} has no forward_features; pass a timm model.")

    def _enc(x: torch.Tensor) -> torch.Tensor:
        return validate_representation(model.forward_features(x), pool_spatial=pool_spatial)

    return _enc


def encoder_torchvision_resnet(
    model: nn.Module,
    *,
    layer: Literal["avgpool", "layer4"] = "avgpool",
    pool_spatial: bool = True,
) -> Encoder:
    """Hook at ResNet ``avgpool`` (pre-FC) or ``layer4`` feature maps."""
    return resolve_hook(model, layer, pool_spatial=pool_spatial)


def encoder_hf_hidden_states(
    model: nn.Module,
    *,
    layer: int = -1,
    pool: Literal["last", "mean", "cls"] = "last",
    attention_mask: torch.Tensor | None = None,
) -> Encoder:
    """Hook for Hugging Face models with ``output_hidden_states=True``.

    ``x`` must be ``input_ids`` shaped ``[B, T]`` (token indices).
    """
    def _enc(x: torch.Tensor) -> torch.Tensor:
        kwargs: dict[str, Any] = {"input_ids": x, "output_hidden_states": True, "return_dict": True}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        out = model(**kwargs)
        hidden = out.hidden_states
        if hidden is None:
            raise ValueError("Model did not return hidden_states; use a HF PreTrainedModel.")
        h = hidden[layer]
        if pool == "cls":
            return h[:, 0]
        if pool == "last":
            return h[:, -1]
        return h.mean(dim=1)

    return _enc


def encoder_hf_with_mask(
    model: nn.Module,
    *,
    layer: int = -1,
    pool: Literal["last", "mean", "cls"] = "last",
) -> tuple[Encoder, Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, torch.Tensor | None]]]:
    """HF encoder plus batch unpacker for ``(input_ids, attention_mask)`` loaders."""

    def unpack(batch: tuple[torch.Tensor, ...] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(batch, (tuple, list)):
            ids = batch[0]
            mask = batch[1] if len(batch) > 1 else None
            return ids, mask
        return batch, None

    base = encoder_hf_hidden_states(model, layer=layer, pool=pool)

    def _enc(x: torch.Tensor) -> torch.Tensor:
        return base(x)

    return _enc, unpack


def encoder_gnn_mean_pool(
    node_encoder: nn.Module,
    *,
    batch_index: torch.Tensor | None = None,
) -> Encoder:
    """Pool node embeddings to graph-level ``[B, d]`` (mean over nodes).

    If ``batch_index`` is None, assumes a single graph (mean over all nodes).
    For PyG-style batches, pass ``batch`` vector from DataLoader.
    """

    def _enc(node_features: torch.Tensor, batch_idx: torch.Tensor | None = batch_index) -> torch.Tensor:
        h = node_encoder(node_features)
        if h.dim() != 2:
            h = validate_representation(h)
        if batch_idx is None:
            return h.mean(dim=0, keepdim=True)
        b = batch_idx.long()
        n_graphs = int(b.max().item()) + 1
        out = []
        for g in range(n_graphs):
            mask = b == g
            out.append(h[mask].mean(dim=0))
        return torch.stack(out, dim=0)

    return _enc  # type: ignore[return-value]


def _wrap_encoder(fn: Encoder, *, pool_spatial: bool) -> Encoder:
    def _enc(x: torch.Tensor) -> torch.Tensor:
        return validate_representation(fn(x), pool_spatial=pool_spatial)

    return _enc


def validate_representation(h: torch.Tensor, *, pool_spatial: bool = True) -> torch.Tensor:
    """Ensure hook output is ``[batch, dim]``; pool 4D feature maps if needed."""
    if h.dim() == 2:
        return h
    if h.dim() == 4 and pool_spatial:
        return h.mean(dim=(2, 3))
    if h.dim() == 3:
        if pool_spatial:
            return h.mean(dim=1)
        return h
    raise ValueError(
        f"Hook must return [B, d] (or [B,C,H,W] with pool_spatial=True). Got shape {tuple(h.shape)}. "
        "Pick a different layer or set pool_spatial=False and flatten manually."
    )


def register_hook_family(name: str, aliases: dict[str, str]) -> None:
    """Register hook path aliases for a custom model family."""
    HOOK_REGISTRY[name] = dict(aliases)
