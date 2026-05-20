"""Thin loaders: two domains from ``.npy`` or tensors → batches for estimate/train."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_FEATURE_NAMES = (
    "features.npy",
    "embeddings.npy",
    "x.npy",
    "representations.npy",
)


def resolve_feature_npy(directory: str | Path) -> Path:
    """Pick a feature matrix ``.npy`` inside ``directory`` (first match or sole ``.npy``)."""
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"not a directory: {d}")
    for name in _FEATURE_NAMES:
        p = d / name
        if p.is_file():
            return p
    npys = sorted(d.glob("*.npy"))
    if len(npys) == 1:
        return npys[0]
    if not npys:
        raise FileNotFoundError(
            f"no .npy in {d}; add features.npy or pass --source-npy explicitly"
        )
    raise FileNotFoundError(f"multiple .npy in {d}: {[p.name for p in npys[:5]]}; specify file")


def resolve_labels_npy(directory: str | Path) -> Path | None:
    """Optional ``labels.npy`` / ``y.npy`` beside features."""
    d = Path(directory)
    for name in ("labels.npy", "y.npy", "label.npy"):
        p = d / name
        if p.is_file():
            return p
    return None


def load_domain_dirs(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    source_npy: str | Path | None = None,
    target_npy: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Load features (and optional labels) from two folders."""
    sd, td = Path(source_dir), Path(target_dir)
    xs_path = Path(source_npy) if source_npy else resolve_feature_npy(sd)
    xt_path = Path(target_npy) if target_npy else resolve_feature_npy(td)
    xs = np.load(xs_path).astype(np.float32)
    xt = np.load(xt_path).astype(np.float32)
    ls_path = resolve_labels_npy(sd)
    lt_path = resolve_labels_npy(td)
    ys = np.load(ls_path).astype(np.int64) if ls_path else None
    yt = np.load(lt_path).astype(np.int64) if lt_path else None
    return xs, ys, xt, yt


def load_domain_arrays(
    source: str | Path | np.ndarray,
    target: str | Path | np.ndarray,
    *,
    source_labels: str | Path | np.ndarray | None = None,
    target_labels: str | Path | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Load ``[N, d]`` feature matrices (and optional labels) from paths or arrays."""

    def _arr(x: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(x, (str, Path)):
            return np.load(x).astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    xs = _arr(source)
    xt = _arr(target)
    ys = _arr(source_labels) if source_labels is not None else None
    yt = _arr(target_labels) if target_labels is not None else None
    return xs, ys, xt, yt


def batch_iterators(
    x_source: np.ndarray | torch.Tensor,
    x_target: np.ndarray | torch.Tensor,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
) -> tuple[Iterator[Any], Iterator[Any]]:
    """Yield PyTorch batch tuples for ``PMHTrainer.estimate`` / ``fit``."""

    def _loader(x: np.ndarray | torch.Tensor) -> DataLoader:
        t = torch.as_tensor(x, dtype=torch.float32)
        return DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=shuffle)

    return iter(_loader(x_source)), iter(_loader(x_target))


def batch_iterators_labeled(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    *,
    batch_size: int = 32,
) -> tuple[Iterator[Any], Iterator[Any]]:
    """Labeled batches ``(x, y)`` for D1 estimate paths."""
    ls = DataLoader(
        TensorDataset(
            torch.from_numpy(x_source.astype(np.float32)),
            torch.from_numpy(y_source.astype(np.int64)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    lt = DataLoader(
        TensorDataset(
            torch.from_numpy(x_target.astype(np.float32)),
            torch.from_numpy(y_target.astype(np.int64)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return iter(ls), iter(lt)
