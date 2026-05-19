"""Office-31 frozen ResNet-18 features (optional torchvision)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DOMAIN_NAMES = ("amazon", "dslr", "webcam")


def list_office31_domains() -> tuple[str, ...]:
    return DOMAIN_NAMES


def _require_torchvision() -> tuple:
    try:
        import torch
        from torchvision import models, transforms
        from torchvision.datasets import ImageFolder
    except ImportError as exc:
        raise ImportError(
            "Office-31 features require torch and torchvision. "
            'Install with: pip install "matching-pmh[vision]"'
        ) from exc
    return torch, models, transforms, ImageFolder


def domain_path(root: str | Path, domain: str) -> Path:
    root = Path(root)
    if domain not in DOMAIN_NAMES:
        raise ValueError(f"domain must be one of {DOMAIN_NAMES}")
    # Common layouts: root/amazon or root/office31/amazon
    for candidate in (root / domain, root / "office31" / domain, root / "Office31" / domain):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Office-31 domain folder not found under {root} (tried {DOMAIN_NAMES})"
    )


def extract_office31_features(
    root: str | Path,
    domain: str,
    *,
    max_samples: int | None = 2000,
    seed: int = 0,
    batch_size: int = 32,
    backbone: str = "resnet18",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate ResNet features and labels for one domain.

    Parameters
    ----------
    root : path
        Dataset root containing domain subfolders (class-per-subfolder layout).
    domain : str
        ``amazon``, ``dslr``, or ``webcam``.
    max_samples : int, optional
        Subsample for faster estimation.

    Returns
    -------
    features : ndarray [N, 512]
    labels : ndarray [N]
    """
    torch, models, transforms, ImageFolder = _require_torchvision()
    root = Path(root)
    folder = domain_path(root, domain)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = ImageFolder(str(folder), transform=tfm)
    n = len(ds)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if max_samples is not None and n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)

    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"unsupported backbone {backbone!r}")

    model.eval().to(device)
    feats_list: list[np.ndarray] = []
    labels_list: list[int] = []

    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            imgs = torch.stack([ds[int(i)][0] for i in batch_idx]).to(device)
            emb = model(imgs).cpu().numpy().astype(np.float32)
            feats_list.append(emb)
            labels_list.extend(int(ds[int(i)][1]) for i in batch_idx)

    return np.concatenate(feats_list, axis=0), np.array(labels_list, dtype=np.int64)
