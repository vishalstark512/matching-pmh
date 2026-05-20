"""Office-31 frozen ResNet-18 features (optional torchvision)."""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

DOMAIN_NAMES = ("amazon", "dslr", "webcam")

# Official domain-adaptation page (Judy Hoffman). Override with download_office31(url=...).
DEFAULT_OFFICE31_TAR_URL = "https://faculty.cc.gatech.edu/~judy/domainadapt/office31.tar"


def list_office31_domains() -> tuple[str, ...]:
    return DOMAIN_NAMES


def verify_office31_layout(root: str | Path) -> None:
    """Raise ``FileNotFoundError`` if any domain folder is missing under *root*."""
    root = Path(root)
    missing = []
    for domain in DOMAIN_NAMES:
        try:
            domain_path(root, domain)
        except FileNotFoundError:
            missing.append(domain)
    if missing:
        raise FileNotFoundError(
            f"Office-31 domains missing under {root}: {missing}. "
            "Run: python scripts/download_office31.py --root YOUR_PATH"
        )


def download_office31(
    root: str | Path,
    *,
    url: str | None = None,
    force: bool = False,
) -> Path:
    """Download and extract Office-31 into *root* (no data committed to git).

    Returns path to the downloaded archive file under *root*.
    """
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if not force:
        try:
            verify_office31_layout(root)
            print(f"Office-31 already present under {root}; skip download (use force=True to re-fetch)")
            return root / "office31.tar"
        except FileNotFoundError:
            pass

    tar_url = url or DEFAULT_OFFICE31_TAR_URL
    archive = root / "office31.tar"
    print(f"Downloading {tar_url} -> {archive} (this may take several minutes)...")

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = block_num * block_size
        pct = min(100, done * 100 // total_size)
        if block_num % 500 == 0:
            print(f"  ... {pct}%", flush=True)

    urllib.request.urlretrieve(tar_url, archive, reporthook=_report)
    print("Extracting...")
    with tarfile.open(archive, "r:*") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(path=root, filter="data")
        else:
            tf.extractall(path=root)

    # Common layouts: office31/amazon or amazon/ at root
    for sub in ("office31", "Office31", "images"):
        candidate = root / sub
        if candidate.is_dir() and any((candidate / d).is_dir() for d in DOMAIN_NAMES):
            for domain in DOMAIN_NAMES:
                src = candidate / domain
                dst = root / domain
                if src.is_dir() and not dst.exists():
                    shutil.move(str(src), str(dst))
            break

    verify_office31_layout(root)
    return archive


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
