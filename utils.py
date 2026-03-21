"""Reproducibility, data loaders, model factory, timing, and sanity checks."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import AZConvConfig, AZConvNet, StandardConvNet, count_parameters

VARIANTS = [
    "baseline",
    "az_full",
    "az_no_fuzzy",
    "az_no_aniso",
    "az_fixed_dirs",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: strict determinism is not guaranteed for all ops; good-enough for prototype.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_channels_and_classes(name: str) -> Tuple[int, int]:
    n = name.lower().replace("-", "_")
    if n in ("cifar10", "cifar_10"):
        return 3, 10
    if n in ("fashion_mnist", "fashionmnist"):
        return 1, 10
    raise ValueError(f"Unknown dataset: {name}")


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    import torchvision
    import torchvision.transforms as T

    name = cfg["dataset"]
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 0))
    data_root = cfg.get("data_root", "./data")

    in_c, num_classes = dataset_channels_and_classes(name)
    n = name.lower().replace("-", "_")

    if n in ("cifar10", "cifar_10"):
        train_tf = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomCrop(32, padding=4),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        test_tf = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        test_full = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    elif n in ("fashion_mnist", "fashionmnist"):
        train_tf = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
        test_tf = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
        train_set = torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True, transform=train_tf)
        test_full = torchvision.datasets.FashionMNIST(root=data_root, train=False, download=True, transform=test_tf)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    val_fraction = float(cfg.get("val_fraction", 0.1))
    n_train = len(train_set)
    n_val = int(n_train * val_fraction)
    n_tr = n_train - n_val
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    train_split, val_split = torch.utils.data.random_split(train_set, [n_tr, n_val], generator=generator)

    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_full,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, in_c, num_classes


def az_config_for_variant(variant: str) -> AZConvConfig:
    if variant == "az_full":
        return AZConvConfig(use_fuzzy=True, use_anisotropy=True, learn_directions=True)
    if variant == "az_no_fuzzy":
        return AZConvConfig(use_fuzzy=False, use_anisotropy=True, learn_directions=True)
    if variant == "az_no_aniso":
        return AZConvConfig(use_fuzzy=True, use_anisotropy=False, learn_directions=True)
    if variant == "az_fixed_dirs":
        return AZConvConfig(use_fuzzy=True, use_anisotropy=True, learn_directions=False)
    raise ValueError(f"Not an AZ variant: {variant}")


def build_model(variant: str, num_classes: int, in_channels: int, num_rules: int = 4) -> nn.Module:
    if variant == "baseline":
        return StandardConvNet(num_classes=num_classes, in_channels=in_channels)
    if variant.startswith("az_"):
        cfg = az_config_for_variant(variant)
        return AZConvNet(num_classes=num_classes, in_channels=in_channels, num_rules=num_rules, cfg=cfg)
    raise ValueError(f"Unknown variant: {variant}")


@torch.no_grad()
def sanity_check_azconv_forward(device: torch.device) -> None:
    """Shape / finiteness checks for AZConv2d."""

    from models.azconv import AZConv2d

    B, C, H, W, R, k, out_c = 2, 8, 32, 32, 4, 3, 16
    x = torch.randn(B, C, H, W, device=device)
    layer = AZConv2d(C, out_c, kernel_size=k, num_rules=R).to(device)
    y = layer(x)
    assert y.shape == (B, out_c, H, W), y.shape
    assert torch.isfinite(y).all(), "non-finite outputs"

    x_unfold = torch.nn.functional.unfold(x, k, padding=k // 2, stride=1)
    assert x_unfold.shape == (B, C * (k * k), H * W), x_unfold.shape


def measure_inference_time(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    in_channels: int,
    spatial: int,
    warmup: int = 3,
    iters: int = 20,
) -> float:
    """Mean seconds per forward pass over `iters` batches after warmup."""

    model.eval()
    x = torch.randn(batch_size, in_channels, spatial, spatial, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / iters


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_interpretation(path: Path, rows: List[Dict[str, Any]], baseline_key: str = "baseline") -> None:
    """Short narrative comparing variants (heuristic, not statistical testing)."""

    by_var = {r.get("variant"): r for r in rows}
    base = by_var.get(baseline_key)
    az = by_var.get("az_full")

    lines = [
        "Interpretation (prototype — not a statistical claim)",
        "=" * 60,
        "",
        "Does AZConv help?",
    ]

    if base and az:
        bacc = float(base.get("test_accuracy", 0.0))
        aacc = float(az.get("test_accuracy", 0.0))
        delta = aacc - bacc
        lines.append(f"- Test accuracy: baseline {bacc:.2f}% vs AZ full {aacc:.2f}% (Δ = {delta:+.2f} pp).")
        if abs(delta) < 0.5:
            lines.append("- On this setup, the two models are likely within noise; run with more epochs/seeds.")
        elif delta > 0:
            lines.append("- AZ full is ahead; fuzzy + anisotropic weighting may help capture directional patterns.")
        else:
            lines.append("- Baseline is ahead; the inductive bias may be too rigid or under-parameterized.")
    else:
        lines.append("- Run all variants so we can compute baseline vs az_full deltas.")

    lines.extend(
        [
            "",
            "Where might it help?",
            "- Tasks with directional structure (edges, strokes) could align with learnable stable/unstable axes.",
            "- Fuzzy compatibilities act like soft local gating between center and neighbor memberships.",
            "",
            "Computational cost:",
            "- AZConv uses `unfold` and aggregates over (rules × neighbors); latency is higher than a single Conv2d.",
            "- Compare `seconds_train_epoch` and `seconds_per_forward_batch` in `metrics.json`.",
            "",
            "Likely reasons for outcome:",
            "- The parameter distribution differs (AZ uses a gate + rule-wise weighting + 1×1 mixing).",
            "- Normalized fuzzy/anisotropic weights constrain effective receptive-field smoothing.",
            "- CIFAR-10 training for 15–30 epochs may still be insufficient for the full expressivity to emerge.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

