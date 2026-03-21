"""Shared utilities for classification and DRIVE-style segmentation experiments."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models import AZConvConfig, AZConvNet, AZSOTAUNet, AZUNet, BaselineUNet, StandardConvNet, count_parameters

VARIANTS = [
    "baseline",
    "az_full",
    "az_no_fuzzy",
    "az_no_aniso",
    "az_fixed_dirs",
    "az_cat",
    "az_sota",
    "az_sota_pure",
    "az_thesis",
]

CANONICAL_DRIVE_VARIANTS = [
    "baseline",
    "az_no_fuzzy",
    "az_no_aniso",
    "az_cat",
    "az_thesis",
]

DRIVE_SUPERIORITY_METRICS = [
    "test_dice",
    "test_iou",
    "test_precision",
    "test_recall",
    "test_specificity",
    "test_balanced_accuracy",
]

DRIVE_METRIC_LABELS = {
    "test_dice": "Dice",
    "test_iou": "IoU",
    "test_precision": "Precision",
    "test_recall": "Recall",
    "test_specificity": "Specificity",
    "test_accuracy": "Accuracy",
    "test_balanced_accuracy": "Balanced Acc",
}

THRESHOLD_DERIVED_METRICS = {
    "core_mean": ("dice", "iou", "precision", "recall", "specificity", "balanced_accuracy"),
    "core_min": ("dice", "iou", "precision", "recall", "specificity", "balanced_accuracy"),
    "dice_balanced_mean": ("dice", "balanced_accuracy"),
}


def _strip_metric_prefix(metric_name: str) -> str:
    if metric_name.startswith("test_"):
        return metric_name[5:]
    if metric_name.startswith("val_"):
        return metric_name[4:]
    return metric_name


def threshold_metric_value(row: Dict[str, float], metric_name: str) -> float:
    metric_key = _strip_metric_prefix(str(metric_name).lower().strip())
    if metric_key in row:
        return float(row[metric_key])
    if metric_key == "core_mean":
        keys = THRESHOLD_DERIVED_METRICS["core_mean"]
        return float(sum(float(row[key]) for key in keys) / len(keys))
    if metric_key == "core_min":
        keys = THRESHOLD_DERIVED_METRICS["core_min"]
        return float(min(float(row[key]) for key in keys))
    if metric_key == "dice_balanced_mean":
        keys = THRESHOLD_DERIVED_METRICS["dice_balanced_mean"]
        return float(sum(float(row[key]) for key in keys) / len(keys))
    raise KeyError(f"Threshold metric '{metric_name}' is not present in threshold sweep rows.")


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def load_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def task_for_dataset(name: str) -> str:
    n = name.lower().replace("-", "_")
    if n in ("cifar10", "cifar_10", "fashion_mnist", "fashionmnist"):
        return "classification"
    if n == "drive":
        return "segmentation"
    raise ValueError(f"Unknown dataset: {name}")


def dataset_channels_and_outputs(name: str) -> Tuple[int, int]:
    n = name.lower().replace("-", "_")
    if n in ("cifar10", "cifar_10"):
        return 3, 10
    if n in ("fashion_mnist", "fashionmnist"):
        return 1, 10
    if n == "drive":
        return 3, 1
    raise ValueError(f"Unknown dataset: {name}")


class DriveDataset(Dataset):
    """DRIVE vessel segmentation dataset loader."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        augment: bool = False,
        use_fov_mask: bool = True,
        crop_size: int | None = None,
        foreground_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.use_fov_mask = use_fov_mask
        self.crop_size = crop_size
        self.foreground_bias = float(foreground_bias)
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Tuple[Path, Path, Path]]:
        split_dir = self.root / self.split
        images_dir = split_dir / "images"
        manual_dir = split_dir / "1st_manual"
        mask_dir = split_dir / "mask"
        if not images_dir.exists() or not manual_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                "Expected DRIVE structure under "
                f"{split_dir} with subfolders images/, 1st_manual/, mask/."
            )

        def by_id(paths: Sequence[Path]) -> Dict[str, Path]:
            out: Dict[str, Path] = {}
            for path in paths:
                sample_id = path.name.split("_")[0]
                out[sample_id] = path
            return out

        image_map = by_id(sorted(images_dir.glob("*.*")))
        manual_map = by_id(sorted(manual_dir.glob("*.*")))
        mask_map = by_id(sorted(mask_dir.glob("*.*")))
        common_ids = sorted(set(image_map) & set(manual_map) & set(mask_map))
        if not common_ids:
            raise FileNotFoundError(f"No matched DRIVE samples found under {split_dir}")
        return [(image_map[idx], manual_map[idx], mask_map[idx]) for idx in common_ids]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_mask(self, path: Path) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        arr = (np.asarray(mask, dtype=np.float32) > 127.0).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
        return (image - mean) / std

    def _crop_triplet(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        fov: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.crop_size is None:
            return image, mask, fov

        crop_h = min(int(self.crop_size), image.shape[1])
        crop_w = min(int(self.crop_size), image.shape[2])
        max_top = image.shape[1] - crop_h
        max_left = image.shape[2] - crop_w
        top = 0 if max_top <= 0 else random.randint(0, max_top)
        left = 0 if max_left <= 0 else random.randint(0, max_left)

        valid_foreground = (mask > 0.5) & (fov > 0.5)
        if self.foreground_bias > 0.0 and valid_foreground.any() and random.random() < self.foreground_bias:
            ys, xs = torch.nonzero(valid_foreground[0], as_tuple=True)
            pick = random.randint(0, ys.numel() - 1)
            center_y = int(ys[pick].item())
            center_x = int(xs[pick].item())
            top = min(max(center_y - crop_h // 2, 0), max_top)
            left = min(max(center_x - crop_w // 2, 0), max_left)

        sl_h = slice(top, top + crop_h)
        sl_w = slice(left, left + crop_w)
        return image[:, sl_h, sl_w], mask[:, sl_h, sl_w], fov[:, sl_h, sl_w]

    def _apply_augment(self, image: torch.Tensor, mask: torch.Tensor, fov: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
            fov = torch.flip(fov, dims=[2])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
            fov = torch.flip(fov, dims=[1])
        if random.random() < 0.5:
            image = torch.rot90(image, k=2, dims=[1, 2])
            mask = torch.rot90(mask, k=2, dims=[1, 2])
            fov = torch.rot90(fov, k=2, dims=[1, 2])
        return image, mask, fov

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, mask_path, fov_path = self.samples[idx]
        image = self._load_rgb(image_path)
        mask = self._load_mask(mask_path)
        fov = self._load_mask(fov_path) if self.use_fov_mask else torch.ones_like(mask)
        image, mask, fov = self._crop_triplet(image, mask, fov)
        image = self._normalize(image)
        if self.augment:
            image, mask, fov = self._apply_augment(image, mask, fov)
        return image, mask, fov


def _build_classification_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    import torchvision
    import torchvision.transforms as T

    name = cfg["dataset"]
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 0))
    data_root = cfg.get("data_root", "./data")

    in_c, num_outputs = dataset_channels_and_outputs(name)
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
        eval_tf = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        val_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=eval_tf)
        test_full = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_tf)
    elif n in ("fashion_mnist", "fashionmnist"):
        train_tf = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
        eval_tf = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
        train_set = torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True, transform=train_tf)
        val_set = torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True, transform=eval_tf)
        test_full = torchvision.datasets.FashionMNIST(root=data_root, train=False, download=True, transform=eval_tf)
    else:
        raise ValueError(f"Unknown classification dataset: {name}")

    val_fraction = float(cfg.get("val_fraction", 0.1))
    n_train = len(train_set)
    n_val = int(n_train * val_fraction)
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    perm = torch.randperm(n_train, generator=generator).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_split = torch.utils.data.Subset(train_set, train_idx)
    val_split = torch.utils.data.Subset(val_set, val_idx)

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
    return train_loader, val_loader, test_loader, in_c, num_outputs


def _build_drive_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 0))
    data_root = Path(cfg.get("data_root", "./data")) / "DRIVE"
    use_fov_mask = bool(cfg.get("use_fov_mask", True))
    patch_size = cfg.get("drive_patch_size")
    patch_size = int(patch_size) if patch_size else None
    foreground_bias = float(cfg.get("drive_foreground_bias", 0.0))

    train_aug = DriveDataset(
        data_root,
        split="training",
        augment=True,
        use_fov_mask=use_fov_mask,
        crop_size=patch_size,
        foreground_bias=foreground_bias,
    )
    train_eval = DriveDataset(
        data_root,
        split="training",
        augment=False,
        use_fov_mask=use_fov_mask,
        crop_size=patch_size,
        foreground_bias=foreground_bias,
    )
    test_set = DriveDataset(
        data_root,
        split="test",
        augment=False,
        use_fov_mask=use_fov_mask,
        crop_size=None,
        foreground_bias=0.0,
    )

    val_fraction = float(cfg.get("val_fraction", 0.2))
    n_train = len(train_aug)
    n_val = max(1, int(round(n_train * val_fraction)))
    n_val = min(n_val, n_train - 1) if n_train > 1 else 1
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    perm = torch.randperm(n_train, generator=generator).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if not train_idx:
        train_idx = val_idx

    train_split = torch.utils.data.Subset(train_aug, train_idx)
    val_split = torch.utils.data.Subset(train_eval, val_idx)

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
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, 3, 1


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, str]:
    task = task_for_dataset(cfg["dataset"])
    if task == "classification":
        train_loader, val_loader, test_loader, in_c, num_outputs = _build_classification_dataloaders(cfg)
    else:
        train_loader, val_loader, test_loader, in_c, num_outputs = _build_drive_dataloaders(cfg)
    return train_loader, val_loader, test_loader, in_c, num_outputs, task


def estimate_drive_pos_weight(dataset: Dataset, min_weight: float = 1.0, max_weight: float = 25.0) -> float:
    positives = 0.0
    negatives = 0.0
    for _image, mask, valid_mask in dataset:
        valid = valid_mask > 0.5
        positives += float(mask[valid].sum().item())
        negatives += float(valid.sum().item() - mask[valid].sum().item())
    if positives <= 0.0:
        return float(min_weight)
    ratio = negatives / positives
    return float(max(min_weight, min(max_weight, ratio)))


def az_config_for_variant(variant: str) -> AZConvConfig:
    if variant in {"az_sota", "az_sota_pure", "az_thesis"}:
        return AZConvConfig(
            use_fuzzy=True,
            use_anisotropy=True,
            learn_directions=False,
            geometry_mode="fixed_cat_map",
            use_value_projection=True,
            normalize_kernel=True,
            min_hyperbolicity=0.1,
        )
    if variant == "az_full":
        return AZConvConfig(
            use_fuzzy=True,
            use_anisotropy=True,
            learn_directions=True,
            geometry_mode="local_hyperbolic",
            use_value_projection=True,
            normalize_kernel=True,
            min_hyperbolicity=0.1,
        )
    if variant == "az_no_fuzzy":
        return AZConvConfig(
            use_fuzzy=False,
            use_anisotropy=True,
            learn_directions=True,
            geometry_mode="local_hyperbolic",
            use_value_projection=True,
            normalize_kernel=True,
            min_hyperbolicity=0.1,
        )
    if variant == "az_no_aniso":
        return AZConvConfig(
            use_fuzzy=True,
            use_anisotropy=False,
            learn_directions=True,
            geometry_mode="local_hyperbolic",
            use_value_projection=True,
            normalize_kernel=True,
            min_hyperbolicity=0.1,
        )
    if variant == "az_fixed_dirs":
        return AZConvConfig(
            use_fuzzy=True,
            use_anisotropy=True,
            learn_directions=False,
            geometry_mode="learned_angle",
            use_value_projection=True,
            normalize_kernel=True,
            min_hyperbolicity=0.1,
        )
    if variant == "az_cat":
        return AZConvConfig(
            use_fuzzy=True,
            use_anisotropy=True,
            learn_directions=False,
            geometry_mode="fixed_cat_map",
            use_value_projection=True,
            normalize_kernel=True,
            min_hyperbolicity=0.1,
        )
    raise ValueError(f"Not an AZ variant: {variant}")


def az_regularization_weights(cfg: Dict[str, Any]) -> Dict[str, float]:
    return {
        "membership_entropy": float(cfg.get("reg_membership_entropy", 0.0)),
        "membership_smoothness": float(cfg.get("reg_membership_smoothness", 0.0)),
        "geometry_smoothness": float(cfg.get("reg_geometry_smoothness", 0.0)),
        "hyperbolicity_penalty": float(cfg.get("reg_hyperbolicity", 0.0)),
    }


def build_model(
    variant: str,
    num_outputs: int,
    in_channels: int,
    num_rules: int = 4,
    task: str = "classification",
) -> nn.Module:
    if task == "classification":
        if variant == "baseline":
            return StandardConvNet(num_classes=num_outputs, in_channels=in_channels)
        if variant.startswith("az_"):
            cfg = az_config_for_variant(variant)
            return AZConvNet(num_classes=num_outputs, in_channels=in_channels, num_rules=num_rules, cfg=cfg)
    elif task == "segmentation":
        if variant == "baseline":
            return BaselineUNet(in_channels=in_channels, out_channels=num_outputs)
        if variant == "az_sota":
            cfg = az_config_for_variant(variant)
            return AZSOTAUNet(in_channels=in_channels, out_channels=num_outputs, num_rules=num_rules, cfg=cfg)
        if variant in {"az_sota_pure", "az_thesis"}:
            cfg = az_config_for_variant(variant)
            return AZSOTAUNet(
                in_channels=in_channels,
                out_channels=num_outputs,
                num_rules=num_rules,
                cfg=cfg,
                pure_az=True,
            )
        if variant.startswith("az_"):
            cfg = az_config_for_variant(variant)
            return AZUNet(in_channels=in_channels, out_channels=num_outputs, num_rules=num_rules, cfg=cfg)
    raise ValueError(f"Unknown variant/task combination: {variant}, {task}")


@torch.no_grad()
def sanity_check_azconv_forward(device: torch.device) -> None:
    from models.azconv import AZConv2d

    batch, channels, height, width, rules, kernel_size, out_channels = 2, 8, 32, 32, 4, 3, 16
    x = torch.randn(batch, channels, height, width, device=device)
    layer = AZConv2d(channels, out_channels, kernel_size=kernel_size, num_rules=rules).to(device)
    y = layer(x)
    assert y.shape == (batch, out_channels, height, width), y.shape
    assert torch.isfinite(y).all(), "non-finite outputs"


@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    in_channels: int,
    spatial_shape: tuple[int, int],
    warmup: int = 3,
    iters: int = 20,
) -> float:
    model.eval()
    height, width = spatial_shape
    x = torch.randn(batch_size, in_channels, height, width, device=device)
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


def spatial_shape_for_dataset(name: str) -> tuple[int, int]:
    n = name.lower().replace("-", "_")
    if n in ("fashion_mnist", "fashionmnist"):
        return (28, 28)
    if n in ("cifar10", "cifar_10"):
        return (32, 32)
    if n == "drive":
        return (584, 565)
    raise ValueError(f"Unknown dataset: {name}")


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    loss = loss * valid_mask
    denom = valid_mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs * valid_mask
    target = target * valid_mask
    numerator = 2.0 * (probs * target).sum()
    denominator = probs.sum() + target.sum()
    return 1.0 - (numerator + eps) / (denominator + eps)


def soft_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits) * valid_mask
    target = target * valid_mask

    tp = (probs * target).sum()
    fp = (probs * (1.0 - target)).sum()
    fn = ((1.0 - probs) * target).sum()
    score = (tp + eps) / (tp + float(alpha) * fp + float(beta) * fn + eps)
    return 1.0 - score


def _soft_erode(mask: torch.Tensor) -> torch.Tensor:
    return torch.minimum(
        -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1)),
    )


def _soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    return torch.maximum(
        F.max_pool2d(mask, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        F.max_pool2d(mask, kernel_size=(1, 3), stride=1, padding=(0, 1)),
    )


def _soft_open(mask: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(mask))


def _soft_skeletonize(mask: torch.Tensor, num_iters: int = 10) -> torch.Tensor:
    if num_iters <= 0:
        return mask
    img = mask
    opened = _soft_open(img)
    skeleton = F.relu(img - opened)
    for _ in range(num_iters - 1):
        img = _soft_erode(img)
        opened = _soft_open(img)
        delta = F.relu(img - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    num_iters: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits) * valid_mask
    target = target * valid_mask

    skel_pred = _soft_skeletonize(probs, num_iters=num_iters)
    skel_target = _soft_skeletonize(target, num_iters=num_iters)

    topological_precision = (skel_pred * target).sum() / (skel_pred.sum() + eps)
    topological_sensitivity = (skel_target * probs).sum() / (skel_target.sum() + eps)
    cl_dice = (2.0 * topological_precision * topological_sensitivity + eps) / (
        topological_precision + topological_sensitivity + eps
    )
    return 1.0 - cl_dice


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    pos_weight: torch.Tensor | None = None,
    overlap_mode: str = "dice",
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
) -> tuple[torch.Tensor, Dict[str, float]]:
    bce = masked_bce_with_logits(logits, target, valid_mask, pos_weight=pos_weight)
    mode = str(overlap_mode).lower().strip()
    if mode == "dice":
        overlap = soft_dice_loss(logits, target, valid_mask)
    elif mode == "tversky":
        overlap = soft_tversky_loss(
            logits,
            target,
            valid_mask,
            alpha=tversky_alpha,
            beta=tversky_beta,
        )
    else:
        raise ValueError(f"Unknown overlap_mode: {overlap_mode}")
    loss = bce_weight * bce + dice_weight * overlap
    return loss, {"bce_loss": float(bce.detach()), "dice_loss": float(overlap.detach())}


def unpack_segmentation_outputs(output: torch.Tensor | Dict[str, Any]) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor | None]:
    if isinstance(output, dict):
        if "logits" in output:
            main_logits = output["logits"]
        elif "main_logits" in output:
            main_logits = output["main_logits"]
        else:
            raise KeyError("Segmentation dict output must contain 'logits' or 'main_logits'.")
        aux = output.get("aux_logits", [])
        aux_logits = [aux] if isinstance(aux, torch.Tensor) else list(aux)
        boundary_logits = output.get("boundary_logits")
        return main_logits, aux_logits, boundary_logits
    return output, [], None


def boundary_target_from_mask(mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded > 0.01).float()
    return boundary * valid_mask


def segmentation_objective(
    output: torch.Tensor | Dict[str, Any],
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    pos_weight: torch.Tensor | None = None,
    overlap_mode: str = "dice",
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
    aux_weight: float = 0.0,
    boundary_weight: float = 0.0,
    topology_weight: float = 0.0,
    topology_num_iters: int = 10,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    main_logits, aux_logits, boundary_logits = unpack_segmentation_outputs(output)
    main_loss, main_aux = segmentation_loss(
        main_logits,
        target,
        valid_mask,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
        overlap_mode=overlap_mode,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
    )
    total = main_loss
    aux_loss_value = main_logits.new_zeros(())
    boundary_loss_value = main_logits.new_zeros(())
    topology_loss_value = main_logits.new_zeros(())

    if aux_logits and aux_weight > 0.0:
        aux_losses = []
        for aux_logits_i in aux_logits:
            aux_loss_i, _ = segmentation_loss(
                aux_logits_i,
                target,
                valid_mask,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
                pos_weight=pos_weight,
                overlap_mode=overlap_mode,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
            )
            aux_losses.append(aux_loss_i)
        aux_loss_value = torch.stack(aux_losses).mean()
        total = total + aux_weight * aux_loss_value

    if boundary_logits is not None and boundary_weight > 0.0:
        boundary_target = boundary_target_from_mask(target, valid_mask)
        boundary_loss_value = masked_bce_with_logits(boundary_logits, boundary_target, valid_mask)
        total = total + boundary_weight * boundary_loss_value

    if topology_weight > 0.0:
        topology_loss_value = soft_cldice_loss(
            main_logits,
            target,
            valid_mask,
            num_iters=int(topology_num_iters),
        )
        total = total + topology_weight * topology_loss_value

    logs = {
        "bce_loss": main_aux["bce_loss"],
        "dice_loss": main_aux["dice_loss"],
        "aux_loss": float(aux_loss_value.detach()),
        "boundary_loss": float(boundary_loss_value.detach()),
        "topology_loss": float(topology_loss_value.detach()),
    }
    return total, logs, main_logits


def binary_confusion_counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float, float, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    mask = valid_mask > 0.5
    pred = pred[mask]
    target = target[mask]

    tp = float(((pred == 1) & (target == 1)).sum().item())
    fp = float(((pred == 1) & (target == 0)).sum().item())
    tn = float(((pred == 0) & (target == 0)).sum().item())
    fn = float(((pred == 0) & (target == 1)).sum().item())
    return tp, fp, tn, fn


def segmentation_metrics_from_counts(tp: float, fp: float, tn: float, fn: float, eps: float = 1e-8) -> Dict[str, float]:
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
    }


def build_threshold_grid(start: float, end: float, step: float) -> List[float]:
    if step <= 0.0:
        raise ValueError("Threshold step must be positive.")
    if end < start:
        raise ValueError("Threshold end must be >= start.")

    thresholds: List[float] = []
    current = float(start)
    while current <= end + 1e-12:
        thresholds.append(round(current, 6))
        current += step
    return thresholds


@torch.no_grad()
def sweep_segmentation_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: Sequence[float],
) -> List[Dict[str, float]]:
    if not thresholds:
        return []

    model.eval()
    counts = {float(thr): [0.0, 0.0, 0.0, 0.0] for thr in thresholds}

    for x, y, valid_mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        output = model(x)
        main_logits, _, _ = unpack_segmentation_outputs(output)
        probs = torch.sigmoid(main_logits)
        valid = valid_mask > 0.5
        probs_valid = probs[valid]
        target_valid = y[valid]

        for thr in thresholds:
            pred = probs_valid >= float(thr)
            tp = float(((pred == 1) & (target_valid == 1)).sum().item())
            fp = float(((pred == 1) & (target_valid == 0)).sum().item())
            tn = float(((pred == 0) & (target_valid == 0)).sum().item())
            fn = float(((pred == 0) & (target_valid == 1)).sum().item())
            c = counts[float(thr)]
            c[0] += tp
            c[1] += fp
            c[2] += tn
            c[3] += fn

    rows: List[Dict[str, float]] = []
    for thr in thresholds:
        tp, fp, tn, fn = counts[float(thr)]
        metrics = segmentation_metrics_from_counts(tp, fp, tn, fn)
        rows.append({"threshold": float(thr), **metrics})
    return rows


def select_best_threshold(
    sweep_rows: Sequence[Dict[str, float]],
    metric: str = "dice",
    reference_threshold: float = 0.5,
) -> Dict[str, float]:
    if not sweep_rows:
        raise ValueError("Threshold sweep rows must not be empty.")

    return max(
        sweep_rows,
        key=lambda row: (
            threshold_metric_value(row, metric),
            -abs(float(row["threshold"]) - float(reference_threshold)),
            -float(row["threshold"]),
        ),
    )


def collect_drive_metrics_records(results_dir: str | Path) -> List[Dict[str, Any]]:
    results_path = Path(results_dir)
    records: List[Dict[str, Any]] = []

    for metrics_path in sorted(results_path.glob("*/metrics.json")):
        run_name = metrics_path.parent.name
        if "smoke" in run_name.lower():
            continue
        try:
            with open(metrics_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        if data.get("dataset") != "drive" or data.get("task") != "segmentation":
            continue
        if float(data.get("test_dice", 0.0)) <= 1e-6:
            continue

        records.append(
            {
                "variant": str(data.get("variant", "unknown")),
                "run_name": str(data.get("run_name", run_name)),
                "seed": data.get("seed"),
                "num_rules": data.get("num_rules"),
                "topology_loss_weight": float(data.get("topology_loss_weight", 0.0)),
                "aux_loss_weight": data.get("aux_loss_weight"),
                "boundary_loss_weight": data.get("boundary_loss_weight"),
                "selected_threshold": data.get("selected_threshold"),
                "protocol_rank": 1 if data.get("selected_threshold") is not None else 0,
                "test_dice": float(data.get("test_dice", 0.0)),
                "test_iou": float(data.get("test_iou", 0.0)),
                "test_precision": float(data.get("test_precision", 0.0)),
                "test_recall": float(data.get("test_recall", 0.0)),
                "test_specificity": float(data.get("test_specificity", 0.0)),
                "test_accuracy": float(data.get("test_accuracy", 0.0)),
                "test_balanced_accuracy": float(data.get("test_balanced_accuracy", 0.0)),
                "seconds_per_forward_batch": float(data.get("seconds_per_forward_batch", 0.0)),
                "metrics_path": str(metrics_path),
            }
        )

    return records


def drive_record_selection_key(record: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(record["protocol_rank"]),
        float(record["test_dice"]),
        float(record["test_iou"]),
        -float(record["seconds_per_forward_batch"]),
    )


def select_best_drive_record(
    records: Sequence[Dict[str, Any]],
    variant: str,
    topology_loss_weight: float | None = None,
) -> Dict[str, Any]:
    filtered: List[Dict[str, Any]] = []
    for record in records:
        if str(record.get("variant")) != variant:
            continue
        if topology_loss_weight is not None and abs(float(record.get("topology_loss_weight", 0.0)) - float(topology_loss_weight)) > 1e-9:
            continue
        filtered.append(dict(record))

    if not filtered:
        topo_note = "" if topology_loss_weight is None else f" with topology_loss_weight={topology_loss_weight}"
        raise ValueError(f"No DRIVE run found for variant '{variant}'{topo_note}.")
    return max(filtered, key=drive_record_selection_key)


def build_drive_superiority_report(
    candidate_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    metric_names: Sequence[str] = DRIVE_SUPERIORITY_METRICS,
    min_delta: float = 0.0,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for metric in metric_names:
        if metric not in candidate_metrics:
            raise KeyError(f"Candidate metrics do not contain '{metric}'.")
        if metric not in baseline_metrics:
            raise KeyError(f"Baseline metrics do not contain '{metric}'.")
        candidate_value = float(candidate_metrics[metric])
        baseline_value = float(baseline_metrics[metric])
        delta = candidate_value - baseline_value
        rows.append(
            {
                "metric": metric,
                "label": DRIVE_METRIC_LABELS.get(metric, metric),
                "candidate": candidate_value,
                "baseline": baseline_value,
                "delta": delta,
                "passed": bool(delta > float(min_delta)),
            }
        )

    return {
        "candidate_variant": str(candidate_metrics.get("variant", "unknown")),
        "candidate_run_name": str(candidate_metrics.get("run_name", "current_run")),
        "baseline_variant": str(baseline_metrics.get("variant", "baseline")),
        "baseline_run_name": str(baseline_metrics.get("run_name", "baseline")),
        "min_delta": float(min_delta),
        "metric_rows": rows,
        "all_passed": all(row["passed"] for row in rows),
    }


def build_drive_threshold_search_report(
    sweep_rows: Sequence[Dict[str, float]],
    baseline_metrics: Dict[str, Any],
    metric_names: Sequence[str] = DRIVE_SUPERIORITY_METRICS,
    min_delta: float = 0.0,
    selection_metric: str = "dice",
) -> Dict[str, Any]:
    if not sweep_rows:
        raise ValueError("Threshold sweep rows must not be empty.")

    rows: List[Dict[str, Any]] = []
    passing_rows: List[Dict[str, Any]] = []
    for row in sweep_rows:
        candidate_metrics = {
            metric_name: float(row[_strip_metric_prefix(metric_name)])
            for metric_name in metric_names
        }
        report = build_drive_superiority_report(
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
            metric_names=metric_names,
            min_delta=min_delta,
        )
        row_report = {
            "threshold": float(row["threshold"]),
            "selection_value": threshold_metric_value(row, selection_metric),
            "metric_rows": report["metric_rows"],
            "all_passed": bool(report["all_passed"]),
        }
        rows.append(row_report)
        if row_report["all_passed"]:
            passing_rows.append(row_report)

    best_row = None
    if passing_rows:
        best_row = max(
            passing_rows,
            key=lambda item: (
                float(item["selection_value"]),
                -abs(float(item["threshold"]) - 0.5),
                -float(item["threshold"]),
            ),
        )

    return {
        "selection_metric": selection_metric,
        "min_delta": float(min_delta),
        "rows": rows,
        "passed_count": len(passing_rows),
        "best_row": best_row,
        "all_passed": bool(passing_rows),
    }


def compare_drive_metrics_to_baseline(
    results_dir: str | Path,
    candidate_metrics: Dict[str, Any],
    metric_names: Sequence[str] = DRIVE_SUPERIORITY_METRICS,
    baseline_variant: str = "baseline",
    min_delta: float = 0.0,
) -> Dict[str, Any]:
    records = collect_drive_metrics_records(results_dir)
    baseline_metrics = select_best_drive_record(records, variant=baseline_variant)
    return build_drive_superiority_report(
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        metric_names=metric_names,
        min_delta=min_delta,
    )


def compare_drive_variant_to_baseline(
    results_dir: str | Path,
    candidate_variant: str,
    metric_names: Sequence[str] = DRIVE_SUPERIORITY_METRICS,
    baseline_variant: str = "baseline",
    topology_loss_weight: float | None = None,
    min_delta: float = 0.0,
) -> Dict[str, Any]:
    records = collect_drive_metrics_records(results_dir)
    candidate_metrics = select_best_drive_record(records, variant=candidate_variant, topology_loss_weight=topology_loss_weight)
    baseline_metrics = select_best_drive_record(records, variant=baseline_variant)
    return build_drive_superiority_report(
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        metric_names=metric_names,
        min_delta=min_delta,
    )


def format_drive_superiority_report(report: Dict[str, Any]) -> str:
    header = (
        "Drive superiority gate: "
        f"{report['candidate_variant']} ({report['candidate_run_name']}) "
        f"vs {report['baseline_variant']} ({report['baseline_run_name']})"
    )
    status = "PASS" if report["all_passed"] else "FAIL"
    lines = [f"{header} -> {status}"]
    for row in report["metric_rows"]:
        marker = "OK" if row["passed"] else "NO"
        lines.append(
            f"  - {row['label']}: candidate={row['candidate']:.4f}, "
            f"baseline={row['baseline']:.4f}, delta={row['delta']:+.4f} [{marker}]"
        )
    return "\n".join(lines)


def format_drive_threshold_search_report(report: Dict[str, Any]) -> str:
    if not report["all_passed"]:
        return "Threshold-dominance search -> FAIL (no threshold in the declared grid beats baseline on all gated metrics)"

    best_row = report["best_row"]
    header = (
        "Threshold-dominance search -> PASS "
        f"(best threshold={float(best_row['threshold']):.3f}, "
        f"{report['selection_metric']}={float(best_row['selection_value']):.4f})"
    )
    lines = [header]
    for row in best_row["metric_rows"]:
        lines.append(
            f"  - {row['label']}: candidate={row['candidate']:.4f}, "
            f"baseline={row['baseline']:.4f}, delta={row['delta']:+.4f} [OK]"
        )
    return "\n".join(lines)


def update_drive_comparison_summary(results_dir: str | Path) -> Path:
    results_path = Path(results_dir)
    records = collect_drive_metrics_records(results_path)

    out_path = results_path / "drive_real_comparison.md"
    if not records:
        out_path.write_text(
            "No DRIVE segmentation runs with metrics.json were found.\n",
            encoding="utf-8",
        )
        return out_path

    best_by_condition: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for record in records:
        key = (record["variant"], round(float(record["topology_loss_weight"]), 6))
        current = best_by_condition.get(key)
        if current is None:
            best_by_condition[key] = record
            continue
        if drive_record_selection_key(record) > drive_record_selection_key(current):
            best_by_condition[key] = record

    ordered = sorted(
        best_by_condition.values(),
        key=lambda row: (-row["test_dice"], row["variant"], row["topology_loss_weight"]),
    )

    lines = [
        "| Variant | Run | Seed | Rules | Aux w | Boundary w | Topology w | Threshold | Test Dice | Test IoU | Precision | Recall | Specificity | Balanced Acc | Fwd batch (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        threshold = row["selected_threshold"]
        threshold_text = f"{float(threshold):.3f}" if threshold is not None else "n/a"
        seed = row["seed"] if row["seed"] is not None else "n/a"
        num_rules = row["num_rules"]
        rules_text = str(num_rules) if num_rules is not None else "n/a"
        aux_weight = row["aux_loss_weight"]
        aux_text = f"{float(aux_weight):.4f}" if aux_weight is not None else "n/a"
        boundary_weight = row["boundary_loss_weight"]
        boundary_text = f"{float(boundary_weight):.4f}" if boundary_weight is not None else "n/a"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    row["run_name"],
                    str(seed),
                    rules_text,
                    aux_text,
                    boundary_text,
                    f"{row['topology_loss_weight']:.4f}",
                    threshold_text,
                    f"{row['test_dice']:.4f}",
                    f"{row['test_iou']:.4f}",
                    f"{row['test_precision']:.4f}",
                    f"{row['test_recall']:.4f}",
                    f"{row['test_specificity']:.4f}",
                    f"{row['test_balanced_accuracy']:.4f}",
                    f"{row['seconds_per_forward_batch']:.5f}",
                ]
            )
            + " |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
