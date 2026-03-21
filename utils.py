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
    crop_size = cfg.get("drive_patch_size")
    crop_size = int(crop_size) if crop_size is not None else None
    foreground_bias = float(cfg.get("drive_foreground_bias", 0.0))

    train_set = DriveDataset(
        data_root,
        split="training",
        augment=True,
        use_fov_mask=use_fov_mask,
        crop_size=crop_size,
        foreground_bias=foreground_bias,
    )
    val_set = DriveDataset(data_root, split="training", augment=False, use_fov_mask=use_fov_mask)
    test_set = DriveDataset(data_root, split="test", augment=False, use_fov_mask=use_fov_mask)

    val_fraction = float(cfg.get("val_fraction", 0.2))
    n_train = len(train_set)
    n_val = max(1, int(round(n_train * val_fraction)))
    n_val = min(n_val, n_train - 1) if n_train > 1 else 1
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    perm = torch.randperm(n_train, generator=generator).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if not train_idx:
        train_idx = val_idx

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
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
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
    """Shape / finiteness checks for AZConv2d."""

    from models.azconv import AZConv2d

    B, C, H, W, R, k, out_c = 2, 8, 32, 32, 4, 3, 16
    x = torch.randn(B, C, H, W, device=device)
    layer = AZConv2d(C, out_c, kernel_size=k, num_rules=R).to(device)
    y = layer(x)
    assert y.shape == (B, out_c, H, W), y.shape
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
    """Mean seconds per forward pass over `iters` batches after warmup."""

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


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    bce = masked_bce_with_logits(logits, target, valid_mask, pos_weight=pos_weight)
    dice = soft_dice_loss(logits, target, valid_mask)
    loss = bce_weight * bce + dice_weight * dice
    return loss, {"bce_loss": float(bce.detach()), "dice_loss": float(dice.detach())}


def unpack_segmentation_outputs(output: torch.Tensor | Dict[str, Any]) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor | None]:
    if isinstance(output, dict):
        if "logits" in output:
            main_logits = output["logits"]
        elif "main_logits" in output:
            main_logits = output["main_logits"]
        else:
            raise KeyError("Segmentation dict output must contain 'logits' or 'main_logits'.")
        aux = output.get("aux_logits", [])
        if isinstance(aux, torch.Tensor):
            aux_logits = [aux]
        else:
            aux_logits = list(aux)
        boundary_logits = output.get("boundary_logits")
        return main_logits, aux_logits, boundary_logits
    return output, [], None


def boundary_target_from_mask(mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    # Morphological gradient approximates vessel boundaries while staying differentiable-friendly.
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
    aux_weight: float = 0.0,
    boundary_weight: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    main_logits, aux_logits, boundary_logits = unpack_segmentation_outputs(output)
    main_loss, main_aux = segmentation_loss(
        main_logits,
        target,
        valid_mask,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
    )
    total = main_loss
    aux_loss_value = main_logits.new_zeros(())
    boundary_loss_value = main_logits.new_zeros(())

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
            )
            aux_losses.append(aux_loss_i)
        aux_loss_value = torch.stack(aux_losses).mean()
        total = total + aux_weight * aux_loss_value

    if boundary_logits is not None and boundary_weight > 0.0:
        boundary_target = boundary_target_from_mask(target, valid_mask)
        boundary_loss_value = masked_bce_with_logits(boundary_logits, boundary_target, valid_mask)
        total = total + boundary_weight * boundary_loss_value

    logs = {
        "bce_loss": main_aux["bce_loss"],
        "dice_loss": main_aux["dice_loss"],
        "aux_loss": float(aux_loss_value.detach()),
        "boundary_loss": float(boundary_loss_value.detach()),
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


def _resolve_drive_subset(dataset: Dataset) -> tuple[DriveDataset, List[int]]:
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
        if not isinstance(base_dataset, DriveDataset):
            raise TypeError("Expected a Subset wrapping DriveDataset.")
        return base_dataset, list(dataset.indices)
    if isinstance(dataset, DriveDataset):
        return dataset, list(range(len(dataset.samples)))
    raise TypeError("Expected DriveDataset or Subset[DriveDataset].")


def estimate_drive_pos_weight(
    dataset: Dataset,
    min_weight: float = 1.0,
    max_weight: float = 25.0,
) -> float:
    base_dataset, indices = _resolve_drive_subset(dataset)
    positives = 0.0
    negatives = 0.0

    for idx in indices:
        _, mask_path, fov_path = base_dataset.samples[idx]
        mask = base_dataset._load_mask(mask_path)
        fov = base_dataset._load_mask(fov_path) if base_dataset.use_fov_mask else torch.ones_like(mask)
        valid = fov > 0.5
        positives += float(((mask > 0.5) & valid).sum().item())
        negatives += float(((mask <= 0.5) & valid).sum().item())

    if positives <= 0.0:
        return float(max_weight)
    weight = negatives / positives
    return float(max(min_weight, min(max_weight, weight)))


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
    az_key = "az_cat" if "az_cat" in by_var else "az_full"
    az = by_var.get(az_key)
    is_seg = any("test_dice" in row for row in rows)

    lines = [
        "Interpretation (prototype - not a statistical claim)",
        "=" * 60,
        "",
        "Does AZConv help?",
    ]

    if base and az:
        if is_seg:
            bscore = float(base.get("test_dice", 0.0))
            ascore = float(az.get("test_dice", 0.0))
            delta = ascore - bscore
            lines.append(f"- Test Dice: baseline {bscore:.4f} vs {az_key} {ascore:.4f} (delta = {delta:+.4f}).")
        else:
            bscore = float(base.get("test_accuracy", 0.0))
            ascore = float(az.get("test_accuracy", 0.0))
            delta = ascore - bscore
            lines.append(f"- Test accuracy: baseline {bscore:.2f}% vs {az_key} {ascore:.2f}% (delta = {delta:+.2f} pp).")

        if abs(delta) < (0.005 if is_seg else 0.5):
            lines.append("- On this setup, the two models are likely within noise; run with more epochs/seeds.")
        elif delta > 0:
            lines.append(f"- {az_key} is ahead; directional structure may match the task.")
        else:
            lines.append("- Baseline is ahead; the inductive bias may be too rigid or under-optimized.")
    else:
        lines.append("- Run all variants so we can compute baseline vs the AZ reference variant.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path
