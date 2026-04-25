"""Shared utilities for classification and DRIVE-style segmentation experiments."""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset

from models import AZConv2d, AZConvConfig, AZConvNet, AZSOTAUNet, AZUNet, AttentionUNet, BaselineUNet, StandardConvNet, count_parameters

VARIANTS = [
    "baseline",
    "attention_unet",
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

ARTICLE_DRIVE_VARIANTS = [
    "baseline",
    "attention_unet",
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

DRIVE_MULTI_SEED_METRICS = [
    "test_dice",
    "test_iou",
    "test_precision",
    "test_recall",
    "test_specificity",
    "test_balanced_accuracy",
    "selected_threshold",
    "seconds_per_forward_batch",
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

RETINAL_SEG_DATASETS = {"drive", "chase_db1", "chasedb1", "chase", "fives"}
ARCADE_SEG_DATASETS = {"arcade", "arcade_syntax", "arcade_stenosis"}


def segmentation_tta_num_views(mode: str | None) -> int:
    tta_mode = str(mode or "none").lower().strip()
    if tta_mode in {"", "none", "off"}:
        return 1
    if tta_mode == "flips":
        return 4
    if tta_mode == "d4":
        return 8
    raise ValueError(f"Unknown eval_tta mode: {mode}")


def _segmentation_tta_ops(mode: str | None):
    tta_mode = str(mode or "none").lower().strip()
    if tta_mode in {"", "none", "off"}:
        return [("id", lambda x: x, lambda y: y)]
    if tta_mode == "flips":
        return [
            ("id", lambda x: x, lambda y: y),
            ("h", lambda x: torch.flip(x, dims=[3]), lambda y: torch.flip(y, dims=[3])),
            ("v", lambda x: torch.flip(x, dims=[2]), lambda y: torch.flip(y, dims=[2])),
            ("hv", lambda x: torch.flip(x, dims=[2, 3]), lambda y: torch.flip(y, dims=[2, 3])),
        ]
    if tta_mode == "d4":
        return [
            ("id", lambda x: x, lambda y: y),
            ("h", lambda x: torch.flip(x, dims=[3]), lambda y: torch.flip(y, dims=[3])),
            ("v", lambda x: torch.flip(x, dims=[2]), lambda y: torch.flip(y, dims=[2])),
            ("hv", lambda x: torch.flip(x, dims=[2, 3]), lambda y: torch.flip(y, dims=[2, 3])),
            ("t", lambda x: x.transpose(2, 3), lambda y: y.transpose(2, 3)),
            ("th", lambda x: torch.flip(x.transpose(2, 3), dims=[3]), lambda y: torch.flip(y, dims=[3]).transpose(2, 3)),
            ("tv", lambda x: torch.flip(x.transpose(2, 3), dims=[2]), lambda y: torch.flip(y, dims=[2]).transpose(2, 3)),
            (
                "thv",
                lambda x: torch.flip(x.transpose(2, 3), dims=[2, 3]),
                lambda y: torch.flip(y, dims=[2, 3]).transpose(2, 3),
            ),
        ]
    raise ValueError(f"Unknown eval_tta mode: {mode}")


class SegmentationTTAWrapper(nn.Module):
    def __init__(self, model: nn.Module, mode: str = "none") -> None:
        super().__init__()
        self.model = model
        self.mode = str(mode).lower().strip()
        self.ops = _segmentation_tta_ops(self.mode)

    def _main_logits(self, output: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        if isinstance(output, dict):
            if "logits" in output:
                return output["logits"]
            if "main_logits" in output:
                return output["main_logits"]
            raise KeyError("Segmentation dict output must contain 'logits' or 'main_logits'.")
        return output

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor] | torch.Tensor:
        if self.mode in {"", "none", "off"}:
            return self.model(x)

        logits = []
        for _, forward_op, inverse_op in self.ops:
            output = self.model(forward_op(x))
            logits.append(inverse_op(self._main_logits(output)))
        return {"logits": torch.stack(logits, dim=0).mean(dim=0)}


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


def canonical_dataset_name(name: str) -> str:
    return str(name).lower().replace("-", "_")


def retinal_dataset_dirname(name: str) -> str:
    n = canonical_dataset_name(name)
    if n == "drive":
        return "DRIVE"
    if n in {"chase_db1", "chasedb1", "chase"}:
        return "CHASE_DB1"
    if n == "fives":
        return "FIVES"
    raise ValueError(f"Unknown retinal dataset: {name}")


def retinal_dataset_root(data_root: str | Path, name: str) -> Path:
    return Path(data_root) / retinal_dataset_dirname(name)


def arcade_dataset_root(data_root: str | Path) -> Path:
    return Path(data_root) / "ARCADE"


def _retinal_sample_id(path: Path) -> str:
    stem = path.stem
    for suffix in ("_manual1", "_training_mask", "_test_mask", "_mask", "_1stHO", "_2ndHO"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for split_suffix in ("_training", "_test"):
        if stem.endswith(split_suffix):
            stem = stem[: -len(split_suffix)]
            break
    return stem


def task_for_dataset(name: str) -> str:
    n = canonical_dataset_name(name)
    if n in ("cifar10", "cifar_10", "fashion_mnist", "fashionmnist"):
        return "classification"
    if n in RETINAL_SEG_DATASETS or n in ARCADE_SEG_DATASETS:
        return "segmentation"
    raise ValueError(f"Unknown dataset: {name}")


def dataset_channels_and_outputs(name: str) -> Tuple[int, int]:
    n = canonical_dataset_name(name)
    if n in ("cifar10", "cifar_10"):
        return 3, 10
    if n in ("fashion_mnist", "fashionmnist"):
        return 1, 10
    if n in RETINAL_SEG_DATASETS or n in ARCADE_SEG_DATASETS:
        return 3, 1
    raise ValueError(f"Unknown dataset: {name}")


class DriveDataset(Dataset):
    """Loader for normalized retinal vessel segmentation datasets."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        augment: bool = False,
        use_fov_mask: bool = True,
        crop_size: int | None = None,
        foreground_bias: float = 0.0,
        thin_vessel_bias: float = 0.0,
        thin_vessel_neighbor_threshold: int = 4,
        hard_mining_dir: str | Path | None = None,
        hard_mining_bias: float = 0.0,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0,
        gamma_jitter: float = 0.0,
        input_mode: str = "rgb",
        green_blend_alpha: float = 0.35,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.use_fov_mask = use_fov_mask
        self.crop_size = crop_size
        self.foreground_bias = float(foreground_bias)
        self.thin_vessel_bias = max(0.0, min(1.0, float(thin_vessel_bias)))
        self.thin_vessel_neighbor_threshold = max(1, min(9, int(thin_vessel_neighbor_threshold)))
        self.hard_mining_dir = Path(hard_mining_dir) if hard_mining_dir else None
        self.hard_mining_bias = max(0.0, min(1.0, float(hard_mining_bias)))
        self._hard_mining_cache: Dict[str, torch.Tensor | None] = {}
        self.brightness_jitter = max(0.0, float(brightness_jitter))
        self.contrast_jitter = max(0.0, float(contrast_jitter))
        self.gamma_jitter = max(0.0, float(gamma_jitter))
        self.input_mode = str(input_mode).lower().strip()
        if self.input_mode not in {"rgb", "green", "green_equalized", "green_hybrid"}:
            raise ValueError(
                f"Unknown retinal input_mode '{input_mode}'. Expected one of ['rgb', 'green', 'green_equalized', 'green_hybrid']."
            )
        self.green_blend_alpha = max(0.0, min(1.0, float(green_blend_alpha)))
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Tuple[Path, Path, Path]]:
        split_dir = self.root / self.split
        images_dir = split_dir / "images"
        manual_dir = split_dir / "1st_manual"
        mask_dir = split_dir / "mask"
        if not images_dir.exists() or not manual_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                "Expected retinal vessel structure under "
                f"{split_dir} with subfolders images/, 1st_manual/, mask/."
            )

        def by_id(paths: Sequence[Path]) -> Dict[str, Path]:
            out: Dict[str, Path] = {}
            for path in paths:
                sample_id = _retinal_sample_id(path)
                out[sample_id] = path
            return out

        image_map = by_id(sorted(images_dir.glob("*.*")))
        manual_map = by_id(sorted(manual_dir.glob("*.*")))
        mask_map = by_id(sorted(mask_dir.glob("*.*")))
        common_ids = sorted(set(image_map) & set(manual_map) & set(mask_map))
        if not common_ids:
            raise FileNotFoundError(f"No matched retinal vessel samples found under {split_dir}")
        return [(image_map[idx], manual_map[idx], mask_map[idx]) for idx in common_ids]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        if self.input_mode == "green":
            green = np.asarray(image.getchannel("G"), dtype=np.float32)[..., None]
            arr = np.repeat(green, 3, axis=2) / 255.0
        elif self.input_mode == "green_equalized":
            green = ImageOps.equalize(image.getchannel("G"))
            green_arr = np.asarray(green, dtype=np.float32)[..., None]
            arr = np.repeat(green_arr, 3, axis=2) / 255.0
        elif self.input_mode == "green_hybrid":
            arr = np.asarray(image, dtype=np.float32) / 255.0
            green_eq = np.asarray(ImageOps.equalize(image.getchannel("G")), dtype=np.float32) / 255.0
            alpha = self.green_blend_alpha
            side_alpha = 0.5 * alpha
            arr[..., 0] = arr[..., 0] * (1.0 - side_alpha) + green_eq * side_alpha
            arr[..., 1] = arr[..., 1] * (1.0 - alpha) + green_eq * alpha
            arr[..., 2] = arr[..., 2] * (1.0 - side_alpha) + green_eq * side_alpha
        else:
            arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def set_foreground_bias(self, value: float) -> None:
        self.foreground_bias = max(0.0, min(1.0, float(value)))

    def set_thin_vessel_bias(self, value: float) -> None:
        self.thin_vessel_bias = max(0.0, min(1.0, float(value)))

    def set_hard_mining_bias(self, value: float) -> None:
        self.hard_mining_bias = max(0.0, min(1.0, float(value)))

    def _load_mask(self, path: Path) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        arr = (np.asarray(mask, dtype=np.float32) > 127.0).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
        return (image - mean) / std

    def _thin_vessel_candidates(self, mask: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
        vessel = ((mask > 0.5) & (fov > 0.5)).float()
        if not vessel.any():
            return torch.zeros_like(mask, dtype=torch.bool)
        kernel = torch.ones((1, 1, 3, 3), dtype=vessel.dtype, device=vessel.device)
        counts = F.conv2d(vessel.unsqueeze(0), kernel, padding=1)[0]
        return (counts <= float(self.thin_vessel_neighbor_threshold)) & (vessel > 0.5)

    @staticmethod
    def _sample_weighted_anchor(
        weight_map: torch.Tensor,
        crop_h: int,
        crop_w: int,
        max_top: int,
        max_left: int,
    ) -> tuple[int, int] | None:
        weights = weight_map[0].reshape(-1)
        total = float(weights.sum().item())
        if total <= 0.0:
            return None
        probs = (weights / total).cpu()
        pick = int(torch.multinomial(probs, num_samples=1).item())
        width = int(weight_map.shape[2])
        center_y = pick // width
        center_x = pick % width
        top = min(max(center_y - crop_h // 2, 0), max_top)
        left = min(max(center_x - crop_w // 2, 0), max_left)
        return top, left

    @staticmethod
    def _sample_crop_anchor(
        candidate_mask: torch.Tensor,
        crop_h: int,
        crop_w: int,
        max_top: int,
        max_left: int,
    ) -> tuple[int, int] | None:
        if not candidate_mask.any():
            return None
        ys, xs = torch.nonzero(candidate_mask[0], as_tuple=True)
        pick = random.randint(0, ys.numel() - 1)
        center_y = int(ys[pick].item())
        center_x = int(xs[pick].item())
        top = min(max(center_y - crop_h // 2, 0), max_top)
        left = min(max(center_x - crop_w // 2, 0), max_left)
        return top, left

    def _load_hard_mining_map(self, sample_id: str, reference: torch.Tensor) -> torch.Tensor | None:
        if self.hard_mining_dir is None:
            return None
        cached = self._hard_mining_cache.get(sample_id, None)
        if sample_id in self._hard_mining_cache:
            return None if cached is None else cached.clone()

        candidate = None
        for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pt", ".npy"):
            path = self.hard_mining_dir / f"{sample_id}{suffix}"
            if path.exists():
                candidate = path
                break
        if candidate is None:
            self._hard_mining_cache[sample_id] = None
            return None

        if candidate.suffix.lower() == ".pt":
            tensor = torch.load(candidate, map_location="cpu")
            if isinstance(tensor, dict):
                tensor = tensor.get("map", tensor.get("hard_map", tensor))
            arr = torch.as_tensor(tensor, dtype=torch.float32)
            if arr.ndim == 2:
                arr = arr.unsqueeze(0)
        elif candidate.suffix.lower() == ".npy":
            arr = torch.from_numpy(np.asarray(np.load(candidate), dtype=np.float32))
            if arr.ndim == 2:
                arr = arr.unsqueeze(0)
        else:
            image = Image.open(candidate).convert("L")
            arr = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)

        if arr.shape != reference.shape:
            arr = F.interpolate(arr.unsqueeze(0), size=reference.shape[-2:], mode="bilinear", align_corners=False)[0]
        arr = arr.clamp(min=0.0)
        self._hard_mining_cache[sample_id] = arr
        return arr.clone()

    def _crop_triplet(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        fov: torch.Tensor,
        hard_map: torch.Tensor | None = None,
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
        biased_anchor: tuple[int, int] | None = None
        if hard_map is not None and self.hard_mining_bias > 0.0 and random.random() < self.hard_mining_bias:
            weighted_candidates = hard_map.clamp(min=0.0) * fov.float()
            biased_anchor = self._sample_weighted_anchor(weighted_candidates, crop_h, crop_w, max_top, max_left)

        if self.thin_vessel_bias > 0.0 and random.random() < self.thin_vessel_bias:
            if biased_anchor is None:
                thin_candidates = self._thin_vessel_candidates(mask, fov)
                biased_anchor = self._sample_crop_anchor(thin_candidates, crop_h, crop_w, max_top, max_left)

        if biased_anchor is None and self.foreground_bias > 0.0 and valid_foreground.any() and random.random() < self.foreground_bias:
            biased_anchor = self._sample_crop_anchor(valid_foreground, crop_h, crop_w, max_top, max_left)

        if biased_anchor is not None:
            top, left = biased_anchor

        sl_h = slice(top, top + crop_h)
        sl_w = slice(left, left + crop_w)
        return image[:, sl_h, sl_w], mask[:, sl_h, sl_w], fov[:, sl_h, sl_w]

    def _apply_photometric_augment(self, image: torch.Tensor) -> torch.Tensor:
        image = image.clamp(0.0, 1.0)

        if self.brightness_jitter > 0.0:
            factor = random.uniform(max(0.0, 1.0 - self.brightness_jitter), 1.0 + self.brightness_jitter)
            image = image * factor

        if self.contrast_jitter > 0.0:
            factor = random.uniform(max(0.0, 1.0 - self.contrast_jitter), 1.0 + self.contrast_jitter)
            mean = image.mean(dim=(1, 2), keepdim=True)
            image = (image - mean) * factor + mean

        if self.gamma_jitter > 0.0:
            gamma = random.uniform(max(0.5, 1.0 - self.gamma_jitter), 1.0 + self.gamma_jitter)
            image = image.clamp(0.0, 1.0).pow(gamma)

        return image.clamp(0.0, 1.0)

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
        image = self._apply_photometric_augment(image)
        return image, mask, fov

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, mask_path, fov_path = self.samples[idx]
        image = self._load_rgb(image_path)
        mask = self._load_mask(mask_path)
        fov = self._load_mask(fov_path) if self.use_fov_mask else torch.ones_like(mask)
        sample_id = _retinal_sample_id(image_path)
        hard_map = self._load_hard_mining_map(sample_id, mask)
        image, mask, fov = self._crop_triplet(image, mask, fov, hard_map)
        if self.augment:
            image, mask, fov = self._apply_augment(image, mask, fov)
        image = self._normalize(image)
        return image, mask, fov


def _normalize_split_name(split: str) -> str:
    token = str(split).lower().strip()
    if token in {"training", "train"}:
        return "train"
    if token in {"validation", "val"}:
        return "val"
    if token in {"testing", "test"}:
        return "test"
    raise ValueError(f"Unknown split name: {split}")


class ArcadeVesselDataset(Dataset):
    """Binary vessel segmentation loader for ARCADE coronary angiography."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        objective: str = "syntax",
        augment: bool = False,
        crop_size: int | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = _normalize_split_name(split)
        self.objective = str(objective).lower().strip()
        if self.objective not in {"syntax", "stenosis"}:
            raise ValueError(f"Unknown ARCADE objective: {objective}. Expected 'syntax' or 'stenosis'.")
        self.augment = bool(augment)
        self.crop_size = int(crop_size) if crop_size else None
        self.image_size = image_size
        self.dataset_root = self._resolve_dataset_root()
        self.images_dir = self.dataset_root / self.objective / self.split / "images"
        self.annotation_path = self.dataset_root / self.objective / self.split / "annotations" / f"{self.split}.json"
        self.samples = self._collect_samples()

    def _resolve_dataset_root(self) -> Path:
        candidates = [self.root, self.root / "arcade", self.root / "ARCADE"]
        for candidate in candidates:
            images_dir = candidate / self.objective / self.split / "images"
            ann_path = candidate / self.objective / self.split / "annotations" / f"{self.split}.json"
            if images_dir.exists() and ann_path.exists():
                return candidate
        raise FileNotFoundError(
            "ARCADE dataset is missing expected structure. "
            f"Tried roots under {self.root} for {self.objective}/{self.split}/images and annotations/{self.split}.json."
        )

    @staticmethod
    def _parse_polygons(segmentation: Any) -> List[List[tuple[float, float]]]:
        polygons: List[List[tuple[float, float]]] = []
        if not isinstance(segmentation, list):
            return polygons
        for candidate in segmentation:
            if not isinstance(candidate, list) or len(candidate) < 6:
                continue
            points: List[tuple[float, float]] = []
            for i in range(0, len(candidate) - 1, 2):
                points.append((float(candidate[i]), float(candidate[i + 1])))
            if len(points) >= 3:
                polygons.append(points)
        return polygons

    def _collect_samples(self) -> List[Dict[str, Any]]:
        with open(self.annotation_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        annotations_by_image: Dict[int, List[List[tuple[float, float]]]] = defaultdict(list)
        for ann in payload.get("annotations", []):
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            image_id = int(ann["image_id"])
            annotations_by_image[image_id].extend(self._parse_polygons(ann.get("segmentation")))

        samples: List[Dict[str, Any]] = []
        missing_images: List[Path] = []
        for image_rec in sorted(payload.get("images", []), key=lambda rec: int(rec["id"])):
            image_path = self.images_dir / str(image_rec["file_name"])
            if not image_path.exists():
                alt_path = self.images_dir / Path(str(image_rec["file_name"])).name
                if alt_path.exists():
                    image_path = alt_path
                else:
                    missing_images.append(image_path)
                    continue
            samples.append(
                {
                    "image_path": image_path,
                    "height": int(image_rec["height"]),
                    "width": int(image_rec["width"]),
                    "polygons": annotations_by_image.get(int(image_rec["id"]), []),
                }
            )
        if missing_images:
            preview = ", ".join(str(path) for path in missing_images[:3])
            raise FileNotFoundError(
                f"ARCADE image files listed in annotations are missing ({len(missing_images)} files), examples: {preview}"
            )
        if not samples:
            raise FileNotFoundError(f"No ARCADE samples found in {self.annotation_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _normalize(image: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
        return (image - mean) / std

    @staticmethod
    def _load_rgb(path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    @staticmethod
    def _build_mask(height: int, width: int, polygons: List[List[tuple[float, float]]]) -> torch.Tensor:
        canvas = Image.new("L", (width, height), color=0)
        draw = ImageDraw.Draw(canvas)
        for polygon in polygons:
            draw.polygon(polygon, fill=255, outline=255)
        arr = (np.asarray(canvas, dtype=np.float32) > 127.0).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)

    def _apply_resize(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.image_size is None:
            return image, mask, valid_mask
        target_h, target_w = self.image_size
        if image.shape[-2:] == (target_h, target_w):
            return image, mask, valid_mask
        image = F.interpolate(image.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask.unsqueeze(0), size=(target_h, target_w), mode="nearest")[0]
        valid_mask = F.interpolate(valid_mask.unsqueeze(0), size=(target_h, target_w), mode="nearest")[0]
        return image, mask, valid_mask

    def _apply_crop(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.crop_size is None:
            return image, mask, valid_mask
        crop_h = min(int(self.crop_size), image.shape[1])
        crop_w = min(int(self.crop_size), image.shape[2])
        max_top = image.shape[1] - crop_h
        max_left = image.shape[2] - crop_w
        top = 0 if max_top <= 0 else random.randint(0, max_top)
        left = 0 if max_left <= 0 else random.randint(0, max_left)
        sl_h = slice(top, top + crop_h)
        sl_w = slice(left, left + crop_w)
        return image[:, sl_h, sl_w], mask[:, sl_h, sl_w], valid_mask[:, sl_h, sl_w]

    @staticmethod
    def _apply_augment(
        image: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
            valid_mask = torch.flip(valid_mask, dims=[2])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
            valid_mask = torch.flip(valid_mask, dims=[1])
        if random.random() < 0.5:
            image = torch.rot90(image, k=2, dims=[1, 2])
            mask = torch.rot90(mask, k=2, dims=[1, 2])
            valid_mask = torch.rot90(valid_mask, k=2, dims=[1, 2])
        return image, mask, valid_mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = self._load_rgb(sample["image_path"])
        mask = self._build_mask(sample["height"], sample["width"], sample["polygons"])
        valid_mask = torch.ones_like(mask)
        image, mask, valid_mask = self._apply_resize(image, mask, valid_mask)
        image, mask, valid_mask = self._apply_crop(image, mask, valid_mask)
        if self.augment:
            image, mask, valid_mask = self._apply_augment(image, mask, valid_mask)
        image = self._normalize(image)
        return image, mask, valid_mask


def _maybe_subset_dataset(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None:
        return dataset
    limit_i = int(limit)
    if limit_i <= 0 or limit_i >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:limit_i].tolist()
    return torch.utils.data.Subset(dataset, indices)


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
    data_root = retinal_dataset_root(cfg.get("data_root", "./data"), cfg["dataset"])
    use_fov_mask = bool(cfg.get("use_fov_mask", True))
    patch_size = cfg.get("retinal_patch_size", cfg.get("drive_patch_size"))
    patch_size = int(patch_size) if patch_size else None
    foreground_bias = float(cfg.get("retinal_foreground_bias", cfg.get("drive_foreground_bias", 0.0)))
    thin_vessel_bias = float(cfg.get("retinal_thin_vessel_bias", 0.0))
    thin_vessel_neighbor_threshold = int(cfg.get("retinal_thin_vessel_neighbor_threshold", 4))
    hard_mining_dir = cfg.get("retinal_hard_mining_dir")
    hard_mining_bias = float(cfg.get("retinal_hard_mining_bias", 0.0))
    brightness_jitter = float(cfg.get("retinal_brightness_jitter", cfg.get("drive_brightness_jitter", 0.0)))
    contrast_jitter = float(cfg.get("retinal_contrast_jitter", cfg.get("drive_contrast_jitter", 0.0)))
    gamma_jitter = float(cfg.get("retinal_gamma_jitter", cfg.get("drive_gamma_jitter", 0.0)))
    input_mode = str(cfg.get("retinal_input_mode", "rgb")).lower().strip()
    green_blend_alpha = float(cfg.get("retinal_green_blend_alpha", 0.35))

    train_aug = DriveDataset(
        data_root,
        split="training",
        augment=True,
        use_fov_mask=use_fov_mask,
        crop_size=patch_size,
        foreground_bias=foreground_bias,
        thin_vessel_bias=thin_vessel_bias,
        thin_vessel_neighbor_threshold=thin_vessel_neighbor_threshold,
        hard_mining_dir=hard_mining_dir,
        hard_mining_bias=hard_mining_bias,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
        gamma_jitter=gamma_jitter,
        input_mode=input_mode,
        green_blend_alpha=green_blend_alpha,
    )
    # Keep validation deterministic and protocol-aligned with test-time evaluation:
    # no random crops and no foreground-biased sampling.
    train_eval = DriveDataset(
        data_root,
        split="training",
        augment=False,
        use_fov_mask=use_fov_mask,
        crop_size=None,
        foreground_bias=0.0,
        input_mode=input_mode,
        green_blend_alpha=green_blend_alpha,
    )
    test_set = DriveDataset(
        data_root,
        split="test",
        augment=False,
        use_fov_mask=use_fov_mask,
        crop_size=None,
        foreground_bias=0.0,
        input_mode=input_mode,
        green_blend_alpha=green_blend_alpha,
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
    # Use an unbiased full-image reference for class-imbalance estimation so
    # pos_weight is not distorted by foreground-biased training crops.
    train_split.pos_weight_reference = torch.utils.data.Subset(train_eval, train_idx)

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


def _parse_optional_hw(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        edge = int(value)
        return (edge, edge)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        if "," in token:
            parts = [part.strip() for part in token.split(",") if part.strip()]
            if len(parts) != 2:
                raise ValueError(f"Expected two integers for image size, got: {value}")
            return int(parts[0]), int(parts[1])
        edge = int(token)
        return (edge, edge)
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected image size sequence of length 2, got: {value}")
        return int(value[0]), int(value[1])
    raise TypeError(f"Unsupported image size value: {value}")


def _build_arcade_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", 42))
    data_root = arcade_dataset_root(cfg.get("data_root", "./data"))
    dataset_name = canonical_dataset_name(cfg["dataset"])
    objective_from_name = "stenosis" if dataset_name == "arcade_stenosis" else "syntax"
    objective = str(cfg.get("arcade_objective", objective_from_name)).lower().strip()
    if dataset_name == "arcade_syntax":
        objective = "syntax"
    elif dataset_name == "arcade_stenosis":
        objective = "stenosis"

    image_size = _parse_optional_hw(cfg.get("arcade_image_size"))
    crop_size = cfg.get("arcade_patch_size", cfg.get("retinal_patch_size", cfg.get("drive_patch_size")))
    crop_size = int(crop_size) if crop_size else None

    train_set = ArcadeVesselDataset(
        root=data_root,
        split="train",
        objective=objective,
        augment=bool(cfg.get("arcade_augment", True)),
        crop_size=crop_size,
        image_size=image_size,
    )
    val_set = ArcadeVesselDataset(
        root=data_root,
        split="val",
        objective=objective,
        augment=False,
        crop_size=None,
        image_size=image_size,
    )
    test_set = ArcadeVesselDataset(
        root=data_root,
        split="test",
        objective=objective,
        augment=False,
        crop_size=None,
        image_size=image_size,
    )

    train_set = _maybe_subset_dataset(train_set, cfg.get("arcade_train_limit"), seed=seed)
    val_set = _maybe_subset_dataset(val_set, cfg.get("arcade_val_limit"), seed=seed + 1)
    test_set = _maybe_subset_dataset(test_set, cfg.get("arcade_test_limit"), seed=seed + 2)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
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
        dataset_name = canonical_dataset_name(cfg["dataset"])
        if dataset_name in RETINAL_SEG_DATASETS:
            train_loader, val_loader, test_loader, in_c, num_outputs = _build_drive_dataloaders(cfg)
        elif dataset_name in ARCADE_SEG_DATASETS:
            train_loader, val_loader, test_loader, in_c, num_outputs = _build_arcade_dataloaders(cfg)
        else:
            raise ValueError(f"Unknown segmentation dataset: {cfg['dataset']}")
    return train_loader, val_loader, test_loader, in_c, num_outputs, task


def estimate_drive_pos_weight(dataset: Dataset, min_weight: float = 1.0, max_weight: float = 25.0) -> float:
    positives = 0.0
    negatives = 0.0
    for index in range(len(dataset)):
        _image, mask, valid_mask = dataset[index]
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
        "anisotropy_gap": float(cfg.get("reg_anisotropy_gap", 0.0)),
        "hybrid_mix_target": float(cfg.get("reg_hybrid_mix", 0.0)),
    }


def parse_model_widths(raw_widths: Any) -> tuple[int, int, int, int] | None:
    if raw_widths is None:
        return None
    if isinstance(raw_widths, str):
        items = [item.strip() for item in raw_widths.split(",") if item.strip()]
    elif isinstance(raw_widths, (list, tuple)):
        items = list(raw_widths)
    else:
        raise TypeError("model_widths must be None, a comma-separated string, or a sequence of four integers.")

    widths = tuple(int(item) for item in items)
    if len(widths) != 4:
        raise ValueError(f"model_widths must contain exactly 4 integers, got {len(widths)}.")
    if any(width <= 0 for width in widths):
        raise ValueError(f"model_widths must be positive, got {widths}.")
    return widths  # type: ignore[return-value]


def resolve_segmentation_model_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_kwargs: Dict[str, Any] = {}
    for key in ("bottleneck_mode", "decoder_mode", "boundary_mode", "encoder_block_mode"):
        value = cfg.get(key)
        if value is not None:
            model_kwargs[key] = str(value)
    encoder_az_stages = cfg.get("encoder_az_stages")
    if encoder_az_stages is not None:
        model_kwargs["encoder_az_stages"] = int(encoder_az_stages)
    hybrid_mix_init = cfg.get("hybrid_mix_init")
    if hybrid_mix_init is not None:
        model_kwargs["hybrid_mix_init"] = float(hybrid_mix_init)
    hybrid_mix_target = cfg.get("hybrid_mix_target")
    if hybrid_mix_target is not None:
        model_kwargs["hybrid_mix_target"] = float(hybrid_mix_target)
    return model_kwargs


def resolve_azconv_config_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve optional AZConvConfig overrides from experiment configs."""

    key_map = {
        "az_use_fuzzy": "use_fuzzy",
        "az_use_anisotropy": "use_anisotropy",
        "az_learn_directions": "learn_directions",
        "az_geometry_mode": "geometry_mode",
        "az_use_value_projection": "use_value_projection",
        "az_normalize_kernel": "normalize_kernel",
        "az_min_hyperbolicity": "min_hyperbolicity",
        "az_fuzzy_temperature": "fuzzy_temperature",
        "az_normalize_mode": "normalize_mode",
        "az_compatibility_floor": "compatibility_floor",
        "az_use_input_residual": "use_input_residual",
        "az_residual_init": "residual_init",
    }
    bool_targets = {
        "use_fuzzy",
        "use_anisotropy",
        "learn_directions",
        "use_value_projection",
        "normalize_kernel",
        "use_input_residual",
    }
    out: Dict[str, Any] = {}
    for source_key, target_key in key_map.items():
        if source_key not in cfg:
            continue
        value = cfg[source_key]
        if target_key in bool_targets:
            out[target_key] = bool(value)
        elif target_key in {"min_hyperbolicity", "fuzzy_temperature", "compatibility_floor", "residual_init"}:
            out[target_key] = float(value)
        else:
            out[target_key] = str(value)
    return out


def az_config_from_variant_and_overrides(variant: str, overrides: Dict[str, Any] | None = None) -> AZConvConfig:
    az_cfg = az_config_for_variant(variant)
    for key, value in dict(overrides or {}).items():
        if not hasattr(az_cfg, key):
            raise ValueError(f"Unknown AZConvConfig override '{key}'.")
        setattr(az_cfg, key, value)
    return az_cfg


def build_model(
    variant: str,
    num_outputs: int,
    in_channels: int,
    num_rules: int = 4,
    task: str = "classification",
    widths: tuple[int, int, int, int] | None = None,
    model_kwargs: Dict[str, Any] | None = None,
    az_cfg_kwargs: Dict[str, Any] | None = None,
) -> nn.Module:
    model_kwargs = dict(model_kwargs or {})
    az_cfg_kwargs = dict(az_cfg_kwargs or {})
    if task == "classification":
        if variant == "baseline":
            return StandardConvNet(num_classes=num_outputs, in_channels=in_channels)
        if variant.startswith("az_"):
            cfg = az_config_from_variant_and_overrides(variant, az_cfg_kwargs)
            return AZConvNet(num_classes=num_outputs, in_channels=in_channels, num_rules=num_rules, cfg=cfg)
    elif task == "segmentation":
        seg_kwargs = {"widths": widths} if widths is not None else {}
        if variant == "baseline":
            return BaselineUNet(in_channels=in_channels, out_channels=num_outputs, **seg_kwargs)
        if variant == "attention_unet":
            return AttentionUNet(in_channels=in_channels, out_channels=num_outputs, **seg_kwargs)
        if variant == "az_sota":
            cfg = az_config_from_variant_and_overrides(variant, az_cfg_kwargs)
            return AZSOTAUNet(
                in_channels=in_channels,
                out_channels=num_outputs,
                num_rules=num_rules,
                cfg=cfg,
                **model_kwargs,
                **seg_kwargs,
            )
        if variant in {"az_sota_pure", "az_thesis"}:
            cfg = az_config_from_variant_and_overrides(variant, az_cfg_kwargs)
            return AZSOTAUNet(
                in_channels=in_channels,
                out_channels=num_outputs,
                num_rules=num_rules,
                cfg=cfg,
                pure_az=True,
                **model_kwargs,
                **seg_kwargs,
            )
        if variant.startswith("az_"):
            cfg = az_config_from_variant_and_overrides(variant, az_cfg_kwargs)
            return AZUNet(
                in_channels=in_channels,
                out_channels=num_outputs,
                num_rules=num_rules,
                cfg=cfg,
                **seg_kwargs,
            )
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


@torch.no_grad()
def estimate_model_complexity(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    in_channels: int,
    spatial_shape: tuple[int, int],
) -> Dict[str, float]:
    model.eval()
    height, width = spatial_shape
    x = torch.randn(batch_size, in_channels, height, width, device=device)
    totals = {
        "macs": 0.0,
        "az_extra_macs": 0.0,
    }
    handles = []

    def conv_hook(module: nn.Conv2d, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        out = output[0] if isinstance(output, (tuple, list)) else output
        batch = float(out.shape[0])
        out_channels = float(out.shape[1])
        out_h = float(out.shape[2])
        out_w = float(out.shape[3])
        kernel_h, kernel_w = module.kernel_size
        kernel_mul = float(kernel_h * kernel_w * (module.in_channels / module.groups))
        totals["macs"] += batch * out_channels * out_h * out_w * kernel_mul

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        inp = inputs[0]
        batch = float(inp.shape[0]) if inp.ndim > 1 else 1.0
        totals["macs"] += batch * float(module.in_features) * float(module.out_features)

    def az_hook(module: AZConv2d, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        out = output[0] if isinstance(output, (tuple, list)) else output
        batch = float(out.shape[0])
        height_out = float(out.shape[2])
        width_out = float(out.shape[3])
        locations = height_out * width_out
        channels = float(module.in_channels)
        rules = float(module.R)
        patch_area = float(module.k * module.k)

        compat_mults = batch * rules * patch_area * locations * 2.0
        compat_norm = batch * rules * patch_area * locations if module.cfg.normalize_kernel else 0.0
        compat_reduce = batch * locations * max(rules * patch_area - 1.0, 0.0)
        agg_mults = batch * rules * channels * patch_area * locations
        agg_adds = batch * rules * channels * max(patch_area - 1.0, 0.0) * locations
        geometry_ops = batch * rules * patch_area * locations * (8.0 if module.cfg.use_anisotropy else 1.0)
        fuzzy_ops = batch * rules * locations * (3.0 if module.cfg.use_fuzzy else 1.0)

        totals["az_extra_macs"] += (
            compat_mults
            + compat_norm
            + compat_reduce
            + agg_mults
            + agg_adds
            + geometry_ops
            + fuzzy_ops
        )

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, AZConv2d):
            handles.append(module.register_forward_hook(az_hook))

    try:
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    macs = totals["macs"] + totals["az_extra_macs"]
    return {
        "approx_macs_per_forward": float(macs),
        "approx_gmacs_per_forward": float(macs / 1e9),
        "approx_flops_per_forward": float(macs * 2.0),
        "approx_gflops_per_forward": float(macs * 2.0 / 1e9),
        "approx_az_extra_macs_per_forward": float(totals["az_extra_macs"]),
        "approx_az_extra_gmacs_per_forward": float(totals["az_extra_macs"] / 1e9),
    }


def spatial_shape_for_dataset(name: str) -> tuple[int, int]:
    n = canonical_dataset_name(name)
    if n in ("fashion_mnist", "fashionmnist"):
        return (28, 28)
    if n in ("cifar10", "cifar_10"):
        return (32, 32)
    if n == "drive":
        return (584, 565)
    if n in {"chase_db1", "chasedb1", "chase"}:
        return (960, 999)
    if n == "fives":
        return (2048, 2048)
    if n in ARCADE_SEG_DATASETS:
        return (512, 512)
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
    score_tolerance: float = 0.0,
    max_threshold: float | None = None,
    min_recall: float | None = None,
) -> Dict[str, float]:
    if not sweep_rows:
        raise ValueError("Threshold sweep rows must not be empty.")
    if float(score_tolerance) < 0.0:
        raise ValueError("score_tolerance must be non-negative.")

    candidates: List[Dict[str, float]] = list(sweep_rows)

    if max_threshold is not None:
        filtered = [row for row in candidates if float(row["threshold"]) <= float(max_threshold) + 1e-12]
        if filtered:
            candidates = filtered

    if min_recall is not None:
        filtered = [row for row in candidates if float(row.get("recall", 0.0)) >= float(min_recall)]
        if filtered:
            candidates = filtered

    if float(score_tolerance) > 0.0:
        scored = [(row, threshold_metric_value(row, metric)) for row in candidates]
        best_score = max(score for _, score in scored)
        eps = float(score_tolerance) + 1e-12
        near_best_rows = [row for row, score in scored if score >= best_score - eps]
        # Conservative policy for vessel segmentation:
        # if multiple thresholds are effectively equal by selection score,
        # prefer a lower threshold to protect recall.
        return min(
            near_best_rows,
            key=lambda row: (
                abs(float(row["threshold"]) - float(reference_threshold)),
                float(row["threshold"]),
                -threshold_metric_value(row, metric),
            ),
        )

    return max(
        candidates,
        key=lambda row: (
            threshold_metric_value(row, metric),
            -abs(float(row["threshold"]) - float(reference_threshold)),
            -float(row["threshold"]),
        ),
    )


def _normalize_dataset_name(name: Any) -> str:
    return str(name or "").lower().replace("-", "_").strip()


def collect_segmentation_metrics_records(results_dir: str | Path, dataset: str) -> List[Dict[str, Any]]:
    results_path = Path(results_dir)
    records: List[Dict[str, Any]] = []
    dataset_name = _normalize_dataset_name(dataset)

    for metrics_path in sorted(results_path.glob("*/metrics.json")):
        run_name = metrics_path.parent.name
        if "smoke" in run_name.lower():
            continue
        try:
            with open(metrics_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        if _normalize_dataset_name(data.get("dataset")) != dataset_name or data.get("task") != "segmentation":
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
                "num_parameters": int(data.get("num_parameters", 0)),
                "approx_gmacs_per_forward": float(data.get("approx_gmacs_per_forward", 0.0)),
                "metrics_path": str(metrics_path),
            }
        )

    return records


def collect_drive_metrics_records(results_dir: str | Path) -> List[Dict[str, Any]]:
    return collect_segmentation_metrics_records(results_dir, dataset="drive")


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    arr = np.asarray(list(values), dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_segmentation_records_by_variant(
    records: Sequence[Dict[str, Any]],
    variants: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    ordered_variants = list(variants) if variants is not None else []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        variant = str(record.get("variant", "unknown"))
        grouped.setdefault(variant, []).append(record)
        if variants is None and variant not in ordered_variants:
            ordered_variants.append(variant)

    rows: List[Dict[str, Any]] = []
    for variant in ordered_variants:
        variant_records = grouped.get(variant, [])
        if not variant_records:
            continue
        best_record = max(variant_records, key=segmentation_record_selection_key)
        row: Dict[str, Any] = {
            "variant": variant,
            "num_runs": len(variant_records),
            "seeds": sorted({int(record["seed"]) for record in variant_records if record.get("seed") is not None}),
            "best_run_name": str(best_record.get("run_name", "unknown")),
            "best_test_dice": float(best_record.get("test_dice", 0.0)),
            "num_rules": best_record.get("num_rules"),
            "num_parameters": int(best_record.get("num_parameters", 0)),
            "approx_gmacs_per_forward": float(best_record.get("approx_gmacs_per_forward", 0.0)),
            "topology_loss_weight": float(best_record.get("topology_loss_weight", 0.0)),
        }
        for metric in DRIVE_MULTI_SEED_METRICS:
            values = [float(record[metric]) for record in variant_records if record.get(metric) is not None]
            mean_value, std_value = _mean_std(values)
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
        rows.append(row)

    if variants is not None:
        return rows
    return sorted(rows, key=lambda row: (-float(row["test_dice_mean"]), row["variant"]))


def aggregate_drive_records_by_variant(
    records: Sequence[Dict[str, Any]],
    variants: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    return aggregate_segmentation_records_by_variant(records, variants=variants)


def update_segmentation_multiseed_summary(
    results_dir: str | Path,
    dataset: str,
    variants: Sequence[str] | None = None,
    out_name: str | None = None,
) -> Path:
    results_path = Path(results_dir)
    dataset_name = _normalize_dataset_name(dataset)
    records = collect_segmentation_metrics_records(results_path, dataset=dataset_name)
    rows = aggregate_segmentation_records_by_variant(records, variants=variants)

    summary_name = out_name or f"{dataset_name}_multiseed_summary.md"
    out_path = results_path / summary_name
    if not rows:
        out_path.write_text(f"No {dataset_name} segmentation runs with metrics.json were found.\n", encoding="utf-8")
        return out_path

    baseline_row = next((row for row in rows if row["variant"] == "baseline"), None)
    lines = [
        "| Variant | Runs | Seeds | Best Run | Params | GMACs | Dice mean+-std | IoU mean+-std | Precision mean+-std | Recall mean+-std | Specificity mean+-std | Balanced Acc mean+-std | Threshold mean+-std | Fwd mean (s) | Dice vs baseline |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        seeds_text = ",".join(str(seed) for seed in row["seeds"]) if row["seeds"] else "n/a"
        num_rules = row["num_rules"]
        best_run_label = row["best_run_name"]
        if num_rules is not None:
            best_run_label = f"{best_run_label} (R={num_rules})"
        dice_delta = 0.0
        if baseline_row is not None:
            dice_delta = float(row["test_dice_mean"]) - float(baseline_row["test_dice_mean"])
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    str(int(row["num_runs"])),
                    seeds_text,
                    best_run_label,
                    str(int(row["num_parameters"])),
                    f"{row['approx_gmacs_per_forward']:.3f}",
                    f"{row['test_dice_mean']:.4f} +- {row['test_dice_std']:.4f}",
                    f"{row['test_iou_mean']:.4f} +- {row['test_iou_std']:.4f}",
                    f"{row['test_precision_mean']:.4f} +- {row['test_precision_std']:.4f}",
                    f"{row['test_recall_mean']:.4f} +- {row['test_recall_std']:.4f}",
                    f"{row['test_specificity_mean']:.4f} +- {row['test_specificity_std']:.4f}",
                    f"{row['test_balanced_accuracy_mean']:.4f} +- {row['test_balanced_accuracy_std']:.4f}",
                    f"{row['selected_threshold_mean']:.4f} +- {row['selected_threshold_std']:.4f}",
                    f"{row['seconds_per_forward_batch_mean']:.5f}",
                    f"{dice_delta:+.4f}",
                ]
            )
            + " |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def update_drive_multiseed_summary(
    results_dir: str | Path,
    variants: Sequence[str] | None = None,
    out_name: str = "drive_multiseed_summary.md",
) -> Path:
    return update_segmentation_multiseed_summary(results_dir, dataset="drive", variants=variants, out_name=out_name)


def segmentation_record_selection_key(record: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(record["protocol_rank"]),
        float(record["test_dice"]),
        float(record["test_iou"]),
        -float(record["seconds_per_forward_batch"]),
    )


def drive_record_selection_key(record: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return segmentation_record_selection_key(record)


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
        "| Variant | Run | Seed | Rules | Params | GMACs | Aux w | Boundary w | Topology w | Threshold | Test Dice | Test IoU | Precision | Recall | Specificity | Balanced Acc | Fwd batch (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                    str(int(row["num_parameters"])),
                    f"{row['approx_gmacs_per_forward']:.3f}",
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
