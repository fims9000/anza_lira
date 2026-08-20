"""GeoCrack pairing, normalization, augmentation, and tensor loading."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset


SOURCE_PATTERNS = (
    re.compile(r"^(?P<source>.+?)_(?:original|binarymask)_patch_?\d+$", re.IGNORECASE),
    re.compile(r"^(?P<source>.+?)_patch_?\d+(?:_mask)?$", re.IGNORECASE),
)
MASK_MARKERS = ("_binarymask_patch", "_mask")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_source_image_id(filename: str | Path) -> str:
    stem = Path(filename).stem
    for pattern in SOURCE_PATTERNS:
        match = pattern.match(stem)
        if match:
            return match.group("source")
    raise ValueError(f"Cannot extract GeoCrack source_image_id from '{filename}'.")


def canonical_patch_id(filename: str | Path) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_binarymask_patch_?(\d+)$", r"_patch\1", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_original_patch_?(\d+)$", r"_patch\1", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_mask$", "", stem, flags=re.IGNORECASE)
    return stem


def _is_mask(path: Path) -> bool:
    stem = path.stem.lower()
    return "_binarymask_patch" in stem or stem.endswith("_mask")


def discover_pairs(root: str | Path) -> list[dict[str, str]]:
    """Discover image/mask pairs recursively and reject ambiguous or missing pairs."""
    root = Path(root).resolve()
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    images: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    for path in paths:
        patch_id = canonical_patch_id(path)
        target = masks if _is_mask(path) else images
        if patch_id in target:
            raise ValueError(f"Duplicate GeoCrack {'mask' if target is masks else 'image'} for {patch_id}: {path}")
        target[patch_id] = path
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            "GeoCrack image/mask pairing failed: "
            f"missing_masks={missing_masks[:5]}, missing_images={missing_images[:5]}"
        )
    if not images:
        raise FileNotFoundError(f"No GeoCrack image/mask pairs found under {root}")
    rows = []
    for patch_id in sorted(images):
        image_path = images[patch_id]
        mask_path = masks[patch_id]
        image_source = extract_source_image_id(image_path)
        mask_source = extract_source_image_id(mask_path)
        if image_source != mask_source:
            raise ValueError(f"Source mismatch for {patch_id}: {image_source} != {mask_source}")
        rows.append(
            {
                "patch_id": patch_id,
                "source_image_id": image_source,
                "image_path": image_path.relative_to(root).as_posix(),
                "mask_path": mask_path.relative_to(root).as_posix(),
            }
        )
    return rows


def read_split_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"patch_id", "source_image_id", "image_path", "mask_path"}
    if not rows:
        raise ValueError(f"GeoCrack split is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"GeoCrack split {path} lacks columns: {sorted(missing)}")
    return rows


def compute_train_normalization(
    root: str | Path,
    train_csv: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compute RGB mean/std solely from rows in the provided training CSV."""
    root = Path(root)
    train_csv = Path(train_csv)
    rows = read_split_csv(train_csv)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for row in rows:
        image_path = root / row["image_path"]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float64) / 255.0
        flat = image.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_sq_sum += np.square(flat).sum(axis=0)
        pixel_count += flat.shape[0]
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_sq_sum / pixel_count - np.square(mean), 1e-12)
    payload: dict[str, Any] = {
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "sample_count": len(rows),
        "pixel_count": pixel_count,
        "source_csv": train_csv.as_posix(),
        "source_csv_sha256": sha256_file(train_csv),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


class GeoCrackDataset(Dataset):
    """Load one frozen GeoCrack CSV split with optional label-safe augmentation."""

    def __init__(
        self,
        root: str | Path,
        split_csv: str | Path,
        *,
        normalization_path: str | Path,
        augment: bool = False,
        brightness_jitter: float = 0.1,
        contrast_jitter: float = 0.1,
    ) -> None:
        self.root = Path(root)
        self.split_csv = Path(split_csv)
        self.rows = read_split_csv(self.split_csv)
        self.augment = bool(augment)
        self.brightness_jitter = max(0.0, float(brightness_jitter))
        self.contrast_jitter = max(0.0, float(contrast_jitter))
        normalization = json.loads(Path(normalization_path).read_text(encoding="utf-8"))
        self.mean = torch.tensor(normalization["mean"], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(normalization["std"], dtype=torch.float32).view(3, 1, 1)
        if self.mean.shape != (3, 1, 1) or self.std.shape != (3, 1, 1) or torch.any(self.std <= 0):
            raise ValueError(f"Invalid GeoCrack normalization file: {normalization_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _augment_pair(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        rotation = random.choice((0, 90, 180, 270))
        if rotation:
            image = image.rotate(rotation, resample=Image.Resampling.BILINEAR)
            mask = mask.rotate(rotation, resample=Image.Resampling.NEAREST)
        if self.brightness_jitter:
            factor = random.uniform(1.0 - self.brightness_jitter, 1.0 + self.brightness_jitter)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast_jitter:
            factor = random.uniform(1.0 - self.contrast_jitter, 1.0 + self.contrast_jitter)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
        row = self.rows[index]
        image_path = self.root / row["image_path"]
        mask_path = self.root / row["mask_path"]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if image.size != (224, 224) or mask.size != (224, 224):
            raise ValueError(f"GeoCrack patch {row['patch_id']} must be 224x224, got image={image.size}, mask={mask.size}")
        if self.augment:
            image, mask = self._augment_pair(image, mask)
        image_array = np.array(image, dtype=np.float32, copy=True) / 255.0
        mask_array = (np.array(mask, dtype=np.uint8, copy=True) > 0).astype(np.float32)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        image_tensor = (image_tensor - self.mean) / self.std
        if not torch.isfinite(image_tensor).all() or not torch.isfinite(mask_tensor).all():
            raise ValueError(f"Non-finite GeoCrack tensor for {row['patch_id']}")
        if mask_tensor.min() < 0 or mask_tensor.max() > 1:
            raise ValueError(f"Non-binary GeoCrack mask for {row['patch_id']}")
        metadata = {
            "patch_id": row["patch_id"],
            "source_image_id": row["source_image_id"],
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
        return image_tensor, mask_tensor, metadata
