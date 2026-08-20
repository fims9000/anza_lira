"""CRACKS crowd-target fusion and section loading for ANZA-LIRA v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


RGB = tuple[int, int, int]
BLUE: RGB = (31, 119, 180)
GREEN: RGB = (44, 160, 44)
ORANGE: RGB = (255, 127, 14)
WHITE: RGB = (255, 255, 255)


@dataclass(frozen=True)
class MaskPolicy:
    name: str
    positive: tuple[RGB, ...]
    negative: tuple[RGB, ...]
    ignored: tuple[RGB, ...]


POLICIES = {
    "paper_like": MaskPolicy("paper_like", (BLUE, GREEN), (WHITE,), (ORANGE,)),
    "conservative": MaskPolicy("conservative", (BLUE, GREEN), (ORANGE,), (WHITE,)),
}


def _matches(rgb: np.ndarray, color: RGB) -> np.ndarray:
    return np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)


def annotator_expertise_weight(annotator: str) -> float:
    if annotator.startswith("practitioner"):
        return 2.0
    if annotator.startswith("novice"):
        return 1.0
    if annotator == "expert":
        return 1.0
    raise ValueError(f"Unknown CRACKS annotator expertise: {annotator}")


def map_mask_rgb(mask_rgb: np.ndarray, policy_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return binary target, valid mask, and source-compatible confidence."""
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown CRACKS mask policy: {policy_name}")
    rgb = np.asarray(mask_rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB mask, got {rgb.shape}")
    policy = POLICIES[policy_name]
    target = np.zeros(rgb.shape[:2], dtype=np.float32)
    valid = np.zeros(rgb.shape[:2], dtype=bool)
    confidence = np.ones(rgb.shape[:2], dtype=np.float32)
    for color in policy.negative:
        valid |= _matches(rgb, color)
    for color in policy.positive:
        selected = _matches(rgb, color)
        valid |= selected
        target[selected] = 1.0
        confidence[selected] = 1.5 if color == BLUE else 1.0
    known = np.zeros_like(valid)
    for color in (*policy.positive, *policy.negative, *policy.ignored):
        known |= _matches(rgb, color)
    if not np.all(known):
        unknown = np.unique(rgb[~known].reshape(-1, 3), axis=0)
        raise ValueError(f"Mask contains RGB values outside frozen policy: {unknown.tolist()}")
    return target, valid, confidence


def fuse_crowd_masks(
    masks_rgb: Sequence[np.ndarray],
    annotators: Sequence[str],
    policy_name: str,
    *,
    minimum_disagreement_support: int = 5,
) -> dict[str, np.ndarray]:
    if not masks_rgb or len(masks_rgb) != len(annotators):
        raise ValueError("Crowd fusion requires equally sized non-empty masks and annotator lists")
    shape = np.asarray(masks_rgb[0]).shape[:2]
    numerator = np.zeros(shape, dtype=np.float64)
    denominator = np.zeros(shape, dtype=np.float64)
    positive_votes = np.zeros(shape, dtype=np.uint16)
    support = np.zeros(shape, dtype=np.uint16)
    for mask, annotator in zip(masks_rgb, annotators):
        if np.asarray(mask).shape[:2] != shape:
            raise ValueError("Crowd masks do not share a common shape")
        target, valid, confidence = map_mask_rgb(mask, policy_name)
        weight = annotator_expertise_weight(annotator) * confidence
        numerator += np.where(valid, weight * target, 0.0)
        denominator += np.where(valid, weight, 0.0)
        support += valid.astype(np.uint16)
        positive_votes += (valid & (target > 0.5)).astype(np.uint16)
    fused = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    valid = denominator > 0
    disagreement_valid = support >= int(minimum_disagreement_support)
    vote_fraction = np.divide(
        positive_votes,
        support,
        out=np.zeros_like(numerator),
        where=support > 0,
    )
    clipped = np.clip(vote_fraction, 1e-7, 1.0 - 1e-7)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)) / np.log(2.0)
    entropy = np.where(disagreement_valid, entropy, 0.0)
    return {
        "target": fused.astype(np.float32),
        "valid": valid,
        "weight_sum": denominator.astype(np.float32),
        "support": np.minimum(support, 255).astype(np.uint8),
        "human_entropy": entropy.astype(np.float32),
        "human_entropy_valid": disagreement_valid,
    }


def load_rgb_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_section_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


class CRACKSSectionDataset(Dataset[dict[str, Any]]):
    """Load padded full sections or deterministic foreground-aware training crops."""

    def __init__(
        self,
        image_root: Path,
        target_root: Path,
        section_ids: Sequence[int],
        *,
        mean: Sequence[float],
        std: Sequence[float],
        crop_size: int | None = None,
        foreground_probability: float = 0.7,
        seed: int = 42,
    ) -> None:
        self.image_root = Path(image_root)
        self.target_root = Path(target_root)
        self.section_ids = [int(value) for value in section_ids]
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).clamp_min(1e-6).view(3, 1, 1)
        self.crop_size = crop_size
        self.foreground_probability = float(foreground_probability)
        self.seed = int(seed)
        self.epoch = 0
        if crop_size is not None and crop_size != 256:
            raise ValueError("CRACKS primary training crop must be 256x256")
        if not 0.0 <= self.foreground_probability <= 1.0:
            raise ValueError("foreground_probability must be in [0,1]")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.section_ids)

    def _paths(self, section_id: int) -> tuple[Path, Path]:
        name = f"section_{section_id:03d}"
        return self.image_root / f"{name}.png", self.target_root / f"{name}.npz"

    def __getitem__(self, index: int) -> dict[str, Any]:
        section_id = self.section_ids[index]
        image_path, target_path = self._paths(section_id)
        image_np = load_section_image(image_path)
        with np.load(target_path) as payload:
            target_np = payload["target"].astype(np.float32)
            valid_np = payload["valid"].astype(bool)
            entropy_np = payload["human_entropy"].astype(np.float32)
            entropy_valid_np = payload["human_entropy_valid"].astype(bool)
        if image_np.shape[:2] != (255, 701) or target_np.shape != (255, 701):
            raise ValueError(f"Unexpected CRACKS section shape for {section_id}")
        image = torch.from_numpy(image_np.transpose(2, 0, 1))
        image = (image - self.mean) / self.std
        target = torch.from_numpy(target_np).unsqueeze(0)
        valid = torch.from_numpy(valid_np).unsqueeze(0)
        entropy = torch.from_numpy(entropy_np).unsqueeze(0)
        entropy_valid = torch.from_numpy(entropy_valid_np).unsqueeze(0)
        image = F.pad(image, (0, 3, 0, 1))
        target = F.pad(target, (0, 3, 0, 1))
        valid = F.pad(valid, (0, 3, 0, 1), value=False)
        entropy = F.pad(entropy, (0, 3, 0, 1))
        entropy_valid = F.pad(entropy_valid, (0, 3, 0, 1), value=False)
        crop_origin = (0, 0)
        if self.crop_size is not None:
            generator = np.random.default_rng(self.seed + self.epoch * len(self) + index)
            foreground = torch.nonzero((target[0] >= 0.5) & valid[0], as_tuple=False)
            choose_foreground = len(foreground) > 0 and generator.random() < self.foreground_probability
            if choose_foreground:
                yx = foreground[int(generator.integers(len(foreground)))].tolist()
                center_y, center_x = int(yx[0]), int(yx[1])
                top = min(max(center_y - 128, 0), image.shape[-2] - 256)
                left = min(max(center_x - 128, 0), image.shape[-1] - 256)
            else:
                top = int(generator.integers(0, image.shape[-2] - 256 + 1))
                left = int(generator.integers(0, image.shape[-1] - 256 + 1))
            crop_origin = (top, left)
            slices = (slice(None), slice(top, top + 256), slice(left, left + 256))
            image, target, valid = image[slices], target[slices], valid[slices]
            entropy, entropy_valid = entropy[slices], entropy_valid[slices]
        return {
            "image": image,
            "target": target,
            "valid": valid,
            "human_entropy": entropy,
            "human_entropy_valid": entropy_valid,
            "section_id": section_id,
            "original_hw": (255, 701),
            "crop_origin": crop_origin,
        }


class CRACKSAnnotatedSectionDataset(Dataset[dict[str, Any]]):
    """Load official RGB annotations under an explicit frozen mask policy."""

    def __init__(
        self,
        image_root: Path,
        annotation_root: Path,
        section_ids: Sequence[int],
        *,
        policy_name: str,
        mean: Sequence[float],
        std: Sequence[float],
        crop_size: int | None = None,
        foreground_probability: float = 0.7,
        seed: int = 42,
    ) -> None:
        if policy_name not in POLICIES:
            raise ValueError(f"Unknown CRACKS mask policy: {policy_name}")
        if crop_size is not None and crop_size != 256:
            raise ValueError("CRACKS primary training crop must be 256x256")
        self.image_root = Path(image_root)
        self.annotation_root = Path(annotation_root)
        self.section_ids = [int(value) for value in section_ids]
        self.policy_name = policy_name
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).clamp_min(1e-6).view(3, 1, 1)
        self.crop_size = crop_size
        self.foreground_probability = float(foreground_probability)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.section_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        section_id = self.section_ids[index]
        name = f"section_{section_id:03d}.png"
        image_np = load_section_image(self.image_root / name)
        target_np, valid_np, confidence_np = map_mask_rgb(
            load_rgb_mask(self.annotation_root / name), self.policy_name
        )
        if image_np.shape[:2] != (255, 701) or target_np.shape != (255, 701):
            raise ValueError(f"Unexpected CRACKS section shape for {section_id}")
        image = torch.from_numpy(image_np.transpose(2, 0, 1))
        image = (image - self.mean) / self.std
        target = torch.from_numpy(target_np).unsqueeze(0)
        valid = torch.from_numpy(valid_np).unsqueeze(0)
        confidence = torch.from_numpy(confidence_np).unsqueeze(0)
        image = F.pad(image, (0, 3, 0, 1))
        target = F.pad(target, (0, 3, 0, 1))
        valid = F.pad(valid, (0, 3, 0, 1), value=False)
        confidence = F.pad(confidence, (0, 3, 0, 1))
        crop_origin = (0, 0)
        if self.crop_size is not None:
            generator = np.random.default_rng(self.seed + self.epoch * len(self) + index)
            foreground = torch.nonzero((target[0] >= 0.5) & valid[0], as_tuple=False)
            choose_foreground = len(foreground) > 0 and generator.random() < self.foreground_probability
            if choose_foreground:
                center_y, center_x = (int(value) for value in foreground[int(generator.integers(len(foreground)))])
                top = min(max(center_y - 128, 0), image.shape[-2] - 256)
                left = min(max(center_x - 128, 0), image.shape[-1] - 256)
            else:
                top = int(generator.integers(0, image.shape[-2] - 256 + 1))
                left = int(generator.integers(0, image.shape[-1] - 256 + 1))
            crop_origin = (top, left)
            slices = (slice(None), slice(top, top + 256), slice(left, left + 256))
            image, target, valid, confidence = (
                image[slices], target[slices], valid[slices], confidence[slices]
            )
        return {
            "image": image,
            "target": target,
            "valid": valid,
            "confidence": confidence,
            "section_id": section_id,
            "original_hw": (255, 701),
            "crop_origin": crop_origin,
        }
