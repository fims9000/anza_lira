"""Per-annotator CRACKS partial labels with white pixels kept unknown."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets.cracks import BLUE, GREEN, ORANGE, WHITE, load_section_image
import utils


PARTIAL_LABEL_WEIGHTS = {
    BLUE: (1.0, 1.0),
    GREEN: (1.0, 0.5),
    ORANGE: (0.0, 1.0),
    WHITE: (0.0, 0.0),
}


def _assert_nonexpert(annotators: Sequence[str]) -> None:
    forbidden = [name for name in annotators if name == "expert" or name.startswith("expert")]
    if forbidden:
        raise PermissionError(f"T1 partial-label protocol forbids expert annotations: {forbidden}")


def map_partial_annotation(mask_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map one official RGB annotation to target and loss weight.

    Blue is certain fault, green uncertain fault, orange certain no-fault, and
    white is unknown. Unknown colors fail closed rather than becoming labels.
    """
    rgb = np.asarray(mask_rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB annotation, got {rgb.shape}")
    target = np.zeros(rgb.shape[:2], dtype=np.float32)
    weight = np.zeros(rgb.shape[:2], dtype=np.float32)
    known = np.zeros(rgb.shape[:2], dtype=bool)
    for color, (value, confidence) in PARTIAL_LABEL_WEIGHTS.items():
        selected = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)
        known |= selected
        target[selected] = value
        weight[selected] = confidence
    if not np.all(known):
        unknown = np.unique(rgb[~known].reshape(-1, 3), axis=0)
        raise ValueError(f"Annotation contains colors outside frozen T1 semantics: {unknown.tolist()}")
    return target, weight


def average_annotator_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    topology_weight: float = 0.2,
    topology_num_iters: int = 5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average independently normalized masked losses over annotators."""
    if logits.shape[0] != 1 or logits.shape[1] != 1:
        raise ValueError("T1 training expects one section crop per optimization microbatch")
    if targets.ndim != 4 or targets.shape[1] != 1 or targets.shape != weights.shape:
        raise ValueError("targets and weights must be Ax1xHxW")
    if tuple(targets.shape[-2:]) != tuple(logits.shape[-2:]):
        raise ValueError("Partial labels and logits do not share spatial shape")
    losses: list[torch.Tensor] = []
    logs: list[dict[str, float]] = []
    for target, weight in zip(targets, weights):
        if float(weight.sum().detach()) <= 0:
            continue
        loss, row, _ = utils.segmentation_objective(
            logits,
            target.unsqueeze(0),
            weight.unsqueeze(0),
            topology_weight=topology_weight,
            topology_num_iters=topology_num_iters,
        )
        losses.append(loss)
        logs.append(row)
    if not losses:
        raise ValueError("T1 crop has no explicit annotation pixels")
    result = torch.stack(losses).mean()
    summary = {
        key: float(np.mean([row[key] for row in logs]))
        for key in logs[0]
    }
    summary["annotator_count"] = float(len(losses))
    summary["total_loss"] = float(result.detach())
    return result, summary


class CRACKSMultiAnnotatorDataset(Dataset[dict[str, Any]]):
    """Return an image and separate partial targets for deterministic annotators."""

    def __init__(
        self,
        image_root: Path,
        annotation_root: Path,
        section_ids: Sequence[int],
        annotators: Sequence[str],
        *,
        mean: Sequence[float],
        std: Sequence[float],
        crop_size: int | None = None,
        foreground_probability: float = 0.7,
        annotators_per_section: int | None = 4,
        seed: int = 42,
    ) -> None:
        _assert_nonexpert(annotators)
        if crop_size is not None and crop_size != 256:
            raise ValueError("CRACKS T1 crop must be 256x256")
        self.image_root = Path(image_root)
        self.annotation_root = Path(annotation_root)
        self.section_ids = [int(value) for value in section_ids]
        self.annotators = tuple(str(value) for value in annotators)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).clamp_min(1e-6).view(3, 1, 1)
        self.crop_size = crop_size
        self.foreground_probability = float(foreground_probability)
        self.annotators_per_section = annotators_per_section
        self.seed = int(seed)
        self.epoch = 0
        self.available: dict[int, tuple[str, ...]] = {}
        for section_id in self.section_ids:
            name = f"section_{section_id:03d}.png"
            names = tuple(a for a in self.annotators if (self.annotation_root / a / name).is_file())
            if not names:
                raise FileNotFoundError(f"No allowed annotations for CRACKS section {section_id}")
            self.available[section_id] = names

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.section_ids)

    def _selected_annotators(self, index: int) -> tuple[str, ...]:
        names = self.available[self.section_ids[index]]
        if self.annotators_per_section is None or self.annotators_per_section >= len(names):
            return names
        generator = np.random.default_rng(self.seed + self.epoch * len(self) + index)
        selected = generator.choice(
            len(names), size=int(self.annotators_per_section), replace=False
        )
        return tuple(names[int(value)] for value in sorted(selected.tolist()))

    def __getitem__(self, index: int) -> dict[str, Any]:
        section_id = self.section_ids[index]
        name = f"section_{section_id:03d}.png"
        image_np = load_section_image(self.image_root / name)
        names = self._selected_annotators(index)
        targets_np: list[np.ndarray] = []
        weights_np: list[np.ndarray] = []
        for annotator in names:
            with Image.open(self.annotation_root / annotator / name) as handle:
                target, weight = map_partial_annotation(np.asarray(handle.convert("RGB"), dtype=np.uint8))
            targets_np.append(target)
            weights_np.append(weight)
        if image_np.shape[:2] != (255, 701):
            raise ValueError(f"Unexpected CRACKS section shape for {section_id}: {image_np.shape}")
        image = (torch.from_numpy(image_np.transpose(2, 0, 1)) - self.mean) / self.std
        targets = torch.from_numpy(np.stack(targets_np, axis=0)).unsqueeze(1)
        weights = torch.from_numpy(np.stack(weights_np, axis=0)).unsqueeze(1)
        image = F.pad(image, (0, 3, 0, 1))
        targets = F.pad(targets, (0, 3, 0, 1))
        weights = F.pad(weights, (0, 3, 0, 1))
        crop_origin = (0, 0)
        if self.crop_size is not None:
            generator = np.random.default_rng(self.seed + self.epoch * len(self) + index)
            positive = torch.nonzero(((targets > 0.5) & (weights > 0)).any(dim=0)[0], as_tuple=False)
            explicit = torch.nonzero((weights > 0).any(dim=0)[0], as_tuple=False)
            if not len(explicit):
                raise ValueError(f"Selected T1 annotations have no explicit labels for section {section_id}")
            if len(positive) and generator.random() < self.foreground_probability:
                anchor = positive[int(generator.integers(len(positive)))]
            else:
                random_top = int(generator.integers(0, image.shape[-2] - 256 + 1))
                random_left = int(generator.integers(0, image.shape[-1] - 256 + 1))
                random_weights = weights[:, :, random_top : random_top + 256, random_left : random_left + 256]
                anchor = None if torch.any(random_weights > 0) else explicit[int(generator.integers(len(explicit)))]
            if anchor is None:
                top, left = random_top, random_left
            else:
                center_y, center_x = (int(v) for v in anchor)
                top = min(max(center_y - 128, 0), image.shape[-2] - 256)
                left = min(max(center_x - 128, 0), image.shape[-1] - 256)
            crop_origin = (top, left)
            image = image[:, top : top + 256, left : left + 256]
            targets = targets[:, :, top : top + 256, left : left + 256]
            weights = weights[:, :, top : top + 256, left : left + 256]
        if not torch.any(weights > 0):
            raise ValueError(f"Selected T1 crop has no explicit labels for section {section_id}")
        return {
            "image": image,
            "targets": targets,
            "weights": weights,
            "annotators": names,
            "section_id": section_id,
            "original_hw": (255, 701),
            "crop_origin": crop_origin,
        }


def audit_nonexpert_annotations(
    annotation_root: Path,
    annotators: Sequence[str],
    section_ids: Sequence[int],
) -> dict[str, Any]:
    """Count exact colors and coverage without ever traversing expert data."""
    _assert_nonexpert(annotators)
    root = Path(annotation_root)
    counts: Counter[tuple[int, int, int]] = Counter()
    by_role: dict[str, Counter[tuple[int, int, int]]] = {
        "novice": Counter(),
        "practitioner": Counter(),
    }
    files = 0
    missing = 0
    for annotator in annotators:
        role = "practitioner" if annotator.startswith("practitioner") else "novice"
        for section_id in section_ids:
            path = root / annotator / f"section_{int(section_id):03d}.png"
            if not path.is_file():
                missing += 1
                continue
            with Image.open(path) as handle:
                rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            known = np.zeros(rgb.shape[:2], dtype=bool)
            for color in PARTIAL_LABEL_WEIGHTS:
                selected = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)
                count = int(np.count_nonzero(selected))
                known |= selected
                counts[color] += count
                by_role[role][color] += count
            if not np.all(known):
                for color in np.unique(rgb[~known].reshape(-1, 3), axis=0):
                    key = tuple(int(v) for v in color)
                    count = int(np.count_nonzero(np.all(rgb == color, axis=-1)))
                    counts[key] += count
                    by_role[role][key] += count
            files += 1
    allowed = set(PARTIAL_LABEL_WEIGHTS)
    unknown = sorted(color for color in counts if color not in allowed)
    if unknown:
        raise ValueError(f"Non-expert CRACKS palette contains unknown colors: {unknown}")

    def summarize(values: Counter[tuple[int, int, int]]) -> dict[str, Any]:
        total = sum(values.values())
        named = {
            "blue": values[BLUE],
            "green": values[GREEN],
            "orange": values[ORANGE],
            "white": values[WHITE],
        }
        return {
            "pixels": named,
            "fractions": {key: float(value / total) if total else 0.0 for key, value in named.items()},
            "explicit_fraction": float((named["blue"] + named["green"] + named["orange"]) / total)
            if total else 0.0,
        }

    return {
        "status": "PASS",
        "annotation_files": files,
        "missing_annotation_files": missing,
        "annotators": list(annotators),
        "section_count": len(section_ids),
        "palette": summarize(counts),
        "by_role": {role: summarize(value) for role, value in by_role.items()},
        "semantics": {
            "blue": {"target": 1.0, "weight": 1.0},
            "green": {"target": 1.0, "weight": 0.5},
            "orange": {"target": 0.0, "weight": 1.0},
            "white": {"target": None, "weight": 0.0, "meaning": "IGNORE_NOT_BACKGROUND"},
        },
        "expert_data_accessed": False,
    }
