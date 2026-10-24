"""Authentic repo AZUNet vs ordinary U-Net, with partial-label safety checks.

Requires the full anza_lira repo's ``models.azconv`` and ``models.segmentation``
for model construction. Never turn unreviewed (255) into background (0).
This is a train/evaluation adapter, NOT a claim of real-world superiority.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class DatasetNotReadyError(ValueError):
    pass


@dataclass(frozen=True)
class BubbleSegmentationConfig:
    width: int = 16
    layers: tuple[int, ...] = (16, 32, 48, 64)
    num_anza_rules: int = 4
    ignore_value: int = 255
    seed: int = 41


def mask_loss(logits: torch.Tensor, partial_target: torch.Tensor, *, unknown: int = 255) -> torch.Tensor:
    """BCE over ONLY certified positive/negative pixels.

    If there are only positive tracks and no verified artifacts, refuse to
    train a discriminative segmentation head rather than learn all-foreground.
    """
    if logits.shape != partial_target.shape:
        raise ValueError('target/logits shapes differ')
    target=partial_target.to(torch.int64)
    if not torch.all((target==0)|(target==1)|(target==unknown)):
        raise ValueError('target values must be exactly 0, 1 or unknown')
    positive=target==1;negative=target==0
    if not bool(positive.any()) or not bool(negative.any()):
        raise DatasetNotReadyError('Need confirmed positive track and negative artifact pixels; other pixels remain UNKNOWN')
    known=positive|negative
    weight=torch.where(positive,negative.sum().to(logits.dtype)/positive.sum().clamp_min(1).to(logits.dtype),torch.ones_like(logits))
    # Pseudolabel imbalance affects optimization; cap weights and normalize.
    losses=F.binary_cross_entropy_with_logits(logits[known],target[known].to(logits.dtype),weight=weight[known].clamp(max=20),reduction='mean')
    return losses


def build_comparison_models(cfg: BubbleSegmentationConfig = BubbleSegmentationConfig()) -> dict[str, torch.nn.Module]:
    """Use the original AZConv implementation, not a newly invented substitute."""
    from models.segmentation import AZUNet, BaselineUNet
    from models.azconv import AZConvConfig
    width=tuple(int(x) for x in cfg.layers)
    if len(width)!=4 or min(width)<4:
        raise ValueError('expect 4 U-Net widths >=4')
    return {'plain_unet': BaselineUNet(in_channels=1,out_channels=1,widths=width),
            'real_azunet': AZUNet(in_channels=1,out_channels=1,num_rules=cfg.num_anza_rules,
                                  cfg=AZConvConfig(geometry_mode='local_hyperbolic',
                                                   use_anisotropy=True,use_fuzzy=True,
                                                   learn_directions=True),widths=width)}


def inspect_reviewed_dataset(items: list[dict[str, Any]], *, require_film_holdout: bool = True) -> dict[str, Any]:
    """Validate an annotated dataset manifest before any supervised training.

    entries: film_id, frame_id, image_path, target_path; masks are 1=track,
    0=certified artifact, 255=unknown. Neighboring crops from same frame
    and film can never count as independent test examples.
    """
    if not items:
        raise DatasetNotReadyError('No expert-reviewed image/mask pairs')
    film_ids=set();classes=Counter(); source_frames=set()
    for item in items:
        if not all(item.get(k) for k in ('film_id','frame_id','image_path','target_path')):
            raise DatasetNotReadyError('Missing film / frame provenance or data paths')
        image=cv2.imread(str(item['image_path']),cv2.IMREAD_GRAYSCALE)
        labels=cv2.imread(str(item['target_path']),cv2.IMREAD_UNCHANGED)
        if image is None or labels is None or image.shape!=labels.shape or labels.dtype!=np.uint8 or labels.ndim!=2:
            raise DatasetNotReadyError('Missing or mismatched original grayscale tile and sparse-label raster')
        unique=np.unique(labels)
        if not set(map(int,unique)).issubset({0,1,255}):
            raise DatasetNotReadyError('Unknown raster label values')
        film_ids.add(item['film_id'])
        source_frames.add((item['film_id'],item['frame_id']))
        for label,count in zip(*np.unique(labels,return_counts=True)):
            classes[int(label)]+=int(count)
    if classes.get(0,0)==0 or classes.get(1,0)==0:
        raise DatasetNotReadyError('Expert-reviewed positives AND certified artifacts are required; cannot treat other pixels as negatives')
    if require_film_holdout and len(film_ids)<3:
        raise DatasetNotReadyError('Need at least three independent films/runs for train, validation and test; crops of two frames of H50Z_030_01 are one run')
    return {'image_count':len(items),'film_count':len(film_ids),'frame_count':len(source_frames),
            'class_pixel_counts':dict(classes),'label_semantics':'1 track, 0 certified artifact, 255 UNKNOWN',
            'test_unit':'entire independent film/run'}
