"""Independent 3D segmentation evaluation for experimental CT28 repair masks.

These metrics consume ground truth only after inference/reconstruction; never
use this module to choose a route, radius, threshold or candidate at inference.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def dice(predicted: np.ndarray, reference: np.ndarray) -> float:
    cardinality = int(predicted.sum() + reference.sum())
    if cardinality == 0:
        return 1.0
    return 2.0 * float(np.count_nonzero(predicted & reference)) / cardinality


def cldice(predicted: np.ndarray, reference: np.ndarray) -> float:
    if not np.any(predicted) or not np.any(reference):
        return 1.0 if not np.any(predicted) and not np.any(reference) else 0.0
    sp = skeletonize(predicted)
    sr = skeletonize(reference)
    if not np.any(sp) or not np.any(sr):
        return 0.0
    t_precision = np.count_nonzero(sp & reference) / np.count_nonzero(sp)
    t_sensitivity = np.count_nonzero(sr & predicted) / np.count_nonzero(sr)
    return float(2 * t_precision * t_sensitivity / (t_precision + t_sensitivity)) if t_precision + t_sensitivity else 0.0


def evaluate(before: np.ndarray, after: np.ndarray, reference: np.ndarray,
             affine: np.ndarray) -> dict:
    if not (before.ndim == 3 and before.shape == after.shape == reference.shape):
        raise ValueError("Masks must have identical 3D shapes")
    if not all(x.dtype == bool for x in (before, after, reference)):
        raise ValueError("Masks must be boolean")
    if np.any(before & ~after):
        raise ValueError("Reconstruction unexpectedly removed original voxels")
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or np.linalg.det(affine[:3, :3]) == 0:
        raise ValueError("Need a nonsingular voxel-to-world affine")
    new = after & ~before
    supported = new & reference
    wrong = new & ~reference
    component_structure = np.ones((3, 3, 3), dtype=np.uint8)
    b_components = int(ndimage.label(before, structure=component_structure)[1])
    a_components = int(ndimage.label(after, structure=component_structure)[1])
    reference_components = int(ndimage.label(reference, structure=component_structure)[1])
    added_count = int(new.sum())
    mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    return {
        "metric_scope": "post-hoc full-mask evaluation; never provided to the solver",
        "voxel_volume_mm3": mm3,
        "dice_before": dice(before, reference),
        "dice_after": dice(after, reference),
        "cldice_before": cldice(before, reference),
        "cldice_after": cldice(after, reference),
        "original_components": b_components,
        "final_components": a_components,
        "reference_components": reference_components,
        "component_count_change": a_components - b_components,
        "added_voxels": added_count,
        "added_volume_mm3": added_count * mm3,
        "reference_supported_added_voxels": int(supported.sum()),
        "reference_unsupported_added_voxels": int(wrong.sum()),
        "added_voxel_precision_vs_reference": float(supported.sum() / added_count) if added_count else None,
    }
