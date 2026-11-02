#!/usr/bin/env python3
"""Generate reproducible *train-only* controlled 3D gaps from a reference mask.

Ground truth is used ONLY to create damaged input and for later evaluation.
The 3D Graph-LIRA runtime must receive ONLY the resulting broken mask + CCTA;
never pass the full reference to its inference engine.

This creates controlled synthetic failures, NOT natural segmentation errors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import product

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from ccta_3d_endpoints import CONNECTIVITY, LPS_SIGN


TRAIN_CT28_IDS = {953, 955, 956, 959, 960, 963, 964, 967, 969,
                  970, 971, 975, 976, 977, 979, 982, 983}
VAL_CT28_IDS = {957, 961, 965, 966, 974}


def _lps(point: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return (affine[:3, :3] @ point + affine[:3, 3]) * LPS_SIGN


def _sphere_to_remove(shape, center, affine, radius_mm):
    """Bounded voxel patch, physically spherical under arbitrary affine."""
    min_scale = min(np.linalg.svd(affine[:3, :3], compute_uv=False))
    reach = int(np.ceil(radius_mm / min_scale))
    left = np.maximum(np.asarray(center) - reach, 0)
    right = np.minimum(np.asarray(center) + reach + 1, shape)
    grid = np.indices(tuple(right - left))
    voxel_offsets = grid.reshape(3, -1).T + left - center
    dist = np.linalg.norm(voxel_offsets @ affine[:3, :3].T, axis=1)
    shape_patch = tuple(right - left)
    return tuple(slice(int(a), int(b)) for a, b in zip(left, right)), (
        dist.reshape(shape_patch) <= radius_mm + 1e-9
    )


def _synthesize_gaps(
    reference: np.ndarray,
    affine: np.ndarray,
    *,
    patient_id: int,
    gaps: int,
    radius_mm: float,
    seed: int = 20261102,
    min_center_spacing_mm: float = 12.0,
    split: str = "train",
) -> tuple[np.ndarray, dict]:
    allowed = TRAIN_CT28_IDS if split == "train" else VAL_CT28_IDS if split == "validation" else set()
    if patient_id not in allowed:
        raise ValueError("Controlled gap generator requires a matching train/validation patient; held-out test forbidden")
    if reference.ndim != 3 or reference.dtype != bool:
        raise ValueError("Need a 3D boolean full reference mask")
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Expected finite voxel-to-RAS affine")
    if gaps < 1 or radius_mm <= 0 or min_center_spacing_mm < 3 * radius_mm:
        raise ValueError("Invalid controlled gap parameters")

    baseline_components = ndimage.label(reference, structure=CONNECTIVITY)[1]
    skeleton = skeletonize(reference)
    degree = ndimage.convolve(
        skeleton.astype(np.uint8), CONNECTIVITY, mode="constant"
    ).astype(np.int16) - skeleton.astype(np.int16)
    interior = np.argwhere(skeleton & (degree == 2))
    endings = np.argwhere(skeleton & (degree == 1))
    junctions = np.argwhere(skeleton & (degree >= 3))
    if len(interior) == 0 or len(endings) < 2:
        raise ValueError("Reference mask has no supported interior centerline")
    if len(interior) > 1_000_000:
        raise ValueError("Too many interior points for bounded simulator")

    transform = affine[:3, :3] @ interior.T
    interior_ras = transform.T + affine[:3, 3]
    safe = np.ones(len(interior), dtype=bool)
    for obstacles, limit in ((endings, max(4.0, 2*radius_mm)),
                             (junctions, max(4.0, 2*radius_mm))):
        if len(obstacles):
            obstacle_ras = (affine[:3, :3] @ obstacles.T).T + affine[:3, 3]
            distances = cKDTree(obstacle_ras).query(interior_ras)[0]
            safe &= distances > limit
    pool = interior[safe]
    if not len(pool):
        raise ValueError("No interior vessel site sufficiently far from tips and junctions")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    damaged = reference.copy()
    accepted = []
    current_components = baseline_components
    for i in order:
        point = pool[int(i)]
        center_ras = affine[:3, :3] @ point + affine[:3, 3]
        if any(np.linalg.norm(center_ras - x["center_ras"]) < min_center_spacing_mm
               for x in accepted):
            continue
        view, sphere = _sphere_to_remove(reference.shape, point, affine, radius_mm)
        if not np.any(damaged[view] & sphere):
            continue
        candidate = damaged.copy()
        candidate[view] &= ~sphere
        new_components = ndimage.label(candidate, structure=CONNECTIVITY)[1]
        if new_components != current_components + 1:
            continue
        removed = int(damaged.sum() - candidate.sum())
        damaged = candidate
        current_components = new_components
        accepted.append({
            "voxel": tuple(int(v) for v in point),
            "center_ras": center_ras,
            "removed_voxels": removed,
            "original_split": 1,
        })
        if len(accepted) == gaps:
            break
    if len(accepted) != gaps:
        raise ValueError(
            f"Could make only {len(accepted)}/{gaps} topology-valid train-only gaps; "
            "increase vessel region or revise the train-only gap plan"
        )
    meta = {
        "protocol": (
            "ct28_train_only_controlled_mask_gap_v1"
            if split == "train" else "ct28_validation_controlled_mask_gap_v1"
        ),
        "patient_id": patient_id,
        "source": "full reference mask used exclusively to create controlled damage",
        "ground_truth_is_never_sent_to_repair_runtime": True,
        "gaps": gaps,
        "gap_radius_mm": radius_mm,
        "seed": seed,
        "min_center_spacing_mm": min_center_spacing_mm,
        "original_components": baseline_components,
        "damaged_components": current_components,
        "removed_voxels": int(reference.sum() - damaged.sum()),
        "centers": [
            {
                "center_voxel": list(x["voxel"]),
                "center_lps": _lps(np.array(x["voxel"]), affine).tolist(),
                "removed_voxels": x["removed_voxels"],
            } for x in accepted
        ],
    }
    return damaged, meta



def synthesize_train_gaps(
    reference: np.ndarray,
    affine: np.ndarray,
    *,
    patient_id: int,
    gaps: int,
    radius_mm: float,
    seed: int = 20261102,
    min_center_spacing_mm: float = 12.0,
) -> tuple[np.ndarray, dict]:
    """Existing TRAIN-only API; protocol and random trajectory unchanged."""
    if patient_id not in TRAIN_CT28_IDS:
        raise ValueError("Controlled gap generation is restricted to CT28 train patients")
    return _synthesize_gaps(
        reference,affine,patient_id=patient_id,gaps=gaps,
        radius_mm=radius_mm,seed=seed,
        min_center_spacing_mm=min_center_spacing_mm,split="train",
    )


def synthesize_validation_gaps(
    reference: np.ndarray,
    affine: np.ndarray,
    *,
    patient_id: int,
    gaps: int,
    radius_mm: float,
    seed: int = 20261102,
    min_center_spacing_mm: float = 12.0,
) -> tuple[np.ndarray, dict]:
    """Frozen validation-only counterpart; NEVER accepts held-out CT28 test."""
    if patient_id not in VAL_CT28_IDS:
        raise ValueError("Validation gap generation is restricted to five CT28 validation patients")
    return _synthesize_gaps(
        reference,affine,patient_id=patient_id,gaps=gaps,
        radius_mm=radius_mm,seed=seed,
        min_center_spacing_mm=min_center_spacing_mm,split="validation",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", type=int, required=True)
    parser.add_argument("--reference-mask", type=Path, required=True)
    parser.add_argument("--output-broken-mask", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gaps", type=int, default=3)
    parser.add_argument("--radius-mm", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if args.patient_id not in TRAIN_CT28_IDS:
        raise ValueError("Only CT28 train subjects are permitted")
    source_paths = {args.reference_mask.resolve()}
    if args.output_broken_mask.resolve() in source_paths or (
        args.report.resolve() in source_paths or
        args.output_broken_mask.resolve() == args.report.resolve()
    ):
        raise ValueError("Cannot overwrite full reference input")
    if args.output_broken_mask.exists() or args.report.exists():
        raise FileExistsError("Outputs exist; do not overwrite experiment evidence")
    if not args.output_broken_mask.name.endswith(".nii.gz") or args.report.suffix != ".json":
        raise ValueError("Expected .nii.gz and .json output names")
    image = nib.load(str(args.reference_mask))
    reference = np.asanyarray(image.dataobj) > 0.5
    corrupted, meta = synthesize_train_gaps(
        reference, image.affine, patient_id=args.patient_id,
        gaps=args.gaps, radius_mm=args.radius_mm, seed=args.seed,
    )
    args.output_broken_mask.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = nib.Nifti1Image(corrupted.astype(np.uint8), image.affine,
                            header=image.header.copy())
    result.header.set_data_dtype(np.uint8)
    nib.save(result, str(args.output_broken_mask))
    args.report.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "patient": args.patient_id,
        "original_components": meta["original_components"],
        "damaged_components": meta["damaged_components"],
        "gaps": args.gaps,
        "output": str(args.output_broken_mask),
    }, indent=2))


if __name__ == "__main__":
    main()
