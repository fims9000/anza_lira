"""Conservative 3D mask reconstruction from already approved CT28 paths.

Experimental method, not a clinical safety guarantee. No model is trained here.
Coordinates are array/voxel indices, not LPS; affine maps voxels to RAS mm.
Outputs are transactional: a rejected repair returns an unchanged input mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage


NEIGHBORHOOD_26 = np.ones((3, 3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class RepairProposal:
    kind: str
    paths_voxel: tuple[tuple[tuple[int, int, int], ...], ...]


@dataclass
class MaskRepairResult:
    accepted: bool
    reason: str
    mask: np.ndarray
    added_voxels: int
    original_components: int
    final_components: int
    connected_components: tuple[tuple[int, ...], ...]


def _paths(proposal: RepairProposal, shape: tuple[int, int, int]) -> list[np.ndarray]:
    if proposal.kind not in {"PAIR", "JUNCTION"}:
        raise ValueError("Only PAIR or JUNCTION repair proposal is supported")
    expected = 1 if proposal.kind == "PAIR" else 3
    if len(proposal.paths_voxel) != expected:
        raise ValueError(f"{proposal.kind} requires {expected} path(s)")
    paths = []
    for arm in proposal.paths_voxel:
        points = np.asarray(arm)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("Each path must have at least two 3D voxels")
        if not np.issubdtype(points.dtype, np.integer):
            raise ValueError("Paths must contain integer voxel coordinates")
        points = points.astype(np.int64)
        if (points < 0).any() or (points >= np.asarray(shape)).any():
            raise ValueError("Path leaves segmentation volume")
        if len({tuple(p) for p in points}) != len(points):
            raise ValueError("Path contains repeated voxels")
        if not np.all(np.max(np.abs(np.diff(points, axis=0)), axis=1) == 1):
            raise ValueError("Path contains non-26-connected steps")
        paths.append(points)
    if proposal.kind == "JUNCTION" and len({tuple(p[-1]) for p in paths}) != 1:
        raise ValueError("JUNCTION arms must meet at the same voxel")
    return paths


def _rasterize(paths: Sequence[np.ndarray], affine: np.ndarray,
               shape: tuple[int, int, int], radius_mm: float) -> np.ndarray:
    """Union of physical-radius balls around voxel-center paths."""
    transform = affine[:3, :3]
    min_scale = float(np.min(np.linalg.svd(transform, compute_uv=False)))
    reach = int(np.ceil(radius_mm / min_scale))
    result = np.zeros(shape, dtype=bool)
    for path in paths:
        for center in path:
            low = np.maximum(center - reach, 0)
            high = np.minimum(center + reach + 1, shape)
            index = np.indices(tuple(high - low), dtype=np.int32)
            offsets = (index.reshape(3, -1).T + low - center).astype(float)
            distance = np.linalg.norm(offsets @ transform.T, axis=1)
            patch = distance.reshape(tuple(high - low)) <= radius_mm + 1e-9
            view = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
            result[view] |= patch
    return result


def apply_approved_paths(
    mask: np.ndarray,
    affine: np.ndarray,
    proposals: Sequence[RepairProposal],
    *,
    radius_mm: float,
    max_new_voxels: int,
    allowed_new_voxels: np.ndarray | None = None,
) -> MaskRepairResult:
    """Attempt an atomic topology-guarded reconstruction.

    Preconditions: input is a broken predicted vessel mask; paths were
    accepted by a separate Graph-LIRA relation/consistency gate. All endpoint
    voxels must exist in distinct original 26-connected components. For BOTH,
    provide exactly one PAIR and one JUNCTION proposal, without overlapping
    original component labels. A new path must not touch any third component.

    radius_mm and max_new_voxels must be frozen using train data, not test.
    allowed_new_voxels can be a train-defined CT-support mask; missing support
    causes rejection rather than silently modifying the route.
    """
    if mask.ndim != 3 or not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError("Mask must be a 3D boolean array")
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Affine must be finite 4x4")
    if not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError("Affine is not homogeneous")
    if np.min(np.linalg.svd(affine[:3, :3], compute_uv=False)) <= 0:
        raise ValueError("Affine is singular")
    if not np.isfinite(radius_mm) or radius_mm < 0:
        raise ValueError("radius_mm must be finite and nonnegative")
    if not isinstance(max_new_voxels, int) or max_new_voxels <= 0:
        raise ValueError("max_new_voxels must be a positive integer")
    if allowed_new_voxels is not None and (allowed_new_voxels.shape != mask.shape or
                                           allowed_new_voxels.dtype != bool):
        raise ValueError("allowed_new_voxels must be a same-shaped boolean mask")
    if not proposals or len(proposals) > 2:
        raise ValueError("Require one relation or two disjoint relations")
    if len(proposals) == 2 and {p.kind for p in proposals} != {"PAIR", "JUNCTION"}:
        raise ValueError("Two proposals must be a PAIR and a JUNCTION (BOTH)")

    current = mask.copy()
    old_components, n_old = ndimage.label(mask, structure=NEIGHBORHOOD_26)
    def reject(why: str, labels: Sequence[tuple[int, ...]] = ()) -> MaskRepairResult:
        return MaskRepairResult(False, why, current.copy(), 0, n_old, n_old, tuple(labels))

    path_groups = [_paths(proposal, mask.shape) for proposal in proposals]
    participating = []
    for proposal, paths in zip(proposals, path_groups):
        anchors = [tuple(arm[0]) for arm in paths]
        if proposal.kind == "PAIR":
            anchors.append(tuple(paths[0][-1]))
        labels = tuple(int(old_components[anchor]) for anchor in anchors)
        if any(label == 0 for label in labels):
            return reject("endpoint_not_on_original_segmentation", [labels])
        if len(set(labels)) != len(labels):
            return reject("endpoints_not_in_distinct_components", [labels])
        participating.append(labels)
    if len(participating) == 2 and set(participating[0]) & set(participating[1]):
        return reject("both_relations_share_original_component", participating)

    reconstruction_groups = [
        _rasterize(paths, affine, mask.shape, radius_mm) for paths in path_groups
    ]
    additions = [candidate & ~mask for candidate in reconstruction_groups]
    if allowed_new_voxels is not None:
        if any(np.any(added & ~allowed_new_voxels) for added in additions):
            return reject("ct_support_rejected_new_voxels", participating)
    if len(additions) == 2:
        # A 26-neighbor contact would join separate intended BOTH relations.
        if np.any(ndimage.binary_dilation(reconstruction_groups[0],
                                          structure=NEIGHBORHOOD_26) & reconstruction_groups[1]):
            return reject("both_repairs_contact_each_other", participating)
    added = np.logical_or.reduce(additions)
    n_added = int(added.sum())
    if n_added == 0:
        return reject("no_new_voxels", participating)
    if n_added > max_new_voxels:
        return reject("volume_budget_exceeded", participating)

    candidate_mask = mask | added
    final_labels, n_final = ndimage.label(candidate_mask, structure=NEIGHBORHOOD_26)
    for labels, paths in zip(participating, path_groups):
        anchor_labels = [int(final_labels[tuple(path[0])]) for path in paths]
        if len(paths) == 1:
            anchor_labels.append(int(final_labels[tuple(paths[0][-1])]))
        if len(set(anchor_labels)) != 1 or 0 in anchor_labels:
            return reject("selected_components_not_connected", participating)
        attached = set(np.unique(old_components[final_labels == anchor_labels[0]])) - {0}
        if attached != set(labels):
            return reject("unselected_component_contact", participating)
    expected_count = n_old - sum(len(labels) - 1 for labels in participating)
    if n_final != expected_count:
        return reject("unexpected_component_count", participating)
    return MaskRepairResult(True, "ok", candidate_mask, n_added,
                            n_old, n_final, tuple(participating))
