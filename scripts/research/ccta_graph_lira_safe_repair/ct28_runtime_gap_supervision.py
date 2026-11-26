"""Label synthetic PAIR gaps from a separate reference mask (supervision/audit only).

Never call from production inference. No candidate is labelled positive solely
because its endpoints belong to the same intact anatomical component.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)


def gap_pair_labels(reference, broken, affine, gaps, endpoints, scenes, tolerance_mm=3.0):
    if reference.shape != broken.shape or reference.ndim != 3:
        raise ValueError("Unaligned reference and damaged mask")
    if reference.dtype != bool or broken.dtype != bool or np.any(broken & ~reference):
        raise ValueError("Expected controlled deletion from a binary reference")
    if not np.isfinite(affine).all() or affine.shape != (4, 4):
        raise ValueError("Invalid affine")
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("Invalid gap matching tolerance")
    nz = np.argwhere(reference)
    if not len(nz):
        raise ValueError("Empty reference")
    low = np.maximum(nz.min(axis=0)-1, 0)
    high = np.minimum(nz.max(axis=0)+2, reference.shape)
    roi = tuple(slice(int(a), int(b)) for a,b in zip(low, high))
    removed = reference[roi] & ~broken[roi]
    removed_xyz = np.argwhere(removed)
    claimed = np.zeros(len(removed_xyz), dtype=bool)
    radius = float(gaps['gap_radius_mm'])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Invalid frozen gap radius')
    components, _ = ndimage.label(broken[roi], structure=CONNECTIVITY)
    lookup = {e.endpoint_id: e for e in endpoints}
    if len(lookup) != len(endpoints):
        raise ValueError("Duplicate endpoint identity")
    pair_keys = {tuple(sorted(c.endpoint_ids)) for s in scenes
                 for c in s.candidates if c.kind == "PAIR"}
    positions = {key: np.asarray(e.voxel, dtype=int)-low for key,e in lookup.items()}
    def physical(voxels):
        return np.asarray(voxels) @ affine[:3,:3].T
    result = []
    for index, gap in enumerate(gaps["centers"], 1):
        center = np.asarray(gap["center_voxel"], dtype=int)-low
        if np.any(center < 0) or np.any(center >= np.asarray(removed.shape)):
            raise ValueError("Gap center outside reference crop")
        inside = np.linalg.norm(physical(removed_xyz-center), axis=1) <= radius + 1e-7
        if not inside.any() or np.any(claimed & inside):
            raise ValueError("Gap spheres overlap or identify no removed voxels")
        claimed |= inside
        deleted = np.zeros_like(removed)
        deleted[tuple(removed_xyz[inside].T)] = True
        if int(deleted.sum()) != int(gap["removed_voxels"]):
            raise ValueError("Gap voxel count differs from frozen damage receipt")
        contact = ndimage.binary_dilation(deleted, structure=CONNECTIVITY) & ~deleted
        adjacent = set(np.unique(components[contact])) - {0}
        record = {"gap": index, "center_voxel": gap["center_voxel"],
                  "removed_voxels": int(deleted.sum()),
                  "adjacent_components": sorted(int(v) for v in adjacent),
                  "candidate_pairs": []}
        if len(adjacent) != 2:
            record["status"] = "ambiguous_gap_topology"
            result.append(record)
            continue
        tree = cKDTree(physical(np.argwhere(deleted)))
        for a,b in sorted(pair_keys):
            va,vb = positions[a],positions[b]
            if any(np.any(v < 0) or np.any(v >= np.asarray(removed.shape)) for v in (va,vb)):
                continue
            if {int(components[tuple(va)]),int(components[tuple(vb)])} != adjacent:
                continue
            distances = tree.query(physical(np.stack([va,vb])))[0]
            if np.max(distances) <= tolerance_mm:
                record["candidate_pairs"].append([a,b])
        count = len(record["candidate_pairs"])
        record["status"] = ("unique" if count == 1 else
                            "uncovered" if count == 0 else "ambiguous_pair")
        result.append(record)
    if not claimed.all():
        raise ValueError("Unrecorded removed voxels; cannot assign reliable labels")
    return result


def window_labels(scenes, gap_records):
    """Only NONE or a single unambiguous PAIR is representable in this experiment.

    Exclude ambiguous topology at the patient level and windows containing
    ambiguous candidate pairs or multiple deleted arcs. Duplicate windows
    can share a positive PAIR but are not independent gaps.
    """
    if any(g["status"] == "ambiguous_gap_topology" for g in gap_records):
        return {}, {s.scene_id: "ambiguous_gap_topology" for s in scenes}
    true_by_key = defaultdict(set)
    ambiguous = set()
    for g in gap_records:
        for pair in g["candidate_pairs"]:
            key = tuple(pair)
            if g["status"] == "unique":
                true_by_key[key].add(g["gap"])
            else:
                ambiguous.add(key)
    labels, skipped = {}, {}
    for scene in scenes:
        keys = {tuple(sorted(c.endpoint_ids)) for c in scene.candidates if c.kind=="PAIR"}
        positives = keys & set(true_by_key)
        if keys & ambiguous or len(positives) > 1 or any(len(true_by_key[k]) != 1 for k in positives):
            skipped[scene.scene_id] = "ambiguous_or_multiple_pairs"
        else:
            labels[scene.scene_id] = "PAIR" if positives else "NONE"
    return labels, skipped
