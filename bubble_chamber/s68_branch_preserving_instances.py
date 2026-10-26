"""S68: branch-preserving, pixel-only 2-D *fragment proposals*.

Crossing pixels are ambiguous: split at junction cores, preserve the separate
arms, and NEVER silently fuse two arms into a physical particle identity.
Junctions are detected by Moore-neighborhood arm transitions (topological
branches), instead of naive 8-neighbor degree at curved one-pixel corners.
No labels or source simulator coordinates enter inference.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .s39_machine_auto import (
    ridge_evidence, raw_detector_mask, _trace_component, sample_bilinear,
)

METHOD = 's68_topology_preserving_fragments_v1'
ARMS = ('s39_original', 's67_crossing_orientation_v1',
        's68_branch_split', 's68_branch_endpoint_refine')


def moore_arm_count(skel: np.ndarray) -> np.ndarray:
    """Count 0->1 runs in the cyclic 8-neighbor ring at each pixel.

    Two adjacent occupied neighbors are a single arm, avoiding fake
    junctions at normal diagonal stair-steps of a skeleton.
    """
    if skel.ndim != 2:
        raise ValueError('Expected 2-D skeleton')
    a = np.pad(np.asarray(skel, dtype=bool), 1)
    h, w = skel.shape
    offsets = ((-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1))
    ring = np.stack([a[1+dy:1+dy+h, 1+dx:1+dx+w] for dy, dx in offsets])
    runs = np.sum(ring & ~np.roll(ring, 1, axis=0), axis=0)
    return (runs * (skel > 0)).astype(np.uint8)


def split_skeleton_at_junctions(skel: np.ndarray, *, radius: int = 1):
    """Remove uncertain branch cores; do not discard the detached arms."""
    if skel.ndim != 2 or radius not in (1, 2):
        raise ValueError('Invalid skeleton or guard radius')
    binary = np.asarray(skel, dtype=bool)
    branch = binary & (moore_arm_count(binary) >= 3)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(
        branch.astype('uint8'), connectivity=8)
    # Group adjacent junction pixels before dilating so one large junction
    # does not create multiple competing identities.
    points = [np.asarray(centers[i], dtype=float) for i in range(1, n)
              if stats[i, cv2.CC_STAT_AREA] >= 1]
    if radius == 1:
        kernel = np.ones((3, 3), dtype=np.uint8)
    else:
        kernel = np.ones((5, 5), dtype=np.uint8)
    blocked = cv2.dilate(branch.astype('uint8'), kernel) > 0
    return (binary & ~blocked).astype('uint8'), points


def _refine_endpoints(piece: np.ndarray, centers: list[np.ndarray],
                      score: np.ndarray) -> np.ndarray:
    """Optional <4px endpoint completion to a detected ambiguous core.

    Keeps each arm separate. Crossing continuation/identity is NOT inferred.
    """
    curve = np.asarray(piece, dtype=float).copy()
    if len(curve) < 8 or not centers:
        return curve
    for endpoint in (0, -1):
        x = curve[endpoint]
        inside = curve[min(4, len(curve)-1) if endpoint == 0 else max(0, len(curve)-5)]
        outward = x - inside
        norm = np.linalg.norm(outward)
        if norm < 1e-8:
            continue
        direction = outward/norm
        options = sorted(((float(np.linalg.norm(c-x)), i, c) for i, c in enumerate(centers)), key=lambda t: (t[0], t[1]))
        if not options:
            continue
        distance, _, c = options[0]
        if not (0.7 <= distance <= 4.0):
            continue
        v = (c-x)/distance
        if float(np.dot(v, direction)) < .80:
            continue
        # Image support is an absolute gate, not learned from generator truth.
        samples = x[None, :] + np.array([.4, .8])[:, None] * (c-x)
        if np.any(samples < 7) or np.any(samples[:, 0] >= score.shape[1]-7) or np.any(samples[:, 1] >= score.shape[0]-7):
            continue
        target_score = sample_bilinear(score, samples)
        if np.max(target_score) <= 1e-9:
            continue
        moved = x + min(distance, 2.5) * direction
        if np.any(moved < 7) or moved[0] >= score.shape[1]-7 or moved[1] >= score.shape[0]-7:
            continue
        curve[endpoint] = moved
    return curve


def extract_fragment_hypotheses(gray: np.ndarray, *, refine: bool = False,
                                max_curves: int = 32) -> dict:
    """Return 2-D polylines with provenance; NO real trajectory assertions."""
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.shape != (192, 192):
        raise ValueError('Require source uint8 192x192 image pixels only')
    if type(refine) is not bool or type(max_curves) is not int or not 1 <= max_curves <= 32:
        raise ValueError('Invalid frozen extraction parameters')
    evidence = ridge_evidence(gray)
    mask = raw_detector_mask(evidence)
    mask = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    skel = skeletonize(mask > 0).astype('uint8')
    cut, centers = split_skeleton_at_junctions(skel, radius=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cut, connectivity=8)
    components = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 18:
            continue
        x, y, w, h, _ = stats[i]
        path = _trace_component(labels[y:y+h, x:x+w] == i)
        if path is None:
            continue
        path[:, 0] += x
        path[:, 1] += y
        if np.any(path < 7) or np.any(path >= 185):
            continue
        if refine:
            path = _refine_endpoints(path, centers, evidence['score'])
        if np.linalg.norm(path[0] - path[-1]) < 17:
            continue
        quality = float(np.median(sample_bilinear(evidence['score'], path)))
        components.append((quality, path))
    components.sort(key=lambda t: (-t[0], -len(t[1]), float(t[1][0, 0])))
    result = [p for _, p in components[:max_curves]]
    return dict(
        method=METHOD + ('_endpoint_refine' if refine else '_branch_split'),
        polylines=result, pixel_junctions=len(centers),
        prefiltered_components=len(components),
        physical_identity_verified=False, native_expert_labels_used=False,
        synthetic_labels_used_by_predictor=False,
        automatic_physical_join_authorized=False,
    )
