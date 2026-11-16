"""V52: label-blind joint boundary/trajectory inference for a known masked gap.

This is an experimental decoder to compare against, NOT replace, frozen V50.
TRAIN/VAL provenance and expert labels never enter the route constructor.
The gap rectangle is an intervention supplied by the experiment (not detected).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage as ndi


# Fixed *a priori* geometry priors: no CAL/TEST parameter search.
RADIUS = 11
MAX_JUMP = 2
STEP_PENALTY = 0.14
CURVATURE_PENALTY = 0.31
LATERAL_PENALTY = 0.018
TANGENT_PENALTY = 0.18
TOP_ANCHORS = 4


@dataclass(frozen=True)
class PathHypothesis:
    vertices: tuple[tuple[int, int], ...]
    score: float
    source_anchor: tuple[int, int]
    target_anchor: tuple[int, int]
    candidates_considered: int
    ambiguous_margin: float | None


def _polyline(coords, shape):
    """Rasterize every segment densely; avoid discontinuous pixel tracks."""
    vertices = np.asarray(coords, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 2:
        raise ValueError("Expected at least two 2D points")
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite route vertices")
    output = []
    h, w = shape
    for a, b in zip(vertices[:-1], vertices[1:]):
        steps = max(2, int(math.ceil(np.linalg.norm(b - a) * 5)) + 1)
        for point in np.linspace(a, b, steps):
            y, x = np.rint(point).astype(int)
            if 0 <= y < h and 0 <= x < w:
                item = (int(y), int(x))
                if not output or output[-1] != item:
                    output.append(item)
    return tuple(output)


def _validate(heat, visible, rect, source, target):
    heat = np.asarray(heat, dtype=float)
    visible = np.asarray(visible, dtype=float)
    if heat.shape != (96, 96) or visible.shape != (96, 96):
        raise ValueError("Expected 96x96 learned heat and observed ANZA probabilities")
    if (not np.isfinite(heat).all() or not np.isfinite(visible).all()
            or heat.min() < 0 or heat.max() > 1
            or visible.min() < 0 or visible.max() > 1):
        raise ValueError("Invalid probabilities")
    if len(rect) != 4 or not all(isinstance(v, (int, np.integer)) for v in rect):
        raise ValueError("Invalid rectangle")
    y0, y1, x0, x1 = rect
    if not (0 <= y0 < y1 <= 96 and 0 <= x0 < x1 <= 96):
        raise ValueError("Rectangle outside image")
    a, b = np.asarray(source, float), np.asarray(target, float)
    if (a.shape != (2,) or b.shape != (2,)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.linalg.norm(a - b) <= 4
            or np.any(a < 0) or np.any(a >= 96)
            or np.any(b < 0) or np.any(b >= 96)):
        raise ValueError("Invalid predicted endpoints")
    return heat, visible, a, b


def _rectangle_crossings(a, b, rect):
    y0, y1, x0, x1 = rect
    seq = a + np.linspace(0, 1, 1001)[:, None] * (b - a)
    hit = ((seq[:, 0] >= y0) & (seq[:, 0] < y1)
           & (seq[:, 1] >= x0) & (seq[:, 1] < x1))
    if hit.any():
        idx = np.flatnonzero(hit)
        return seq[idx[0]], seq[idx[-1]]
    lo = np.array([y0 - 3, x0 - 3])
    hi = np.array([y1 + 2, x1 + 2])
    return np.clip(a, lo, hi), np.clip(b, lo, hi)


def _anchor_pool(prob, rect, endpoint, crossing, opposite, max_count=TOP_ANCHORS):
    """Rank distinct outside-ROI support points without expert annotation."""
    y0, y1, x0, x1 = rect
    inside = np.zeros((96, 96), bool)
    inside[y0:y1, x0:x1] = True
    ring = ndi.binary_dilation(inside, iterations=3) & ~inside
    ys, xs = np.nonzero(ring & (prob >= .45))
    if not len(ys):
        return []
    pts = np.column_stack([ys, xs]).astype(float)
    delta = opposite - endpoint
    direction = delta / max(np.linalg.norm(delta), 1e-10)
    frac = np.clip(((pts - endpoint) @ delta) / max(delta @ delta, 1e-10), 0, 1)
    foot = endpoint + frac[:, None] * delta
    distance_from_line = np.linalg.norm(pts - foot, axis=1)
    distance_from_crossing = np.linalg.norm(pts - crossing, axis=1)
    distance_from_endpoint = np.linalg.norm(pts - endpoint, axis=1)
    # A candidate belongs to the endpoint's own half of the intervention.
    # At the end point, direction is reversed so the same rule applies.
    valid = ((distance_from_crossing <= 14)
             & (distance_from_endpoint <= 30)
             & (distance_from_line <= 12)
             & (((pts - endpoint) @ direction) >= -5))
    idx = np.flatnonzero(valid)
    if not len(idx):
        return []
    quality = (5 * prob[ys, xs] - .35 * distance_from_crossing
               - .12 * distance_from_line - .025 * distance_from_endpoint)
    ordered = sorted(idx, key=lambda i: (-float(quality[i]), int(ys[i]), int(xs[i])))
    picked = []
    for i in ordered:
        point = (int(ys[i]), int(xs[i]))
        if all(math.dist(point, other[0]) >= 3.0 for other in picked):
            picked.append((point, float(quality[i])))
            if len(picked) == max_count:
                break
    return picked


def _tangent_from_visible(prob, rect, anchor, toward):
    """Weighted PCA near observed flank; return chord if support is ambiguous."""
    inside = np.zeros((96, 96), bool)
    y0, y1, x0, x1 = rect
    inside[y0:y1, x0:x1] = True
    direction = np.asarray(toward, float)
    direction /= max(float(np.linalg.norm(direction)), 1e-10)
    ys, xs = np.nonzero((prob >= .55) & ~inside)
    if len(ys) < 9:
        return direction
    pts = np.column_stack([ys, xs]).astype(float)
    d = np.linalg.norm(pts - np.asarray(anchor)[None], axis=1)
    subset = d <= 10
    if subset.sum() < 9:
        return direction
    pts = pts[subset]
    weights = np.clip(prob[ys[subset], xs[subset]] - .5, .05, 1.)
    center = np.average(pts, axis=0, weights=weights)
    dpts = pts - center
    cov = (dpts * weights[:, None]).T @ dpts / max(weights.sum(), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if eigenvalues[-1] / max(eigenvalues[0], 1e-6) < 1.3:
        return direction
    tangent = eigenvectors[:, -1]
    if tangent @ direction < 0:
        tangent = -tangent
    return tangent


def curvature_ribbon(heat, start, end, *, start_tangent=None, end_tangent=None,
                     radius=RADIUS):
    """Second-order Viterbi with fixed endpoints and tangent/curvature priors.

    Energy is accumulated on the *whole* route. Each state consists of the
    current and previous lateral offset so bending, not just steps, is penalized.
    Returns rasterized path and mean log-energy for fair anchor-pair comparison.
    """
    h = np.asarray(heat, float)
    a = np.asarray(start, float)
    b = np.asarray(end, float)
    if (h.shape != (96, 96) or not np.isfinite(h).all()
            or np.any(h < 0) or np.any(h > 1)
            or a.shape != (2,) or b.shape != (2,)
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Invalid learned map or anchors")
    delta = b - a
    distance = float(np.linalg.norm(delta))
    if distance <= 4:
        raise ValueError("Anchors are too close")
    unit = delta / distance
    normal = np.array([-unit[1], unit[0]])
    nsteps = max(16, min(96, int(math.ceil(distance * 2))))
    offsets = np.arange(-radius, radius + 1)
    kcount = len(offsets)
    center = radius
    base = np.linspace(a, b, nsteps)
    lattice = base[:, None, :] + offsets[None, :, None] * normal
    on_image = ((lattice[..., 0] >= 0) & (lattice[..., 0] < 96)
                & (lattice[..., 1] >= 0) & (lattice[..., 1] < 96))
    sampled = ndi.map_coordinates(
        h, [lattice[..., 0], lattice[..., 1]], order=1,
        mode="constant", cval=1e-5)
    reward = np.log(np.clip(sampled, 1e-5, 1)) - LATERAL_PENALTY * abs(offsets)[None]
    # Invalid outside-image states must be impossible, not merely costly.
    reward[~on_image] = -np.inf

    def desired_tangent(tangent):
        if tangent is None:
            return 0.
        tangent = np.asarray(tangent, float)
        if tangent.shape != (2,) or not np.isfinite(tangent).all():
            raise ValueError("Invalid observed tangent")
        if tangent @ unit < 0:
            tangent = -tangent
        projection = float(tangent @ unit)
        # An almost orthogonal tangent is too uncertain; use chord prior.
        if projection < .25:
            return 0.
        lateral_per_step = float(tangent @ normal) / projection * distance / (nsteps - 1)
        return float(np.clip(lateral_per_step, -2, 2))

    start_slope = desired_tangent(start_tangent)
    end_slope = desired_tangent(end_tangent)
    score = np.full((kcount, kcount), -np.inf)
    ptr = np.full((nsteps, kcount, kcount), -1, np.int16)
    for k in range(max(0, center - MAX_JUMP), min(kcount, center + MAX_JUMP + 1)):
        change = offsets[k] - offsets[center]
        score[center, k] = (reward[0, center] + reward[1, k]
                            - STEP_PENALTY * abs(change)
                            - TANGENT_PENALTY * abs(change - start_slope))
    for t in range(2, nsteps):
        nxt = np.full_like(score, -np.inf)
        for prev in range(kcount):
            old = np.flatnonzero(np.isfinite(score[:, prev]))
            if len(old) == 0:
                continue
            for k in range(max(0, prev - MAX_JUMP), min(kcount, prev + MAX_JUMP + 1)):
                if t == nsteps - 1 and k != center:
                    continue
                change = offsets[k] - offsets[prev]
                curvature = np.abs(change - (offsets[prev] - offsets[old]))
                candidates = (score[old, prev]
                              - CURVATURE_PENALTY * curvature
                              - STEP_PENALTY * abs(change))
                if t == nsteps - 1:
                    candidates -= TANGENT_PENALTY * abs(change - end_slope)
                j = int(np.argmax(candidates))
                previous_previous = int(old[j])
                nxt[prev, k] = candidates[j] + reward[t, k]
                ptr[t, prev, k] = previous_previous
        score = nxt
    # End anchor is exactly centered; next-to-last offset is optimized.
    last_prev = int(np.argmax(score[:, center]))
    best_score = float(score[last_prev, center])
    if not np.isfinite(best_score):
        raise RuntimeError("No valid curvature-constrained path")
    states = [center] * nsteps
    states[-2] = last_prev
    for t in range(nsteps - 1, 1, -1):
        states[t - 2] = int(ptr[t, states[t - 1], states[t]])
        if states[t - 2] < 0:
            raise RuntimeError("Broken Viterbi backtrace")
    if states[0] != center or states[-1] != center:
        raise RuntimeError("Decoder violated fixed endpoints")
    coords = lattice[np.arange(nsteps), states]
    return _polyline(coords, (96, 96)), best_score / nsteps


def infer_joint_path(heat, visible_prob, rect, source, target):
    """Jointly rank observed anchors and learned routes; no truth inputs."""
    heat, prob, a, b = _validate(heat, visible_prob, rect, source, target)
    cross_a, cross_b = _rectangle_crossings(a, b, rect)
    first = _anchor_pool(prob, rect, a, cross_a, b)
    last = _anchor_pool(prob, rect, b, cross_b, a)
    if not first or not last:
        # Deliberate abstention; do not silently hallucinate a straight line.
        return None
    alternatives = []
    direction = (b - a) / np.linalg.norm(b - a)
    for source_anchor, src_q in first:
        for target_anchor, dst_q in last:
            src = np.asarray(source_anchor, float)
            dst = np.asarray(target_anchor, float)
            chord = dst - src
            if (np.linalg.norm(chord) <= 4 or chord @ direction < 4):
                continue
            t0 = _tangent_from_visible(prob, rect, src, chord)
            t1 = _tangent_from_visible(prob, rect, dst, chord)
            try:
                route, energy = curvature_ribbon(
                    heat, src, dst, start_tangent=t0, end_tangent=t1)
            except RuntimeError:
                continue
            y0, y1, x0, x1 = rect
            # A geometric connection outside the intervention is not a
            # reconstruction of the missing segment, even if well supported.
            if not any(y0 <= y < y1 and x0 <= x < x1 for y, x in route):
                continue
            # Average energy (fair across different path lengths), plus a small
            # fixed anchor-quality regularizer independent of expert labels.
            geometry_quality = energy + .015 * (src_q + dst_q)
            alternatives.append((geometry_quality, source_anchor, target_anchor, route))
    if not alternatives:
        return None
    alternatives.sort(key=lambda p: (-p[0], p[1], p[2]))
    best = alternatives[0]
    margin = best[0] - alternatives[1][0] if len(alternatives) > 1 else None
    return PathHypothesis(
        vertices=best[3], score=float(best[0]), source_anchor=best[1],
        target_anchor=best[2], candidates_considered=len(alternatives),
        ambiguous_margin=float(margin) if margin is not None else None)


def decode_candidate_v52(model, scene, proposal):
    """Drop-in observational decoder for V50 learned path heatmap.

    Does not inspect proposal['y'], scene['kind'], true_ends or centerline.
    Returns (route, metadata); missing support explicitly returns route=None.
    """
    from scripts.tver_v50_conditional_path import path_heatmap

    predicted = path_heatmap(model, scene, proposal)
    outcome = infer_joint_path(
        predicted, scene["visible_prob"], scene["rectangle"],
        proposal["source"], proposal["target"])
    if outcome is None:
        return None, {"mode": "v52_joint_curvature", "status": "ABSTAIN_NO_OBSERVED_PAIR"}
    return outcome.vertices, {
        "mode": "v52_joint_curvature", "status": "JOINT_OBSERVED_ANCHORS",
        "source_anchor": outcome.source_anchor,
        "target_anchor": outcome.target_anchor,
        "mean_energy": outcome.score,
        "n_hypotheses": outcome.candidates_considered,
        "ambiguity_margin": outcome.ambiguous_margin,
        "known_intervention_roi": True,
        "expert_labels_not_used_by_decoder": True,
    }