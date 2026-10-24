"""Controlled 2D relation benchmark inspired by Graph-LIRA decomposition.

The source contours are provided by a synthetic generator (oracle tracelets);
segmentation errors are intentionally NOT evaluated in this layer. No particle
labels from actual film exist here. Do not transfer a threshold to real images.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from .pilot import propose_joins


@dataclass(frozen=True)
class Protocol:
    seed: int = 20261024
    train_scenes: int = 160
    val_scenes: int = 70
    test_scenes: int = 70
    max_gap_px: float = 53.0
    max_angle_deg: float = 38.0
    max_val_false_positive_rate: float = 0.02
    image_size: int = 320


def scene(seed: int, *, size: int = 320) -> dict:
    """Create synthetic near-parallel/curved tracks and intentionally missing links.

    The reference IDs are generator truth and never features. Different seeds
    represent independent generated scenes, not independent real detector runs.
    """
    rng = np.random.default_rng(int(seed))
    image = 227.0 + rng.normal(0, rng.uniform(4.5, 10.0), (size, size))
    h, w = image.shape
    tracks = []
    identities = []
    y0 = float(rng.uniform(97, 178))
    slope0 = float(rng.uniform(-0.22, 0.23))
    base_curvature = float(rng.uniform(-0.0009, 0.0009))
    # Deterministic stress types; no use of these identities as features.
    scenario = int(seed) % 5
    if scenario == 0:  # close parallel hard negatives
        displacement = float(rng.uniform(0.5, 3.5))
        slope_delta = float(rng.uniform(-0.012, 0.012))
    elif scenario == 1:  # near-collinear decoy
        displacement = float(rng.uniform(-1.2, 1.2))
        slope_delta = float(rng.uniform(-0.02, 0.02))
    elif scenario == 2:  # small separation, divergent beyond the gap
        displacement = float(rng.uniform(2, 7))
        slope_delta = float(rng.uniform(-0.075, 0.075))
    elif scenario == 3:  # independent near-crossing arcs
        displacement = float(rng.uniform(-7, 7))
        slope_delta = float(rng.uniform(-0.15, 0.15))
    else:  # comparatively easy separated parallel tracks
        displacement = float(rng.uniform(7, 15))
        slope_delta = float(rng.uniform(-0.04, 0.04))
    specs = [
        (y0, slope0, base_curvature),
        (y0 + displacement, slope0 + slope_delta,
         base_curvature + float(rng.uniform(-0.00045, 0.00045))),
        (y0 + float(rng.uniform(24, 55)), slope0 + float(rng.uniform(-0.32, 0.32)),
         float(rng.uniform(-0.0012, 0.0012))),
    ]
    for identity, (offset, slope, curvature) in enumerate(specs):
        darkness = float(rng.uniform(22, 110))
        thickness = int(rng.integers(1, 4))
        # Some entire continuations disappear: true NONE examples remain.
        has_right = bool(rng.random() > 0.28) or identity == 0
        for side, (start, stop) in enumerate(((30, 140), (161, 289))):
            if side == 1 and not has_right:
                continue
            xs = np.arange(start, stop + 1, dtype=float)
            ys = offset + slope * (xs - 150) + curvature * (xs - 150) ** 2
            ys += rng.normal(0, 0.6 if scenario < 4 else 0.25, len(xs))
            xy = np.column_stack((xs, ys))
            pts = np.column_stack((ys, xs))
            for j in range(len(xy) - 1):
                a, b = np.rint(xy[j]).astype(int), np.rint(xy[j + 1]).astype(int)
                if np.any(a < 0) or np.any(b < 0) or a[0] >= w or b[0] >= w or a[1] >= h or b[1] >= h:
                    continue
                # Fragmentary bubble occupancy, consistent with each synthetic trace.
                if rng.random() >= 0.12:
                    cv2.line(image, tuple(a), tuple(b), color=max(0.0, 227.0 - darkness), thickness=thickness, lineType=cv2.LINE_8)
            tracks.append(pts.astype(np.float64))
            identities.append((identity, side))
    # Scattered ink specks, not ground truth labels.
    specks = rng.random((size, size)) < rng.uniform(0.0004, 0.0025)
    image[specks] -= rng.uniform(35, 85)
    return {"image": np.clip(image, 0, 255).astype(np.uint8), "segments": tracks,
            "identity_side": identities, "scene_id": int(seed)}


def _segment_darkness(image: np.ndarray, segment: np.ndarray) -> float:
    # The local intensity evidence is observed only, never ground-truth instance ID.
    xy = np.rint(segment[8:-8, ::-1]).astype(np.int64)
    if len(xy) == 0:
        return 0.0
    xx = np.clip(xy[:, 0], 0, image.shape[1]-1)
    yy = np.clip(xy[:, 1], 0, image.shape[0]-1)
    return float(np.median(225.0 - image[yy, xx].astype(np.float64)))


def _curve_error(p: np.ndarray, q: np.ndarray, side_p: str, side_q: str) -> float:
    """Compare extrapolated endpoint y across a gap in the global x axis.

    The study generator is x-monotonic, so this feature is valid HERE but does
    not define a general bubble-chamber feature for vertical/spiral tracks.
    """
    point_p = p if side_p == "B" else p[::-1]
    point_q = q if side_q == "A" else q[::-1]
    window_p = point_p[-min(25, len(point_p)):]
    window_q = point_q[:min(25, len(point_q))]
    # Beware extrapolation over a too-large gap: diagnostic only.
    xp, yp = window_p[:, 1], window_p[:, 0]
    xq, yq = window_q[:, 1], window_q[:, 0]
    center_x = float((xp[-1] + xq[0]) * 0.5)
    a = np.polyfit(xp - center_x, yp, 2)
    b = np.polyfit(xq - center_x, yq, 2)
    return float(np.clip(abs(a[2] - b[2]), 0.0, 100.0))


def examples(sample: dict, protocol: Protocol) -> tuple[list[dict], int]:
    segments = sample['segments']
    pairs = propose_joins(segments, max_gap_px=protocol.max_gap_px, max_join_angle_deg=protocol.max_angle_deg, max_candidates_per_endpoint=6)
    ids = sample['identity_side']
    true_possible = sum(1 for track_id, side in ids if side == 0 and (track_id, 1) in ids)
    rows = []
    darkness = [_segment_darkness(sample['image'], s) for s in segments]
    for p in pairs:
        ia, ib = p['segment_A'], p['segment_B']
        a_id, a_side = ids[ia]
        b_id, b_side = ids[ib]
        true_link = bool(a_id == b_id and a_side != b_side and ((a_side == 0 and p['endpoint_A'] == 'B' and p['endpoint_B'] == 'A') or (b_side == 0 and p['endpoint_B'] == 'B' and p['endpoint_A'] == 'A')))
        contrast_a, contrast_b = darkness[ia], darkness[ib]
        feats_geo = [p['gap_px'], p['angle_A_deg'], p['angle_B_deg'], _curve_error(segments[ia], segments[ib], p['endpoint_A'], p['endpoint_B'])]
        feats_img = [abs(contrast_a - contrast_b), min(contrast_a, contrast_b), max(contrast_a, contrast_b)]
        rows.append({'scene_id': sample['scene_id'], 'pair': [ia, p['endpoint_A'], ib, p['endpoint_B']],
                     'truth_same_track': true_link, 'geometry': feats_geo, 'image_evidence': feats_img})
    return rows, true_possible


def create_split(protocol: Protocol, *, group: str) -> dict:
    intervals = {'train': (0, protocol.train_scenes), 'validation': (protocol.train_scenes, protocol.val_scenes),
                 'test': (protocol.train_scenes + protocol.val_scenes, protocol.test_scenes)}
    start, count = intervals[group]
    rows = []
    possible = 0
    present = 0
    for idx in range(start, start + count):
        item, count_true = examples(scene(protocol.seed + idx, size=protocol.image_size), protocol)
        rows.extend(item)
        possible += count_true
        present += sum(int(x['truth_same_track']) for x in item)
    return {'rows': rows, 'total_true_links': possible, 'candidate_true_links': present,
            'scene_count': count, 'seed_start': protocol.seed + start}


def threshold_at_fpr(y_true: np.ndarray, prob: np.ndarray, limit: float) -> float:
    """Choose strict threshold on validation ONLY, never on held-out test."""
    negatives = (y_true == 0)
    if not negatives.any():
        raise ValueError('validation lacks NONE/hard negatives')
    # Candidates include a zero-accept threshold above all scores.
    thresholds = np.r_[np.nextafter(float(prob.max()), np.inf), np.unique(prob)]
    admissible = [float(t) for t in thresholds if float(((prob >= t) & negatives).sum() / negatives.sum()) <= limit]
    if not admissible:
        return float(np.nextafter(prob.max(), np.inf))
    # Min threshold -> max positive recall within the fixed FPR constraint.
    return min(admissible)


def _features(rows: list[dict], with_image: bool) -> tuple[np.ndarray, np.ndarray]:
    values = [[*r['geometry'], *(r['image_evidence'] if with_image else [])] for r in rows]
    return np.asarray(values, dtype=float), np.asarray([r['truth_same_track'] for r in rows], dtype=int)


def _metrics(rows: list[dict], predictions: np.ndarray, labels: np.ndarray, scores: np.ndarray, *, threshold: float) -> dict:
    accepted = predictions >= threshold
    t = labels > 0
    tp = int((accepted & t).sum())
    fp = int((accepted & ~t).sum())
    fn = int((~accepted & t).sum())
    tn = int((~accepted & ~t).sum())
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'threshold_validation_selected': threshold,
        'recall_candidates': round(tp / max(tp + fn, 1), 6),
        'false_positive_rate_candidates': round(fp / max(fp + tn, 1), 6),
        'false_joins_per_scene': round(fp / max(len(set(x['scene_id'] for x in rows)), 1), 6),
        'accepted_candidates': int(accepted.sum()),
        'auroc': round(float(roc_auc_score(labels, scores)), 6) if 0 < labels.sum() < len(labels) else None,
        'auprc': round(float(average_precision_score(labels, scores)), 6) if labels.sum() else None,
        'automatic_merges_in_real_pilot': 0,
    }


def run_study(protocol: Protocol = Protocol()) -> dict:
    splits = {k: create_split(protocol, group=k) for k in ('train', 'validation', 'test')}
    outputs = {}
    for name, with_image in (('geometry_only', False), ('geometry_plus_image', True)):
        x_train, y_train = _features(splits['train']['rows'], with_image)
        x_val, y_val = _features(splits['validation']['rows'], with_image)
        x_test, y_test = _features(splits['test']['rows'], with_image)
        if len(set(y_train)) < 2:
            raise RuntimeError('not enough positive/negative train candidates')
        model = Pipeline([('scale', StandardScaler()), ('classifier', LogisticRegression(max_iter=1500, class_weight='balanced', random_state=protocol.seed))])
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)[:, 1]
        thr = threshold_at_fpr(y_val, val_prob, protocol.max_val_false_positive_rate)
        test_prob = model.predict_proba(x_test)[:, 1]
        outputs[name] = {'validation': _metrics(splits['validation']['rows'], val_prob, y_val, val_prob, threshold=thr),
                         'test': _metrics(splits['test']['rows'], test_prob, y_test, test_prob, threshold=thr),
                         'model_features': 'geometry and source-image darkness' if with_image else 'geometry only'}
    return {
        'status': 'SYNTHETIC_RELATION_ISOLATION_ONLY',
        'protocol': asdict(protocol),
        'splits': {k: {field: v for field, v in data.items() if field != 'rows'} | {'candidate_count': len(data['rows'])} for k, data in splits.items()},
        'results': outputs,
        'interpretation_warning': 'Oracle synthetic fragments and scene design: cannot claim real segmentation, run transfer, JINR precision, physical validation, or superiority of ANZA.'
    }
