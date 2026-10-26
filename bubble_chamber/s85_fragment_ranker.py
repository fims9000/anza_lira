"""S85: learned ranking of *extra* pixel-derived 2D curve hypotheses.

This inference module NEVER accesses source GT, synthetic nodes, particle IDs,
physical labels or camera-run identities. Model fitting is permitted only on
separate TRAIN scenes through an evaluator-only script.

The S72 original line objects and ordering are immutable. No physical joins.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree

FEATURE_NAMES = (
    'mean_local_dark_contrast',
    'p20_local_dark_contrast',
    'fraction_positive_dark_contrast',
    'relative_dark_contrast_variation',
    'mean_edge_gradient',
    'length_normalized',
    'straightness',
    'mean_absolute_turn',
    'nearest_original_coverage',
    'top_contact_probability',
    'top_contact_margin',
    'free_contact_fraction',
    'two_port_bridge_available',
)
FEATURE_SCHEMA = 'S85_PIXEL_GRAPH_FEATURES_V1'
MODEL_SCHEMA = 'S85_TRAIN_ONLY_REGULARIZED_LOGISTIC_V1'
ARMS = ('baseline', 's85_learned_p85', 's85_learned_p95')
THRESHOLDS = {'s85_learned_p85': .85, 's85_learned_p95': .95}
MIN_POSITIVES = 12
MIN_NEGATIVES = 24
MIN_GROUPS_EACH_CLASS = 4
RESAMPLE = 31
SIDE_OFFSETS = (3., 5.)


def _curve(curve):
    p = np.asarray(curve, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 8 or not np.isfinite(p).all():
        raise ValueError('Expected finite pixel-derived (N,2) curve of length >=8')
    return p


def _sample_geometry(curve):
    p = _curve(curve)
    length = np.linalg.norm(np.diff(p, axis=0), axis=1)
    arc = np.r_[0., np.cumsum(length)]
    good = np.r_[True, length > 1e-6]
    arc, p = arc[good], p[good]
    if len(arc) < 2 or arc[-1] < 6:
        return None
    def interpolate(frac):
        position = np.asarray(frac) * arc[-1]
        return np.column_stack((np.interp(position, arc, p[:, 0]), np.interp(position, arc, p[:, 1])))
    fractions = np.linspace(.04, .96, RESAMPLE)
    center = interpolate(fractions)
    tangents = interpolate(np.clip(fractions + .018, 0, 1)) - interpolate(np.clip(fractions - .018, 0, 1))
    norm = np.linalg.norm(tangents, axis=1)
    if np.any(norm < 1e-7):
        return None
    unit = tangents / norm[:, None]
    normal = np.column_stack((-unit[:, 1], unit[:, 0]))
    return center, normal, unit, float(arc[-1])


def _samples(image, points):
    return cv2.remap(image, points[:, 0].astype(np.float32)[None, :],
                     points[:, 1].astype(np.float32)[None, :],
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)[0]


def _image_features(gray: np.ndarray, curve):
    if not isinstance(gray, np.ndarray) or gray.shape != (192, 192) or gray.dtype != np.uint8:
        raise ValueError('Only 192x192 uint8 pixels; neither GT nor IDs accepted')
    geom = _sample_geometry(curve)
    if geom is None:
        return np.zeros(8, dtype=np.float64)
    xy, normal, tangent, length = geom
    points = [xy]
    for offset in SIDE_OFFSETS:
        points += [xy + offset * normal, xy - offset * normal]
    all_points = np.vstack(points)
    if (np.any(all_points < 0) or np.any(all_points[:, 0] > 191) or
            np.any(all_points[:, 1] > 191)):
        return np.zeros(8, dtype=np.float64)
    image = gray.astype(np.float32) / 255.
    values = _samples(image, all_points).reshape(len(points), RESAMPLE)
    darkness = np.maximum(values[1:].mean(axis=0) - values[0], 0.)
    smooth = cv2.GaussianBlur(image, (0, 0), .8)
    grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    grad_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
    gradients = _samples(np.hypot(grad_x, grad_y), xy)
    change = np.linalg.norm(np.diff(tangent, axis=0), axis=1)
    chord = np.linalg.norm(xy[-1] - xy[0])
    return np.array([
        float(darkness.mean()), float(np.quantile(darkness, .20)),
        float(np.mean(darkness > .008)),
        float(darkness.std() / (darkness.mean() + .005)),
        float(gradients.mean()), min(1., length / 192.),
        float(np.clip(chord / length, 0., 1.)),
        float(np.mean(change)),
    ], dtype=np.float64)


def candidate_features(gray: np.ndarray, candidate, originals: Sequence,
                       contacts: Sequence[Mapping], occupied_ports: set[tuple[int,int]]):
    """Produce one 13-number, order-invariant feature vector from image + S56.

    contacts are S84's strict-S59 extra-to-original proposals with fields:
    score, original_port, extra_port, original. They contain no GT.
    """
    a = _image_features(gray, candidate)
    p = _curve(candidate)
    if originals:
        original_points = np.vstack([_curve(o) for o in originals])
        coverage = float(np.mean(cKDTree(original_points).query(p)[0] < 3.))
    else:
        coverage = 0.
    valid = []
    for c in contacts:
        q = float(c['score'])
        port = tuple(c['original_port'])
        end = int(c['extra_port'])
        orig = int(c['original'])
        if (not np.isfinite(q) or not (0 <= q <= 1) or
                len(port) != 2 or port[1] not in (0, -1) or
                end not in (0, -1) or orig < 0 or orig >= len(originals)):
            raise ValueError('Malformed S56 image-only graph contact')
        valid.append((q, port, end, orig))
    valid.sort(key=lambda z: (-z[0], z[3], z[2]))
    best = valid[0][0] if valid else 0.
    margin = max(0., best - valid[1][0]) if len(valid) > 1 else (best if valid else 0.)
    free = [v for v in valid if v[1] not in occupied_ports]
    bridge = any(a[2] != b[2] and a[3] != b[3] and a[1] != b[1]
                 for i, a in enumerate(free) for b in free[i+1:])
    features = np.r_[a, coverage, best, margin,
                     len(free) / max(1, len(valid)), float(bridge)]
    if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
        raise AssertionError('Invalid S85 feature vector')
    return features.astype(np.float64)


def fit_train_only(features, labels, scene_groups, *, random_state=85):
    """Train on disjoint synthetic TRAIN scene groups; fail closed if sparse.

    Logistic regression is regularized and numerically frozen. Labels are only
    supplied by a separate TRAIN evaluator. This function does not create GT.
    """
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    groups = np.asarray(scene_groups)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or y.ndim != 1 or len(y) != len(X) or len(groups) != len(y):
        raise ValueError('Feature/label/group dimensionality mismatch')
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Non-finite features or invalid TRAIN labels/groups')
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    gp, gn = len(np.unique(groups[y == 1])), len(np.unique(groups[y == 0]))
    base = {'schema': MODEL_SCHEMA, 'feature_schema': FEATURE_SCHEMA,
            'feature_names': list(FEATURE_NAMES), 'train_positives': pos,
            'train_negatives': neg, 'train_positive_scene_groups': gp,
            'train_negative_scene_groups': gn, 'trained': False,
            'physical_auto_joins_authorized': False}
    if (pos < MIN_POSITIVES or neg < MIN_NEGATIVES or
            gp < MIN_GROUPS_EACH_CLASS or gn < MIN_GROUPS_EACH_CLASS):
        return {**base, 'reason': 'INSUFFICIENT_UNAMBIGUOUS_TRAIN_LABELS'}
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler = StandardScaler().fit(X)
    standardized = scaler.transform(X)
    counts = np.bincount(y.astype(int), minlength=2)
    # Balance class frequencies and downweight scenes with many repeated nuisance views.
    totals = {str(g): int(np.sum(groups == g)) for g in np.unique(groups)}
    weights = np.array([len(y)/(2*counts[int(label)]) / totals[str(group)]
                        for label, group in zip(y, groups)], dtype=float)
    weights *= len(weights) / weights.sum()
    model = LogisticRegression(C=.25, solver='liblinear', random_state=random_state,
                               max_iter=1000, tol=1e-9)
    model.fit(standardized, y, sample_weight=weights)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError('Unstable fitted S85 TRAIN model')
    return {**base, 'trained': True, 'reason': 'TRAIN_ONLY',
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'coef': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0])}


def predict_probability(frozen_model: Mapping, features) -> float:
    if (frozen_model.get('schema') != MODEL_SCHEMA or
            frozen_model.get('feature_schema') != FEATURE_SCHEMA or
            frozen_model.get('feature_names') != list(FEATURE_NAMES)):
        raise ValueError('Frozen TRAIN model has different feature schema')
    if not frozen_model.get('trained'):
        return 0.  # Fail closed. Never pretend an unavailable model is accurate.
    a = np.asarray(features, dtype=float)
    if a.shape != (len(FEATURE_NAMES),) or not np.isfinite(a).all():
        raise ValueError('Invalid image-only inference vector')
    fields = ('scaler_mean', 'scaler_scale', 'coef')
    params = [np.asarray(frozen_model[k], dtype=float) for k in fields]
    if any(p.shape != a.shape or not np.isfinite(p).all() for p in params):
        raise ValueError('Malformed frozen TRAIN weights')
    mean, scale, coef = params
    if np.any(scale <= 0) or not np.isfinite(frozen_model['intercept']):
        raise ValueError('Invalid model scaling or intercept')
    t = float(np.dot((a-mean)/scale, coef) + frozen_model['intercept'])
    return float(np.exp(-np.logaddexp(0., -t)))


def select_extras(gray, original_lines, consensus_lines, contact_map, occupied_ports,
                  frozen_model, arm):
    if arm not in ARMS or len(original_lines) > 32 or not len(original_lines) <= len(consensus_lines) <= 32:
        raise ValueError('Invalid arm or source line budget')
    if any(x is not y for x, y in zip(original_lines, consensus_lines)):
        raise ValueError('Must retain all original S72 instance identities')
    source = list(original_lines)
    extras = consensus_lines[len(source):]
    if arm == 'baseline' or not frozen_model.get('trained'):
        return {'polylines': source, 'extra_considered': len(extras),
                'extra_accepted': 0, 'extra_rejected': len(extras),
                'reason': 'BASELINE' if arm == 'baseline' else 'UNTRAINED_ABSTAIN',
                'physical_auto_joins_authorized': False}
    keep = []
    for j, line in enumerate(extras):
        f = candidate_features(gray, line, source, contact_map.get(j, []), occupied_ports)
        if predict_probability(frozen_model, f) >= THRESHOLDS[arm]:
            keep.append(line)
    return {'polylines': source+keep, 'extra_considered': len(extras),
            'extra_accepted': len(keep), 'extra_rejected': len(extras)-len(keep),
            'reason': 'LEARNED_ADVISORY', 'physical_auto_joins_authorized': False}


def model_sha256(model: Mapping) -> str:
    return hashlib.sha256(json.dumps(model, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()
