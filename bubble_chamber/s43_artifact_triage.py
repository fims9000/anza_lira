"""S43: synthetic-origin artifact ranking, NOT classification of native particles.

Uses geometry and a strip intensity profile. Never writes 0/1 labels for real
H50Z, deletes a candidate, or authorizes an auto-join. Fiducials are handled
separately using genuine calibration-mark metadata from S40.
"""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .s39_machine_auto import ridge_evidence, sample_bilinear

SCHEMA = 's43_simulator_trained_artifact_similarity_advisory_v1'
FEATURE_NAMES = (
    'log_geodesic_length', 'sinuosity', 'p90_local_turn_degrees',
    'median_flank_minus_center', 'mean_flank_minus_center',
    'std_flank_minus_center', 'p90_flank_minus_center',
    'fraction_dark_against_flanks_8', 'along_path_highpass_fraction',
    'darkness_contrast_p90_minus_p10', 'median_near_flank_darkness',
    'near_to_far_contrast_ratio', 'hessian_top5_support',
    'best_scales_top5_support',
)


def resample_polyline(points: np.ndarray | list, shape: tuple[int, int], n: int = 80) -> np.ndarray:
    p = np.asarray(points, dtype='float64')
    h, w = shape
    if (p.ndim != 2 or p.shape[1] != 2 or len(p) < 3
            or not np.isfinite(p).all() or np.any(p < 0)
            or np.any(p[:, 0] >= w) or np.any(p[:, 1] >= h)
            or not 24 <= n <= 150):
        raise ValueError('Invalid/unregistered candidate polyline')
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = np.r_[True, d > 1e-6]
    p = p[keep]
    if len(p) < 3:
        raise ValueError('Degenerate candidate centerline')
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if length[-1] < 7:
        raise ValueError('Candidate too short for strip features')
    arc = np.linspace(0, length[-1], n)
    return np.column_stack((np.interp(arc, length, p[:, 0]),
                            np.interp(arc, length, p[:, 1])))


def path_features(gray: np.ndarray, pts: np.ndarray | list,
                  *, maps: dict | None = None) -> np.ndarray:
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 96:
        raise ValueError('Expected raw grayscale crop')
    p = resample_polyline(pts, gray.shape)
    if maps is None:
        maps = ridge_evidence(gray)
    if maps['score'].shape != gray.shape or maps['scales'].shape[1:] != gray.shape:
        raise ValueError('Hessian map / H50Z crop mismatch')
    v = np.gradient(p, axis=0)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    n = np.column_stack((-v[:, 1], v[:, 0]))
    midpoint = sample_bilinear(gray, p)
    def strip(offset: float) -> np.ndarray:
        return (sample_bilinear(gray, p + offset * n) +
                sample_bilinear(gray, p - offset * n)) / 2.0
    near, far, outer = strip(2.5), strip(6.), strip(10.)
    contrast = far - midpoint
    near_contrast = near - midpoint
    med = float(np.median(contrast))
    sm = gaussian_filter1d(contrast, 2., mode='nearest')
    highpass = float(np.std(contrast - sm) / (np.std(contrast) + 1e-4))
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(p[-1] - p[0]))
    dot = np.clip(np.sum(v[1:] * v[:-1], axis=1), -1, 1)
    angle = np.degrees(np.arccos(dot))
    p95 = float(np.quantile(maps['score'], .95))
    ridge = float(np.mean(sample_bilinear(maps['score'], p) >= p95)) if p95 > 0 else 0.
    per_scale = []
    for scale in maps['scales']:
        q95 = float(np.quantile(scale, .95))
        per_scale.append(float(np.mean(sample_bilinear(scale, p) >= q95)) if q95 > 0 else 0.)
    feat = np.array([
        np.log1p(length), np.clip(chord / max(length, 1e-6), 0, 1),
        np.quantile(angle, .90), med, np.mean(contrast), np.std(contrast),
        np.quantile(contrast, .90), np.mean(contrast > 8), highpass,
        np.quantile(contrast, .90) - np.quantile(contrast, .10),
        np.median(near_contrast), np.median(near_contrast) / (abs(med) + 1.),
        ridge, max(per_scale),
    ], dtype='float64')
    if len(feat) != len(FEATURE_NAMES) or not np.isfinite(feat).all():
        raise ValueError('Invalid machine-only feature vector')
    return feat


def score_native_candidates(gray: np.ndarray, candidates: list[dict],
                            classifier: Any) -> list[dict]:
    """Advisory ranking only: no labels, no deletions, no physical auto-join."""
    if not candidates:
        return []
    maps = ridge_evidence(gray)
    ids = [c['draft_id'] for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError('Ambiguous native proposal identities')
    x = np.stack([path_features(gray, c['points_xy'], maps=maps) for c in candidates])
    sim_scores = classifier.predict_proba(x)[:, 1]
    if np.any(~np.isfinite(sim_scores)) or np.any((sim_scores < 0) | (sim_scores > 1)):
        raise ValueError('Invalid synthetic-origin model outputs')
    return [{
        'draft_id': ident,
        'SIMULATED_measured_curve_similarity_NOT_NATIVE_PROBABILITY': round(float(value), 6),
        'native_particle_identity': 'UNKNOWN',
        'candidate_kept': True,
        'can_auto_join_native_tracks': False,
    } for ident, value in zip(ids, sim_scores)]


def annotate_graph_advisory(graph: dict, native_scores: list[dict],
                            s40_guard: dict) -> dict:
    """Exact ID/edge mapping, fiducial protection; scores never grant a join."""
    by_id = {r['draft_id']: r for r in native_scores}
    if len(by_id) != len(native_scores):
        raise ValueError('Duplicate native candidate ID')
    nodes = graph['nodes']
    edges = graph['candidate_edges']
    prior = s40_guard['candidate_edges']
    if len(edges) != len(prior) or len(s40_guard['nodes']) != len(nodes):
        raise ValueError('Graph/fiducial ordering mismatch')
    node_rank = []
    for node in nodes:
        source_ids = node['source_candidate_ids']
        if not source_ids or any(x not in by_id for x in source_ids):
            raise ValueError('Missing source candidate for collapsed node')
        node_rank.append(float(np.mean([
            by_id[k]['SIMULATED_measured_curve_similarity_NOT_NATIVE_PROBABILITY']
            for k in source_ids])))
    out = []
    for e, fid in zip(edges, prior):
        i, j = e['i'], e['j']
        if not (0 <= i < len(nodes) and 0 <= j < len(nodes) and i != j
                and fid.get('i') == i and fid.get('j') == j):
            raise ValueError('Graph edge provenance mismatch')
        overlap = bool(fid['fiducial_overlap'])
        out.append({
            'i': i, 'j': j,
            'ranked_endpoint_similarity_SIMULATOR_ONLY': round(min(node_rank[i], node_rank[j]), 6),
            'fiducial_risk': overlap,
            'status': 'FIDUCIAL_RISK_ABSTAIN' if overlap else 'UNVERIFIED_MACHINE_JOIN_NO_AUTO_ACCEPT',
            'native_same_particle_relation': 'UNKNOWN',
            'physical_auto_join_performed': False,
        })
    return {'schema': SCHEMA, 'nodes_count': len(nodes), 'edges': out,
            'fiducial_risk_edges': sum(bool(e['fiducial_risk']) for e in out),
            'real_physical_auto_joins': 0}


def export_simulator_linear_model(clf: Any, *, source_photo: str) -> dict:
    """Portable numeric weights; avoids unsafe pickle and false native labels."""
    if source_photo not in {'005', '006'}:
        raise ValueError('Unknown H50Z photo source')
    scale = clf.named_steps['standardscaler']
    model = clf.named_steps['logisticregression']
    if (len(model.classes_) != 2 or not np.array_equal(model.classes_, [0, 1])
            or len(scale.mean_) != len(FEATURE_NAMES)
            or model.coef_.shape != (1, len(FEATURE_NAMES))):
        raise ValueError('Unexpected S43 classifier shape')
    return {
        'schema': SCHEMA, 'trained_on_H50Z_photo': source_photo,
        'feature_names': list(FEATURE_NAMES),
        'standard_scaler_mean': scale.mean_.tolist(),
        'standard_scaler_scale': scale.scale_.tolist(),
        'logistic_coefficients': model.coef_[0].tolist(),
        'logistic_intercept': float(model.intercept_[0]),
        'training_labels_are_synthetic_origins_only': True,
        'native_particle_GT_observations': 0,
        'real_physics_autojoin_allowed': False,
    }


def infer_simulator_linear_model(features: np.ndarray, saved: dict) -> np.ndarray:
    """Exact portable inference; output is a simulator score, NOT physics probability."""
    from scipy.special import expit
    x = np.asarray(features, dtype=float)
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all()
            or saved.get('schema') != SCHEMA
            or saved.get('feature_names') != list(FEATURE_NAMES)
            or saved.get('native_particle_GT_observations') != 0
            or saved.get('real_physics_autojoin_allowed') is not False):
        raise ValueError('Unsafe/mismatched simulator model checkpoint')
    mu = np.asarray(saved['standard_scaler_mean'], dtype=float)
    sd = np.asarray(saved['standard_scaler_scale'], dtype=float)
    weights = np.asarray(saved['logistic_coefficients'], dtype=float)
    bias = float(saved['logistic_intercept'])
    if (any(v.shape != (len(FEATURE_NAMES),) for v in (mu, sd, weights))
            or not np.isfinite([*mu,*sd,*weights,bias]).all()
            or (sd <= 0).any()):
        raise ValueError('Invalid portable linear parameters')
    return expit(((x - mu) / sd) @ weights + bias)
