"""Exploratory Graph-LIRA relation learning on *real sparse visualMetro tracks*.

Real expert coordinates/track identity are used, but occlusions are synthetic.
There are NO paired BMPs for film 070, so this is not an image segmentation
benchmark. A 7-frame leave-one-frame-out analysis has ONE shared film: it is
not evidence of cross-run generalisation or novel segmentation performance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .visualmetro_import import SCHEMA


@dataclass(frozen=True)
class SparseProtocol:
    cuts: tuple[float, ...] = (0.35, 0.50, 0.65)
    k_candidates: int = 5
    seed: int = 20261024
    missing_right_fraction: float = 0.18
    max_val_fpr: float = 0.02
    margin: float = 0.05


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.acos(np.clip(float(np.dot(a, b)), -1, 1)))


def _curvature(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    v = np.diff(points, axis=0)
    units = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    cross = units[:-1, 0] * units[1:, 1] - units[:-1, 1] * units[1:, 0]
    return float(np.mean(cross))


def _steps(points: np.ndarray) -> float:
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return max(float(np.median(d)), 1.0)


def _local_residual(points: np.ndarray) -> float:
    xs = points - points.mean(0)
    singular = np.linalg.svd(xs, compute_uv=False)
    return float(singular[-1] / max(singular[0], 1e-6))


def _features(left: np.ndarray, right: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    """All features depend on observable left and right points ONLY.

    Axial fuzzy support is ANZA-inspired, not the trained AZConv convolution.
    We include a capacity-increased generic geometric comparator separately.
    """
    p, q = left[-1], right[0]
    v = q - p
    distance = max(float(np.linalg.norm(v)), 1e-6)
    direction = v / distance
    left_axis = _unit(left[-1] - left[-3])
    right_axis = _unit(right[2] - right[0])
    la = _angle(left_axis, direction)
    ra = _angle(right_axis, direction)
    ia = _angle(left_axis, right_axis)
    sl, sr = _steps(left), _steps(right)
    scale = math.sqrt(sl * sr)
    kl = _curvature(left[-min(5, len(left)):])
    kr = _curvature(right[:min(5, len(right))])
    rl = _local_residual(left[-min(5, len(left)):])
    rr = _local_residual(right[:min(5, len(right))])
    basic = [distance / scale, la, ra]
    generic = basic + [ia, abs(kl - kr), abs(sl - sr) / scale, rl, rr]

    # Anisotropic fuzzy bilateral support aligned with each locally observed
    # direction: q = d' M d, M = R(theta) diag(sig_parallel^-2,
    # sig_perp^-2) R(theta)^T. Here sigmas are fixed from observed geometry,
    # not learned from hidden target labels or film metadata.
    def axial_support(axis: np.ndarray, coherence: float) -> tuple[float, float, float]:
        longitudinal = float(np.dot(v, axis) / scale)
        perpendicular = float(np.linalg.det(np.column_stack((axis, v))) / scale)
        anisotropy = 2.0 + 2.0 * max(0.0, 1.0 - coherence)
        qf = (longitudinal / 4.0) ** 2 + (perpendicular / anisotropy) ** 2
        membership = math.exp(-3.0 * coherence)
        return math.exp(-0.5 * min(qf, 120)), membership, abs(perpendicular) / anisotropy

    a = axial_support(left_axis, rl)
    b = axial_support(right_axis, rr)
    axial = generic + [a[0], b[0], math.sqrt(a[0] * b[0]), abs(a[0] - b[0]),
                       math.sqrt(a[1] * b[1]), max(a[2], b[2])]
    return basic, generic, axial


def _split_track(points: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(points)
    if n < 7:
        raise ValueError("need at least seven sparse expert points")
    # Leave at least one full expert click unobserved. Near-short track cuts
    # have fewer distinct available positions; no gap labels are invented.
    left_end = min(max(2, int(round((n - 1) * fraction))), n - 5)
    right_start = left_end + 2
    assert len(points[:left_end+1]) >= 3 and len(points[right_start:]) >= 3
    return points[:left_end + 1], points[right_start:]


def _omit(seed: int, frame: str, view: int, track: str, cut: int, rate: float) -> bool:
    digest = sha256(f"{seed}|{frame}|{view}|{track}|{cut}|right-removed".encode()).digest()
    return int.from_bytes(digest[:8], 'little') / 2**64 < rate


def make_examples(manifest: dict, config: SparseProtocol = SparseProtocol()) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("requires a validated visualMetro sparse manifest")
    if not (0 <= config.missing_right_fraction < 1) or config.k_candidates < 1:
        raise ValueError("invalid candidate protocol")
    if any(not 0.0 < c < 1.0 for c in config.cuts):
        raise ValueError("cut fractions out of range")
    rows: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    for frame in manifest['frames']:
        fid = frame['frame_id']
        tracks = [t for t in frame['tracks'] if t['n_points'] >= 7]
        for view in (1, 3, 2):
            for cut_index, fraction in enumerate(config.cuts):
                scene = f"{fid}:v{view}:cut{cut_index}"
                left, right = [], []
                for tr in tracks:
                    p = np.asarray(tr['measured_pixel_xy'][str(view)], dtype=float)
                    l, r = _split_track(p, fraction)
                    left.append((tr['track_id'], l))
                    if not _omit(config.seed, fid, view, tr['track_id'], cut_index, config.missing_right_fraction):
                        right.append((tr['track_id'], r))
                positives_possible = sum(int(t in dict(right)) for t, _ in left)
                found = 0
                for left_id, x in left:
                    candidates = []
                    for right_id, y in right:
                        basic, generic, axial = _features(x, y)
                        # This selection score is label-blind. No true candidates
                        # are forced into the pool after ranking.
                        score = basic[0] + 2.0 * (basic[1] + basic[2])
                        candidates.append((score, right_id, basic, generic, axial))
                    candidates.sort(key=lambda item: (item[0], item[1]))
                    for rank, (score, right_id, basic, generic, axial) in enumerate(candidates[:config.k_candidates]):
                        truth = left_id == right_id
                        found += int(truth)
                        rows.append({
                            'frame_id': fid, 'film_id': frame['film_id'], 'view': view,
                            'scene_id': scene, 'cut_index': cut_index,
                            'left_id': left_id, 'right_id': right_id,
                            'candidate_rank': rank, 'truth_same_track': truth,
                            'geometry_basic': basic, 'geometry_generic': generic,
                            'geometry_axial_fuzzy': axial,
                            'selection_score': score,
                        })
                scenes.append({'scene_id': scene, 'frame_id': fid, 'view': view,
                               'cut_index': cut_index, 'left_tracks': len(left),
                               'right_tracks': len(right), 'positive_possible': positives_possible,
                               'positive_in_candidates': found})
    return rows, {'scenes': scenes, 'frame_count': len(manifest['frames']),
                  'true_possible': sum(s['positive_possible'] for s in scenes),
                  'candidate_true': sum(s['positive_in_candidates'] for s in scenes),
                  'candidates': len(rows), 'positive_rows': sum(r['truth_same_track'] for r in rows),
                  'negative_rows': sum(not r['truth_same_track'] for r in rows),
                  'all_missing_right_is_NONE': True,
                  'data_basis': 'real human-marked coordinates, artificial withheld interior clicks'}


def _xy(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([r[key] for r in rows], dtype=np.float64), np.asarray([int(r['truth_same_track']) for r in rows], dtype=int)


def _threshold_at_fpr(labels: np.ndarray, scores: np.ndarray, bound: float) -> float:
    negatives = ~labels.astype(bool)
    if not np.any(negatives):
        raise ValueError('validation needs negative candidates')
    thresholds = np.r_[np.nextafter(float(scores.max()), np.inf), np.unique(scores)]
    feasible = [t for t in thresholds if ((scores >= t) & negatives).sum() / negatives.sum() <= bound]
    return float(min(feasible))


def _accept(rows: list[dict], scores: np.ndarray, threshold: float, margin: float) -> np.ndarray:
    """One-to-one, mutual best continuation; flags none if competitor too close."""
    accepted = np.zeros(len(rows), dtype=bool)
    by_left, by_right = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        by_left[(row['scene_id'], row['left_id'])].append(i)
        by_right[(row['scene_id'], row['right_id'])].append(i)
    rank_left = {k: sorted(ids, key=lambda i: (-scores[i], i)) for k, ids in by_left.items()}
    rank_right = {k: sorted(ids, key=lambda i: (-scores[i], i)) for k, ids in by_right.items()}
    for i, row in enumerate(rows):
        if scores[i] < threshold:
            continue
        al, ar = rank_left[(row['scene_id'], row['left_id'])], rank_right[(row['scene_id'], row['right_id'])]
        if al[0] != i or ar[0] != i:
            continue
        if len(al) > 1 and scores[i] - scores[al[1]] < margin:
            continue
        if len(ar) > 1 and scores[i] - scores[ar[1]] < margin:
            continue
        accepted[i] = True
    return accepted


def _metrics(rows: list[dict], scores: np.ndarray, threshold: float, margin: float) -> dict:
    y = np.asarray([r['truth_same_track'] for r in rows], dtype=bool)
    accepted = _accept(rows, scores, threshold, margin)
    tp, fp = int((accepted & y).sum()), int((accepted & ~y).sum())
    fn, tn = int((~accepted & y).sum()), int((~accepted & ~y).sum())
    with_pos = len(set(r['scene_id'] for r in rows))
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'recall_on_present_candidates': round(tp / max(tp+fn, 1), 6),
            'fpr_on_negative_candidates': round(fp / max(fp+tn, 1), 6),
            'false_links_per_scene': round(fp / max(with_pos, 1), 6),
            'review_needed': int((~accepted).sum()),
            'candidate_auroc': round(float(roc_auc_score(y, scores)), 6) if len(np.unique(y)) == 2 else None,
            'candidate_ap': round(float(average_precision_score(y, scores)), 6) if np.any(y) else None}


def evaluate_leave_one_frame_out(manifest: dict, cfg: SparseProtocol = SparseProtocol()) -> dict[str, Any]:
    rows, manifest_stats = make_examples(manifest, cfg)
    frames = sorted({r['frame_id'] for r in rows})
    if len(frames) < 4:
        raise ValueError('need >=4 whole frames for nested train/val/test')
    specs = {'basic': ('geometry_basic', 'simple'),
             'generic_poly2': ('geometry_generic', 'polynomial'),
             'axial_fuzzy': ('geometry_axial_fuzzy', 'simple')}
    predictions = []
    for fold, test_frame in enumerate(frames):
        validation_frame = frames[(fold+1) % len(frames)]
        splits = {'train': [r for r in rows if r['frame_id'] not in (test_frame, validation_frame)],
                  'val': [r for r in rows if r['frame_id'] == validation_frame],
                  'test': [r for r in rows if r['frame_id'] == test_frame]}
        fold_result = {'test_frame': test_frame, 'validation_frame': validation_frame,
                       'train_frames': sorted(set(r['frame_id'] for r in splits['train'])),
                       'protocol': 'whole-frame split, still same film: exploratory only', 'variants': {}}
        for name, (field, model_kind) in specs.items():
            xx, yy = _xy(splits['train'], field)
            xv, yv = _xy(splits['val'], field)
            xt, yt = _xy(splits['test'], field)
            if not np.all(np.isfinite(xx)) or set(yy) != {0, 1}:
                raise ValueError('invalid training candidates')
            if model_kind == 'polynomial':
                model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), LogisticRegression(max_iter=2500, C=0.15, class_weight='balanced', random_state=cfg.seed))
            else:
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2500, C=0.5, class_weight='balanced', random_state=cfg.seed))
            model.fit(xx, yy)
            val_prob = model.predict_proba(xv)[:, 1]
            threshold = _threshold_at_fpr(yv, val_prob, cfg.max_val_fpr)
            pred = model.predict_proba(xt)[:, 1]
            val_metrics = _metrics(splits['val'], val_prob, threshold, cfg.margin)
            test_metrics = _metrics(splits['test'], pred, threshold, cfg.margin)
            fold_result['variants'][name] = {'threshold_from_validation': round(threshold, 8),
                                              'val': val_metrics, 'test': test_metrics,
                                              'n_features': int(xx.shape[1]),
                                              'train_samples': len(xx), 'val_samples': len(xv), 'test_samples': len(xt)}
            predictions.extend({'fold': fold, 'test_frame': test_frame, 'view': row['view'],
                                'scene_id': row['scene_id'], 'variant': name,
                                'left_id': row['left_id'], 'right_id': row['right_id'],
                                'truth': int(row['truth_same_track']), 'score': float(score),
                                'chosen': bool(chosen)} for row, score, chosen in zip(splits['test'], pred, _accept(splits['test'], pred, threshold, cfg.margin)))
        yield fold_result, predictions, manifest_stats


def run_sparse_study(manifest: dict, cfg: SparseProtocol = SparseProtocol()) -> tuple[dict[str, Any], list[dict]]:
    results, preds, stats = [], [], {}
    for result, rows, stats in evaluate_leave_one_frame_out(manifest, cfg):
        results.append(result)
        preds = rows
    variants = {}
    for name in results[0]['variants']:
        x = [res['variants'][name]['test'] for res in results]
        totals = {key: sum(m[key] for m in x) for key in ('tp', 'fp', 'fn', 'tn')}
        variants[name] = {**totals,
                          'candidate_recall': round(totals['tp']/max(totals['tp']+totals['fn'], 1), 6),
                          'candidate_fpr': round(totals['fp']/max(totals['fp']+totals['tn'], 1), 6),
                          'per_frame_auroc': {r['test_frame']:r['variants'][name]['test']['candidate_auroc'] for r in results},
                          'mean_per_frame_auroc': round(float(np.mean([x['candidate_auroc'] for x in x if x['candidate_auroc'] is not None])), 6)}
    return ({'status': 'EXPLORATORY_SAME_FILM_SPARSE_COORDINATE_GRAPH_ONLY',
             'source': 'user visualMetro film 070: 7 frames, 42 expert identified tracks, 3 views',
             'real_images_matched': 0,
             'annotation_semantics': 'same-track labels from visualMetro IDs; gaps and removed right branches are artificial',
             'protocol': asdict(cfg), 'dataset_summary': stats, 'folds': results, 'aggregate': variants,
             'scientific_limitations': ['all 7 frames belong to the SAME film', 'not real image segmentation',
                                       'ANZA-inspired axial features are not a trained AZConv CNN',
                                       'interior gaps are simulated by withholding expert points',
                                       'real automatic merging remains forbidden']}, preds)
