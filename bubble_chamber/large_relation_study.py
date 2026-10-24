"""Frame-disjoint Graph-LIRA coordinate-only experiment on visualMetro exports.

This is **not** training/evaluation of AZConv segmentation: the measurements
have no corresponding BMP. All occlusions are deliberately simulated by
withholding expert clicks; model inputs never include truth track identifiers.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .real_sparse_lira import SparseProtocol, _accept, _metrics, _threshold_at_fpr, make_examples


@dataclass(frozen=True)
class LargeRelationProtocol:
    seed: int = 20261024
    cuts: tuple[float, ...] = (0.35, 0.5, 0.65)
    k_candidates: int = 5
    missing_right_fraction: float = 0.18
    max_validation_candidate_fpr: float = 0.02
    global_margin: float = 0.05
    bootstrap_repetitions: int = 400
    missing_clicks: int = 1


def frame_partition(frame_ids: list[str], seed: int) -> dict[str, list[str]]:
    """Stable groups: train 65%, validation 18%, test 17% by WHOLE frame.

    One frame's 3 views, tracks, and artificial gaps always share a partition.
    Different frames in the same film are NOT independent runs.
    """
    unique = sorted(set(frame_ids))
    if len(unique) < 20:
        raise ValueError("need >=20 whole frames for held-out protocol")
    shuffled = sorted(unique, key=lambda x: sha256(f"{seed}|{x}|frame-holdout-v1".encode()).digest())
    n = len(shuffled)
    n_train = int(0.65 * n)
    n_val = int(0.18 * n)
    split = {"train": sorted(shuffled[:n_train]), "val": sorted(shuffled[n_train:n_train+n_val]), "test": sorted(shuffled[n_train+n_val:])}
    if min(map(len, split.values())) < 3:
        raise ValueError("too few frames in one partition")
    assert len(set(split['train']) & set(split['val'])) == 0
    assert len(set(split['train']) & set(split['test'])) == 0
    assert len(set(split['test']) & set(split['val'])) == 0
    return split


def select_models(seed: int):
    """Controlled comparisons: no tuning on held-out test."""
    return {
        "distance_direction_fixed": ("selection_score", None),
        "basic_logreg": ("geometry_basic", make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=seed))),
        "generic_logreg": ("geometry_generic", make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=seed))),
        "axial_logreg": ("geometry_axial_fuzzy", make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=seed))),
        "generic_hgb": ("geometry_generic", HistGradientBoostingClassifier(max_iter=130, max_leaf_nodes=15, min_samples_leaf=30, learning_rate=0.06, l2_regularization=5.0, random_state=seed)),
        "axial_hgb": ("geometry_axial_fuzzy", HistGradientBoostingClassifier(max_iter=130, max_leaf_nodes=15, min_samples_leaf=30, learning_rate=0.06, l2_regularization=5.0, random_state=seed)),
    }


def _predict(model, feature: str, rows: list[dict]) -> np.ndarray:
    if feature == "selection_score":
        return np.exp(-np.clip(np.asarray([r[feature] for r in rows], dtype=np.float64), 0, 100))
    x = np.asarray([r[feature] for r in rows], dtype=np.float64)
    if not bool(np.isfinite(x).all()):
        raise ValueError("nonfinite feature in held-out partition")
    return model.predict_proba(x)[:, 1]


def _get_metrics(rows: list[dict], scores: np.ndarray, threshold: float, margin: float) -> dict[str, Any]:
    out = _metrics(rows, scores, threshold, margin)
    out['total_candidates'] = len(rows)
    out['positive_candidates'] = sum(bool(r['truth_same_track']) for r in rows)
    out['scenes'] = len({r['scene_id'] for r in rows})
    out['distinct_frames'] = len({r['frame_id'] for r in rows})
    return out


def _bootstrap_frame_ci(rows: list[dict], accepted: np.ndarray, *, repeats: int, seed: int) -> dict[str, list[float]]:
    """Cluster-resample **frames**, not pixels, points or individual candidates."""
    by_frame = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    for row, accepted_here in zip(rows, accepted):
        truth = bool(row['truth_same_track'])
        slot = 0 if (accepted_here and truth) else 1 if (accepted_here and not truth) else 2 if truth else 3
        by_frame[row['frame_id']][slot] += 1
    counts = np.stack(list(by_frame.values()))
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(counts), size=(repeats, len(counts)))
    sums = counts[boot].sum(axis=1)
    recall = sums[:, 0] / np.maximum(1, sums[:, 0] + sums[:, 2])
    fpr = sums[:, 1] / np.maximum(1, sums[:, 1] + sums[:, 3])
    return {"recall_ci95": [round(float(x), 6) for x in np.quantile(recall, [0.025, 0.975])],
            "fpr_ci95": [round(float(x), 6) for x in np.quantile(fpr, [0.025, 0.975])]}


def study_large_manifest(manifest: dict[str, Any], cfg: LargeRelationProtocol = LargeRelationProtocol()):
    if manifest.get("audit_schema") != "visualmetro_large_audit_v1":
        raise ValueError("expected audited large visualMetro manifest; do not feed unvalidated frames")
    if set(manifest['summary']['distinct_film_ids']) != {'070'}:
        raise ValueError("this release protocol is explicitly limited to film 070")
    frames = [f['frame_id'] for f in manifest['frames']]
    splits = frame_partition(frames, cfg.seed)
    sub_cfg = SparseProtocol(cuts=cfg.cuts, k_candidates=cfg.k_candidates, seed=cfg.seed,
                             missing_right_fraction=cfg.missing_right_fraction,
                             max_val_fpr=cfg.max_validation_candidate_fpr, margin=cfg.global_margin,
                             missing_clicks=cfg.missing_clicks)
    rows, candidate_stats = make_examples(manifest, sub_cfg)
    sets = {s: [r for r in rows if r['frame_id'] in set(fids)] for s, fids in splits.items()}
    if any(not r for r in sets.values()):
        raise ValueError('empty split')
    results = {}
    pred_rows = []
    truth_train = np.asarray([int(r['truth_same_track']) for r in sets['train']])
    assert set(truth_train) == {0, 1}
    for model_id, (field, clf) in select_models(cfg.seed).items():
        if clf is not None:
            clf.fit(np.asarray([r[field] for r in sets['train']], dtype=np.float64), truth_train)
        vscore = _predict(clf, field, sets['val'])
        yval = np.asarray([int(r['truth_same_track']) for r in sets['val']])
        threshold = _threshold_at_fpr(yval, vscore, cfg.max_validation_candidate_fpr)
        tscore = _predict(clf, field, sets['test'])
        tacc = _accept(sets['test'], tscore, threshold, cfg.global_margin)
        all_labels = np.asarray([int(r['truth_same_track']) for r in sets['test']])
        out = {
            'feature_group': field,
            'feature_count': (len(sets['train'][0][field]) if field!='selection_score' else 1),
            'threshold_selected_on_validation': round(float(threshold), 10),
            'val': _get_metrics(sets['val'], vscore, threshold, cfg.global_margin),
            'test': _get_metrics(sets['test'], tscore, threshold, cfg.global_margin),
            'frame_bootstrap_ci95_test': _bootstrap_frame_ci(sets['test'], tacc, repeats=cfg.bootstrap_repetitions, seed=cfg.seed+101),
            'candidate_auc_test': round(float(roc_auc_score(all_labels, tscore)), 6),
            'candidate_ap_test': round(float(average_precision_score(all_labels, tscore)), 6),
            'max_test_acceptance_per_scene': max((sum(1 for i, r in enumerate(sets['test']) if r['scene_id']==sid and tacc[i]) for sid in {r['scene_id'] for r in sets['test']}), default=0),
        }
        results[model_id] = out
        for row, score, accepted in zip(sets['test'], tscore, tacc):
            pred_rows.append({k: row[k] for k in ('frame_id','view','scene_id','left_id','right_id','truth_same_track','candidate_rank')} | {
                'model': model_id, 'score': round(float(score), 8), 'accepted_offline': bool(accepted),
            })

    pstats = {k: v for k, v in candidate_stats.items() if k != 'scenes'}
    return ({
        'experiment_id': 'S5_REAL_070_FRAME_HOLDOUT_V1',
        'status': 'REAL_EXPERT_SPARSE_GEOMETRY_SIMULATED_GAPS_ONLY',
        'dataset_summary': manifest['summary'],
        'provenance_sha256': manifest['provenance_sha256'],
        'parameters': asdict(cfg),
        'split_frame_ids': splits,
        'split_counts': {key: {'frames': len(split), 'candidates': len(sets[key]),
                              'positive_candidates': sum(int(r['truth_same_track']) for r in sets[key])}
                         for key, split in splits.items()},
        'candidate_generation': pstats,
        'variants': results,
        'limitations': [
            'All train, validation and test frames are from the same film 070: no run or medium transfer demonstrated.',
            'Measured points are real operator annotations, but missing segments are generated by hiding measured clicks.',
            'No paired BMPs, no image-based ANZA convolution or real segmentation accuracy can be evaluated.',
            'Axial engineered geometry features are ANZA-inspired only: not the trained AZConv implementation.',
            'The accepted_offline field is benchmark output, NOT permission for unattended physical track merging.',
        ]
    }, pred_rows)


def save_study(manifest: dict, output: Path, cfg: LargeRelationProtocol = LargeRelationProtocol()):
    metrics, predictions = study_large_manifest(manifest, cfg)
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    import csv
    with (output / 'test_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    return metrics
