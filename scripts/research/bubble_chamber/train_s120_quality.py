"""Fit a private S120 review classifier on manually marked fragments.

Cross-validation groups by complete human stroke. This is still the SAME
film frame, not a run-disjoint evaluation or calibrated physical truth.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold


def model():
    return RandomForestClassifier(
        n_estimators=200, max_depth=11, min_samples_leaf=12,
        n_jobs=4, random_state=12, class_weight='balanced_subsample')


def train(features_file: Path, model_file: Path, report_file: Path,
          threshold: float = .5):
    z = np.load(features_file)
    labels = z['labels']
    mask = labels != 'unknown'
    X = z['features'][mask]
    y = (labels[mask] == 'track').astype(int)
    groups = z['label_ids'][mask]
    counts = Counter(groups.tolist())
    if len(counts) < 10 or len(set(y)) != 2:
        raise ValueError('Insufficient independent annotated stroke IDs')
    weights = np.asarray([1./np.sqrt(counts[int(k)]) for k in groups])
    weights /= weights.mean()
    out_of_fold = np.full(len(y), np.nan)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=27)
    for train_rows, test_rows in folds.split(X, y, groups):
        fit = model()
        fit.fit(X[train_rows], y[train_rows], sample_weight=weights[train_rows])
        out_of_fold[test_rows] = fit.predict_proba(X[test_rows])[:, 1]
    if not np.isfinite(out_of_fold).all():
        raise RuntimeError('Missing out-of-fold predictions')
    cm = confusion_matrix(y, out_of_fold >= threshold, labels=[0, 1])
    final = model()
    final.fit(X, y, sample_weight=weights)
    report = {
        'feature_names': z['names'].tolist(),
        'label_source_sha256': str(z['source_sha256']),
        'labeled_curve_instances': len(y),
        'distinct_annotated_strokes': len(counts),
        'manual_track_instances': int(y.sum()),
        'manual_trash_instances': int((1-y).sum()),
        'cross_val': '5-fold grouped by human stroke; SAME FRAME, NOT independent test',
        'oof_auc': float(roc_auc_score(y, out_of_fold)),
        'display_threshold': float(threshold),
        'oof_track_instance_retention': float(cm[1, 1]/max(1, cm[1].sum())),
        'oof_trash_instance_retention': float(cm[0, 1]/max(1, cm[0].sum())),
        'unlabeled_instances_not_negative_class': True,
        'score_not_calibrated_probability': True,
        'accuracy_on_unlabeled_006_unknown': True,
    }
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_file)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(train(a.features, a.model, a.report), indent=2))
