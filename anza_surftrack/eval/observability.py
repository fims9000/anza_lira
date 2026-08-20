"""Center-only non-identifiability and adjacent-context observability audit."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..synthetic3d.families import observability_dataset


def evaluate_observability() -> dict:
    train_x, train_y, _ = observability_dataset("geom_train", 12_000)
    dev_x, dev_y, context = observability_dataset("geom_dev_iid", 4_000)
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=41).fit(train_x, train_y)
    center_score = classifier.predict_proba(dev_x)[:, 1]
    center_auc = float(roc_auc_score(dev_y, center_score))
    context_top1 = float(np.mean((context > 0).astype(np.int8) == dev_y))
    return {
        "center_feature_names": ["endpoint_distance", "axial_tangent_delta", "curvature", "local_branch_length"],
        "center_auroc": center_auc, "context_oracle_top1": context_top1,
        "center_features_surface_id_free": True, "context_uses_adjacent_history": True,
        "center_gate_pass": 0.45 <= center_auc <= 0.55,
        "context_gate_pass": context_top1 >= 0.85,
        "confirm_accessed": False,
    }
