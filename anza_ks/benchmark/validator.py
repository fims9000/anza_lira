"""Static-only benchmark validity audit, run before any symbolic score."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .matched_generator import SPLIT_SIZES, TASKS, generate_pair
from .static_signature import static_signature


def _features(task: str, split: str) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    positive = []
    negative = []
    maximum_delta = 0.0
    minimum_pixel_l2 = float("inf")
    identical = 0
    for index in range(SPLIT_SIZES[split]):
        pair = generate_pair(task, split, index)
        p_feature = static_signature(pair["positive"])
        n_feature = static_signature(pair["negative"])
        positive.append(p_feature)
        negative.append(n_feature)
        maximum_delta = max(maximum_delta, float(np.linalg.norm(p_feature - n_feature)))
        minimum_pixel_l2 = min(minimum_pixel_l2, float(pair["l2_difference"]))
        identical += int(pair["pixel_equal"])
    return np.asarray(positive), np.asarray(negative), maximum_delta, minimum_pixel_l2, identical


def validate_static_matching() -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    for task in TASKS:
        train_positive, train_negative, train_delta, train_l2, train_identical = _features(task, "train")
        dev_positive, dev_negative, dev_delta, dev_l2, dev_identical = _features(task, "dev")
        train_x = np.concatenate((train_positive, train_negative))
        train_y = np.concatenate((np.ones(len(train_positive)), np.zeros(len(train_negative))))
        dev_x = np.concatenate((dev_positive, dev_negative))
        dev_y = np.concatenate((np.ones(len(dev_positive)), np.zeros(len(dev_negative))))
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, penalty="l2", solver="liblinear", random_state=17, max_iter=500),
        )
        model.fit(train_x, train_y)
        dev_scores = model.predict_proba(dev_x)[:, 1]
        auroc = float(roc_auc_score(dev_y, dev_scores))
        diagnostics.append(
            {
                "task": task,
                "train_pairs": len(train_positive),
                "dev_pairs": len(dev_positive),
                "static_dev_auroc": auroc,
                "maximum_static_pair_delta": max(train_delta, dev_delta),
                "minimum_pixel_l2": min(train_l2, dev_l2),
                "pixel_identical_pairs": train_identical + dev_identical,
                "valid_auroc_range": 0.45 <= auroc <= 0.60,
                "static_tolerance_pass": max(train_delta, dev_delta) <= 1e-8,
            }
        )
    passed = all(row["valid_auroc_range"] and row["static_tolerance_pass"] and row["pixel_identical_pairs"] == 0 for row in diagnostics)
    return {
        "status": "ANZA_KS_STATIC_MATCH_PASS" if passed else "STOP_STATIC_MATCH_BENCH_INVALID",
        "checks_pass": passed,
        "diagnostics": diagnostics,
        "symbolic_features_accessed": False,
        "anza_ks_outputs_used_for_generation": False,
    }
