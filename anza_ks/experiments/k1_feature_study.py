"""Frozen small-readout K1 causal feature study."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ..benchmark.matched_generator import SPLIT_SIZES, TASKS, generate_pair
from ..benchmark.static_signature import static_signature
from ..constants import FEATURE_WIDTH
from ..features import METHODS, dynamic_feature_vector
from ..orientation_bank import align_patch
from ..stats.low_fpr import operating_curve, threshold_at_fpr, tpr_at_fpr_curve
from ..stats.matched_metrics import auroc, matched_ranking
from ..stats.paired_bootstrap import bootstrap_macro_ranking_delta


FIT_PAIRS = 1536
CALIBRATION_PAIRS = 512


def extract_feature(patch: np.ndarray, method: str, orientation: float) -> np.ndarray:
    if method == "K1_A_static":
        return static_signature(patch)
    aligned = align_patch(patch, orientation)
    return dynamic_feature_vector(aligned, method)


def _fit_model(positive: np.ndarray, negative: np.ndarray) -> Pipeline:
    x = np.concatenate((positive, negative))
    y = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, penalty="l2", solver="liblinear", random_state=17, max_iter=500),
    )
    model.fit(x, y)
    return model


def _model_hash(model: Pipeline) -> str:
    logistic = model.named_steps["logisticregression"]
    digest = hashlib.sha256()
    digest.update(logistic.coef_.astype(np.float64).tobytes())
    digest.update(logistic.intercept_.astype(np.float64).tobytes())
    return digest.hexdigest()


def _score(model: Pipeline, values: np.ndarray) -> np.ndarray:
    return model.predict_proba(np.asarray(values))[:, 1]


def _perturb(patch: np.ndarray, task_index: int, pair_index: int, polarity: int) -> np.ndarray:
    rng = np.random.default_rng(991_000_000 + task_index * 100_000 + pair_index * 2 + polarity)
    shifted = np.roll(np.asarray(patch, dtype=np.float64), shift=(1, -1), axis=(0, 1))
    return 0.94 * shifted + rng.normal(0.0, 0.025, size=shifted.shape)


def _summarize(
    positive: np.ndarray,
    negative: np.ndarray,
    positive_perturbed: np.ndarray,
    negative_perturbed: np.ndarray,
    calibration_threshold: float,
) -> dict[str, float | int]:
    curve_tpr, curve_fpr, curve_threshold = tpr_at_fpr_curve(positive, negative, 0.05)
    all_clean = np.concatenate((positive, negative))
    all_perturbed = np.concatenate((positive_perturbed, negative_perturbed))
    correlation = 0.0 if min(all_clean.std(), all_perturbed.std()) <= 1e-12 else float(np.corrcoef(all_clean, all_perturbed)[0, 1])
    return {
        "pair_count": len(positive),
        "matched_ranking": matched_ranking(positive, negative),
        "auroc": auroc(positive, negative),
        "curve_tpr_at_fpr05": curve_tpr,
        "curve_realized_fpr": curve_fpr,
        "curve_threshold": curve_threshold,
        "calibration_threshold": calibration_threshold,
        "development_tpr_at_calibrated_threshold": float(np.mean(positive >= calibration_threshold)),
        "development_fpr_at_calibrated_threshold": float(np.mean(negative >= calibration_threshold)),
        "perturbed_matched_ranking": matched_ranking(positive_perturbed, negative_perturbed),
        "perturbation_score_correlation": correlation,
    }


def run_k1() -> dict[str, Any]:
    raw_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, dict[str, float | int]]] = {method: {} for method in METHODS}
    score_differences: dict[str, dict[str, np.ndarray]] = {method: {} for method in METHODS}
    readout_hashes: dict[str, dict[str, str]] = {method: {} for method in METHODS}
    thresholds: dict[str, dict[str, float]] = {method: {} for method in METHODS}

    for task_index, task in enumerate(TASKS):
        train_features = {method: {"positive": [], "negative": []} for method in METHODS}
        for index in range(SPLIT_SIZES["train"]):
            pair = generate_pair(task, "train", index)
            for method in METHODS:
                train_features[method]["positive"].append(extract_feature(pair["positive"], method, pair["orientation"]))
                train_features[method]["negative"].append(extract_feature(pair["negative"], method, pair["orientation"]))

        models: dict[str, Pipeline] = {}
        for method in METHODS:
            positive = np.asarray(train_features[method]["positive"])
            negative = np.asarray(train_features[method]["negative"])
            model = _fit_model(positive[:FIT_PAIRS], negative[:FIT_PAIRS])
            calibration_negative = _score(model, negative[FIT_PAIRS:])
            threshold = threshold_at_fpr(calibration_negative, 0.05)
            models[method] = model
            thresholds[method][task] = threshold
            readout_hashes[method][task] = _model_hash(model)

        dev_features = {
            method: {"positive": [], "negative": [], "positive_perturbed": [], "negative_perturbed": []}
            for method in METHODS
        }
        static_deltas = []
        for index in range(SPLIT_SIZES["dev"]):
            pair = generate_pair(task, "dev", index)
            positive_perturbed = _perturb(pair["positive"], task_index, index, 1)
            negative_perturbed = _perturb(pair["negative"], task_index, index, 0)
            static_deltas.append(pair["static_delta"])
            for method in METHODS:
                dev_features[method]["positive"].append(extract_feature(pair["positive"], method, pair["orientation"]))
                dev_features[method]["negative"].append(extract_feature(pair["negative"], method, pair["orientation"]))
                dev_features[method]["positive_perturbed"].append(extract_feature(positive_perturbed, method, pair["orientation"]))
                dev_features[method]["negative_perturbed"].append(extract_feature(negative_perturbed, method, pair["orientation"]))

        for method in METHODS:
            scored = {name: _score(models[method], np.asarray(values)) for name, values in dev_features[method].items()}
            task_metrics = _summarize(
                scored["positive"],
                scored["negative"],
                scored["positive_perturbed"],
                scored["negative_perturbed"],
                thresholds[method][task],
            )
            metrics[method][task] = task_metrics
            score_differences[method][task] = scored["positive"] - scored["negative"]
            for index in range(SPLIT_SIZES["dev"]):
                raw_rows.append(
                    {
                        "task": task,
                        "pair_index": index,
                        "method": method,
                        "positive_score": scored["positive"][index],
                        "negative_score": scored["negative"][index],
                        "positive_perturbed_score": scored["positive_perturbed"][index],
                        "negative_perturbed_score": scored["negative_perturbed"][index],
                        "calibration_threshold": thresholds[method][task],
                        "static_pair_delta": static_deltas[index],
                    }
                )
            for point in operating_curve(scored["positive"], scored["negative"]):
                curve_rows.append({"task": task, "method": method, **point})

    macro = {
        method: {
            key: float(np.mean([metrics[method][task][key] for task in TASKS]))
            for key in ("matched_ranking", "auroc", "curve_tpr_at_fpr05", "perturbed_matched_ranking", "perturbation_score_correlation")
        }
        for method in METHODS
    }
    full_bootstrap = bootstrap_macro_ranking_delta(score_differences["K1_D_anza_ks"], score_differences["K1_A_static"])
    kolmogorov_bootstrap = bootstrap_macro_ranking_delta(score_differences["K1_D_anza_ks"], score_differences["K1_C_cat_raw"])
    anosov_bootstrap = bootstrap_macro_ranking_delta(score_differences["K1_C_cat_raw"], score_differences["K1_B_shear_raw"])
    task_gates = {}
    for task in TASKS:
        ranking_delta = float(metrics["K1_D_anza_ks"][task]["matched_ranking"] - metrics["K1_A_static"][task]["matched_ranking"])
        tpr_delta = float(metrics["K1_D_anza_ks"][task]["curve_tpr_at_fpr05"] - metrics["K1_A_static"][task]["curve_tpr_at_fpr05"])
        task_gates[task] = {"ranking_delta": ranking_delta, "tpr_delta": tpr_delta, "pass": max(ranking_delta, tpr_delta) >= 0.08}
    passing_tasks = sum(int(value["pass"]) for value in task_gates.values())
    kolmogorov_gain = float(macro["K1_D_anza_ks"]["matched_ranking"] - macro["K1_C_cat_raw"]["matched_ranking"])
    anosov_gain = float(macro["K1_C_cat_raw"]["matched_ranking"] - macro["K1_B_shear_raw"]["matched_ranking"])
    anosov_winning_tasks = sum(
        int(metrics["K1_C_cat_raw"][task]["matched_ranking"] > metrics["K1_B_shear_raw"][task]["matched_ranking"])
        for task in TASKS
    )
    full_pass = passing_tasks >= 3
    kolmogorov_pass = kolmogorov_gain >= 0.04 and kolmogorov_bootstrap["ci95_lower"] > 0
    anosov_pass = anosov_gain >= -0.02 and anosov_winning_tasks >= 2
    if not full_pass:
        status = "STOP_SYMBOLIC_DYNAMICS_NO_INCREMENTAL_SIGNAL"
    elif not kolmogorov_pass:
        status = "STOP_KOLMOGOROV_FEATURES_REDUNDANT"
    elif not anosov_pass:
        status = "STOP_ANOSOV_NOT_SPECIFIC_SHEAR_EQUAL"
    else:
        status = "ANZA_KS_CAUSAL_FEATURE_PASS"
    return {
        "status": status,
        "metrics": metrics,
        "macro": macro,
        "raw_rows": raw_rows,
        "curve_rows": curve_rows,
        "readout_hashes": readout_hashes,
        "thresholds": thresholds,
        "bootstrap": {"full_vs_static": full_bootstrap, "kolmogorov_vs_cat_raw": kolmogorov_bootstrap, "cat_raw_vs_shear": anosov_bootstrap},
        "gate": {
            "task_gates": task_gates,
            "passing_task_count": passing_tasks,
            "required_task_count": 3,
            "kolmogorov_macro_ranking_gain": kolmogorov_gain,
            "kolmogorov_pass": kolmogorov_pass,
            "anosov_macro_ranking_gain": anosov_gain,
            "anosov_winning_task_count": anosov_winning_tasks,
            "anosov_pass": anosov_pass,
            "pass": status == "ANZA_KS_CAUSAL_FEATURE_PASS",
        },
        "feature_width": FEATURE_WIDTH,
        "segmentation_training_performed": False,
        "tiny_logistic_readouts_trained": True,
        "confirm_evaluated": False,
        "K2_opened": False,
        "cracks_accessed": False,
        "expert_accessed": False,
    }
