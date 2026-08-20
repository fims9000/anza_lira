"""One-factor K1.5 Shear+KS causal control on frozen K1 samples."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from anza_ks.benchmark.matched_generator import SPLIT_SIZES, TASKS, generate_pair
from anza_ks.experiments.k1_feature_study import FIT_PAIRS, _fit_model, _model_hash, _perturb, _score, _summarize
from anza_ks.orientation_bank import align_patch
from anza_ks.stats.low_fpr import threshold_at_fpr

from .features import shear_ks_feature_vector


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results/anza_ks/k0_k1"
RESULT = ROOT / "results/anza_ks/k1_5"
METHOD = "K1_E_shear_ks"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature(patch: np.ndarray, orientation: float) -> np.ndarray:
    return shear_ks_feature_vector(align_patch(patch, orientation))


def _paired_tpr_bootstrap(
    candidate: dict[str, np.ndarray], control: dict[str, np.ndarray], *, resamples: int = 10_000, seed: int = 1_519_019
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros(resamples, dtype=np.float64)
    observed = []
    for task in TASKS:
        delta = np.asarray(candidate[task], dtype=np.float64) - np.asarray(control[task], dtype=np.float64)
        observed.append(float(delta.mean()))
        indices = rng.integers(0, len(delta), size=(resamples, len(delta)))
        bootstrap += delta[indices].mean(axis=1) / len(TASKS)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "observed_macro_tpr_delta": float(np.mean(observed)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "resamples": resamples,
        "unit": "paired independent dev pair at each method's frozen full-dev FPR<=0.05 operating point, task-macro averaged",
    }


def run_k1_5() -> dict[str, Any]:
    parent_rows = list(csv.DictReader((PARENT / "per_pair.csv").open()))
    parent_metrics = json.loads((PARENT / "metrics.json").read_text())["k1"]
    parent_d = {(row["task"], int(row["pair_index"])): row for row in parent_rows if row["method"] == "K1_D_anza_ks"}
    if len(parent_d) != len(TASKS) * SPLIT_SIZES["dev"]:
        raise ValueError("frozen K1-D rows are incomplete")

    metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    readout_hashes: dict[str, str] = {}
    thresholds: dict[str, float] = {}
    candidate_indicators: dict[str, np.ndarray] = {}
    control_indicators: dict[str, np.ndarray] = {}

    for task_index, task in enumerate(TASKS):
        positive_train = []
        negative_train = []
        for index in range(SPLIT_SIZES["train"]):
            pair = generate_pair(task, "train", index)
            positive_train.append(_feature(pair["positive"], pair["orientation"]))
            negative_train.append(_feature(pair["negative"], pair["orientation"]))
        positive_train_array = np.asarray(positive_train)
        negative_train_array = np.asarray(negative_train)
        model = _fit_model(positive_train_array[:FIT_PAIRS], negative_train_array[:FIT_PAIRS])
        threshold = threshold_at_fpr(_score(model, negative_train_array[FIT_PAIRS:]), 0.05)
        thresholds[task] = threshold
        readout_hashes[task] = _model_hash(model)

        dev = {name: [] for name in ("positive", "negative", "positive_perturbed", "negative_perturbed")}
        for index in range(SPLIT_SIZES["dev"]):
            pair = generate_pair(task, "dev", index)
            dev["positive"].append(_feature(pair["positive"], pair["orientation"]))
            dev["negative"].append(_feature(pair["negative"], pair["orientation"]))
            dev["positive_perturbed"].append(_feature(_perturb(pair["positive"], task_index, index, 1), pair["orientation"]))
            dev["negative_perturbed"].append(_feature(_perturb(pair["negative"], task_index, index, 0), pair["orientation"]))
        scores = {name: _score(model, np.asarray(value)) for name, value in dev.items()}
        metrics[task] = _summarize(scores["positive"], scores["negative"], scores["positive_perturbed"], scores["negative_perturbed"], threshold)
        e_threshold = float(metrics[task]["curve_threshold"])
        d_threshold = float(parent_metrics["metrics"]["K1_D_anza_ks"][task]["curve_threshold"])
        candidate_indicators[task] = (scores["positive"] >= e_threshold).astype(np.float64)
        control_indicators[task] = np.asarray(
            [float(parent_d[(task, index)]["positive_score"]) >= d_threshold for index in range(SPLIT_SIZES["dev"])], dtype=np.float64
        )
        for index in range(SPLIT_SIZES["dev"]):
            rows.append({
                "task": task,
                "pair_index": index,
                "method": METHOD,
                "positive_score": float(scores["positive"][index]),
                "negative_score": float(scores["negative"][index]),
                "positive_perturbed_score": float(scores["positive_perturbed"][index]),
                "negative_perturbed_score": float(scores["negative_perturbed"][index]),
                "calibration_threshold": threshold,
            })

    macro = {key: float(np.mean([metrics[task][key] for task in TASKS])) for key in (
        "matched_ranking", "auroc", "curve_tpr_at_fpr05", "perturbed_matched_ranking", "perturbation_score_correlation"
    )}
    parent_macro = parent_metrics["macro"]["K1_D_anza_ks"]
    delta_tpr = float(parent_macro["curve_tpr_at_fpr05"] - macro["curve_tpr_at_fpr05"])
    bootstrap = _paired_tpr_bootstrap(control_indicators, candidate_indicators)
    specific_pass = delta_tpr >= 0.04 and bootstrap["ci95_lower"] > 0
    status = "ANOSOV_KS_SPECIFIC_PASS" if specific_pass else "SYMBOLIC_INFORMATION_PASS_ANOSOV_NOT_SPECIFIC"
    return {
        "status": status,
        "method": METHOD,
        "metrics": metrics,
        "macro": macro,
        "cat_ks_macro": parent_macro,
        "cat_ks_minus_shear_ks_macro_tpr": delta_tpr,
        "bootstrap": bootstrap,
        "readout_hashes": readout_hashes,
        "thresholds": thresholds,
        "rows": rows,
        "parent_artifacts": {
            "package_sha256": "cd4de1fb01551e616acab9270f984726a8c92264892b2a98559d68001a56df67",
            "per_pair_sha256": _sha256(PARENT / "per_pair.csv"),
            "source_manifest_sha256": _sha256(PARENT / "freeze/source_manifest.json"),
        },
        "old_readouts_retrained": False,
        "confirm_evaluated": False,
        "segmentation_training_performed": False,
        "k2_authorized": True,
    }


def save_k1_5(result: dict[str, Any]) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    rows = result.pop("rows")
    (RESULT / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (RESULT / "per_pair.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = [
        "# ANZA-KS K1.5 Factorial Report", "", f"Status: `{result['status']}`", "",
        "Only the new Shear+KS readout was fitted. Frozen K1 A-D readouts, features, scores, confirm, CRACKS, and expert data were not opened or changed.", "",
        "| Comparison | Macro TPR@FPR<=0.05 | Paired CI |", "|---|---:|---:|",
        f"| CatKS - ShearKS | {result['cat_ks_minus_shear_ks_macro_tpr']:+.6f} | [{result['bootstrap']['ci95_lower']:+.6f}, {result['bootstrap']['ci95_upper']:+.6f}] |", "",
        "The Anosov-specific attribution gate requires delta >= 0.04 and a strictly positive lower 95% paired-bootstrap bound.",
    ]
    (RESULT / "ANZA_KS_K1_5_FACTORIAL_REPORT.md").write_text("\n".join(report) + "\n")
