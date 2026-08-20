#!/usr/bin/env python3
"""Bounded evaluator-only repair for the frozen ANZA-KS K1 OR gate.

The original run correctly froze and computed all scores, but its Kolmogorov and
Anosov gates inspected matched ranking only. The master specification permits
TPR@FPR0.05 OR matched ranking. This audit consumes immutable score rows and
changes no model, feature, benchmark, split, or threshold policy.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from anza_ks.benchmark.matched_generator import TASKS
from anza_ks.stats.low_fpr import tpr_at_fpr_curve


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "anza_ks" / "k0_k1"
METHODS = ("K1_A_static", "K1_B_shear_raw", "K1_C_cat_raw", "K1_D_anza_ks")
RESAMPLES = 10_000
SEED = 941_019


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_scores() -> dict[str, dict[str, dict[str, np.ndarray]]]:
    collected = {method: {task: {"positive": [], "negative": []} for task in TASKS} for method in METHODS}
    with (RESULT_ROOT / "per_pair.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            target = collected[row["method"]][row["task"]]
            target["positive"].append(float(row["positive_score"]))
            target["negative"].append(float(row["negative_score"]))
    return {
        method: {task: {name: np.asarray(values) for name, values in score.items()} for task, score in tasks.items()}
        for method, tasks in collected.items()
    }


def _rowwise_tpr_at_fpr(positive: np.ndarray, negative: np.ndarray, maximum_fpr: float = 0.05) -> np.ndarray:
    if positive.shape != negative.shape or positive.ndim != 2:
        raise ValueError("bootstrap score matrices must align")
    count = negative.shape[1]
    allowed = int(np.floor(maximum_fpr * count))
    if allowed == 0:
        threshold = np.nextafter(np.max(negative, axis=1), np.inf)
    else:
        index = count - allowed - 1
        boundary = np.partition(negative, index, axis=1)[:, index]
        threshold = np.nextafter(boundary, np.inf)
    return np.mean(positive >= threshold[:, None], axis=1)


def bootstrap_macro_tpr_delta(
    candidate: dict[str, dict[str, np.ndarray]],
    control: dict[str, dict[str, np.ndarray]],
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    chunk_size: int = 200,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros(resamples, dtype=np.float64)
    observed = []
    for task in TASKS:
        c = candidate[task]
        b = control[task]
        observed.append(
            tpr_at_fpr_curve(c["positive"], c["negative"], 0.05)[0]
            - tpr_at_fpr_curve(b["positive"], b["negative"], 0.05)[0]
        )
        count = len(c["positive"])
        for start in range(0, resamples, chunk_size):
            stop = min(start + chunk_size, resamples)
            indices = rng.integers(0, count, size=(stop - start, count))
            c_tpr = _rowwise_tpr_at_fpr(c["positive"][indices], c["negative"][indices])
            b_tpr = _rowwise_tpr_at_fpr(b["positive"][indices], b["negative"][indices])
            bootstrap[start:stop] += (c_tpr - b_tpr) / len(TASKS)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "observed_macro_tpr_delta": float(np.mean(observed)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "resamples": int(resamples),
        "seed": int(seed),
        "unit": "paired development example within each frozen task, task-macro averaged",
    }


def run_audit() -> dict[str, Any]:
    metrics_path = RESULT_ROOT / "metrics.json"
    pair_path = RESULT_ROOT / "per_pair.csv"
    original = json.loads(metrics_path.read_text())
    if original["status"] != "STOP_KOLMOGOROV_FEATURES_REDUNDANT":
        raise ValueError("gate audit requires the original frozen K1 STOP")
    scores = _load_scores()
    kolmogorov_tpr = bootstrap_macro_tpr_delta(scores["K1_D_anza_ks"], scores["K1_C_cat_raw"])
    anosov_tpr = bootstrap_macro_tpr_delta(scores["K1_C_cat_raw"], scores["K1_B_shear_raw"], seed=SEED + 1)
    full_tpr = bootstrap_macro_tpr_delta(scores["K1_D_anza_ks"], scores["K1_A_static"], seed=SEED + 2)
    original_gate = original["k1"]["gate"]
    ranking_k_gain = float(original_gate["kolmogorov_macro_ranking_gain"])
    ranking_k_ci = original["k1"]["bootstrap"]["kolmogorov_vs_cat_raw"]
    kolmogorov_paths = {
        "ranking": ranking_k_gain >= 0.04 and ranking_k_ci["ci95_lower"] > 0,
        "tpr_at_fpr05": kolmogorov_tpr["observed_macro_tpr_delta"] >= 0.04 and kolmogorov_tpr["ci95_lower"] > 0,
    }
    task_tpr = {}
    for task in TASKS:
        cat = original["k1"]["metrics"]["K1_C_cat_raw"][task]["curve_tpr_at_fpr05"]
        shear = original["k1"]["metrics"]["K1_B_shear_raw"][task]["curve_tpr_at_fpr05"]
        task_tpr[task] = float(cat - shear)
    anosov_ranking_gain = float(original_gate["anosov_macro_ranking_gain"])
    ranking_wins = int(original_gate["anosov_winning_task_count"])
    tpr_wins = sum(int(delta > 0) for delta in task_tpr.values())
    anosov_paths = {
        "ranking": anosov_ranking_gain >= -0.02 and ranking_wins >= 2,
        "tpr_at_fpr05": anosov_tpr["observed_macro_tpr_delta"] >= -0.02 and tpr_wins >= 2,
    }
    full_pass = int(original_gate["passing_task_count"]) >= int(original_gate["required_task_count"])
    kolmogorov_pass = any(kolmogorov_paths.values())
    anosov_pass = any(anosov_paths.values())
    corrected_pass = full_pass and kolmogorov_pass and anosov_pass
    status = "ANZA_KS_CAUSAL_FEATURE_PASS" if corrected_pass else original["status"]
    result = {
        "audit_version": "ANZA_KS_K1_GATE_AUDIT_R1",
        "status": status,
        "root_cause": "Original evaluator implemented ranking-only mechanism gates although the frozen master specification permits TPR@FPR0.05 OR matched ranking.",
        "inputs": {
            "metrics_sha256": _sha256(metrics_path),
            "per_pair_sha256": _sha256(pair_path),
            "protocol_sha256": original["protocol_sha256"],
            "benchmark_sha256": original["benchmark_sha256"],
            "source_sha256": original["source_sha256"],
        },
        "full_vs_static": {"original_task_pass": full_pass, "tpr_bootstrap": full_tpr},
        "kolmogorov_vs_cat_raw": {
            "ranking_macro_delta": ranking_k_gain,
            "ranking_ci95": [ranking_k_ci["ci95_lower"], ranking_k_ci["ci95_upper"]],
            "tpr_bootstrap": kolmogorov_tpr,
            "allowed_metric_paths": kolmogorov_paths,
            "pass": kolmogorov_pass,
        },
        "cat_raw_vs_shear": {
            "ranking_macro_delta": anosov_ranking_gain,
            "ranking_winning_tasks": ranking_wins,
            "task_tpr_deltas": task_tpr,
            "tpr_winning_tasks": tpr_wins,
            "tpr_bootstrap": anosov_tpr,
            "allowed_metric_paths": anosov_paths,
            "pass": anosov_pass,
        },
        "corrected_gate_pass": corrected_pass,
        "tiny_readouts_retrained": False,
        "features_recomputed": False,
        "benchmark_changed": False,
        "confirm_evaluated": False,
        "K2_opened": False,
        "cracks_accessed": False,
        "expert_accessed": False,
    }
    output = RESULT_ROOT / "gate_audit_r1.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = [
        "# ANZA-KS K1 gate audit R1",
        "",
        f"Corrected research status: `{status}`",
        "",
        "This is an evaluator-only repair over immutable K1 score rows. No readout was retrained and no feature, benchmark, split, confirm sample, CRACKS, or expert data was changed or opened.",
        "",
        f"- Full symbolic vs static task gate: `{'PASS' if full_pass else 'FAIL'}`.",
        f"- Kolmogorov vs CatRaw macro TPR delta: `{kolmogorov_tpr['observed_macro_tpr_delta']:+.6f}`, 95% CI `[{kolmogorov_tpr['ci95_lower']:+.6f}, {kolmogorov_tpr['ci95_upper']:+.6f}]`, gate `{'PASS' if kolmogorov_pass else 'FAIL'}`.",
        f"- CatRaw vs shear macro TPR delta: `{anosov_tpr['observed_macro_tpr_delta']:+.6f}`, TPR wins `{tpr_wins}/5`, gate `{'PASS' if anosov_pass else 'FAIL'}`.",
        "",
        f"K2 authorized by corrected frozen gate: **{'yes' if corrected_pass else 'no'}**. K2 was not opened in this audit.",
    ]
    (RESULT_ROOT / "ANZA_KS_K0_K1_GATE_AUDIT_R1.md").write_text("\n".join(report) + "\n")
    return result


if __name__ == "__main__":
    print(run_audit()["status"])
