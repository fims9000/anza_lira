"""Frozen zero-training ANZA-EK E0/E1 causal audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .e0_audit import run_e0, save_e0_figures
from .e1_bench import PAIRS_PER_TASK, TASKS, freeze_benchmark, generate_pair
from .kernels import METHODS, deterministic_structure_score, generated_kernel_bank, kernel_feature_vector, local_correlations
from .metrics import summarize_scores
from .protocol import FREEZE_ROOT, RESULT_ROOT, canonical_hash, freeze_protocol


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "anza_ek/torus.py", "anza_ek/kernels.py", "anza_ek/e1_bench.py",
    "anza_ek/metrics.py", "anza_ek/e0_audit.py", "anza_ek/protocol.py",
    "anza_ek/run_e0_e1.py",
)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def source_manifest() -> dict[str, Any]:
    files = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    combined = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"files": files, "sha256": combined}


def freeze_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = freeze_protocol()
    benchmark = freeze_benchmark(FREEZE_ROOT / "e1_benchmark.json")
    code = source_manifest()
    _json(FREEZE_ROOT / "source_freeze.json", code)
    receipt = {
        "status": "ANZA_EK_E0_E1_INPUTS_FROZEN",
        "protocol_sha256": canonical_hash(protocol),
        "benchmark_sha256": benchmark["sha256"],
        "source_sha256": code["sha256"],
        "training_performed": False,
        "E2_opened": False,
        "cracks_accessed": False,
        "expert_accessed": False,
    }
    _json(FREEZE_ROOT / "freeze_receipt.json", receipt)
    return protocol, benchmark, receipt


def _score_patch(patch: np.ndarray, kernels: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    correlations = local_correlations(patch, kernels)
    return deterministic_structure_score(correlations), correlations, kernel_feature_vector(correlations)


def _e1_figure(metrics: dict[str, Any], output_root: Path) -> list[str]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    x = np.arange(len(TASKS))
    width = 0.19
    for method_index, method in enumerate(METHODS):
        ranking = [metrics[method][task]["matched_ranking"] for task in TASKS]
        tpr = [metrics[method][task]["tpr_at_fpr05"] for task in TASKS]
        offset = (method_index - 1.5) * width
        axes[0].bar(x + offset, ranking, width=width, label=method)
        axes[1].bar(x + offset, tpr, width=width, label=method)
    for axis, title in zip(axes, ("Matched-pair ranking", "TPR at FPR <= 0.05"), strict=True):
        axis.set_title(title)
        axis.set_ylim(0.0, 1.02)
        axis.set_xticks(x, [task.replace("_", "\n") for task in TASKS], fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7, loc="lower right")
    png = output_root / "e1_causal_metrics.png"
    svg = output_root / "e1_causal_metrics.svg"
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    return [str(png), str(svg)]


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# ANZA-EK E0/E1 report",
        "",
        "## Status",
        "",
        f"`{result['status']}`",
        "",
        "This is a zero-training mathematical and causal feature audit. It is not a learned segmentation, CRACKS, confirm, or expert result.",
        "",
        "## E0",
        "",
        f"- Mathematical status: `{result['e0']['status']}`",
        f"- Bilinear-grid L2 relative error: `{result['e0']['l2_relative_error_bilinear_grid']:.8g}`",
        f"- Bilinear-grid integral error: `{result['e0']['integral_error_bilinear_grid']:.8g}`",
        f"- Exact finite permutation inverse error: `{result['e0']['exact_discrete_permutation_error']:.8g}`",
    ]
    if result["e1"] is None:
        lines.extend(["", "E1 was not run because E0 failed.", ""])
        return "\n".join(lines)
    lines.extend([
        "",
        "## E1 task metrics",
        "",
        "| Method | Task | Ranking | AUROC | TPR@FPR05 | Fisher | Perturbed ranking | Stability corr. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        for task in TASKS:
            row = result["e1"]["metrics"][method][task]
            lines.append(f"| {method} | {task} | {row['matched_ranking']:.4f} | {row['auroc']:.4f} | {row['tpr_at_fpr05']:.4f} | {row['fisher_separation']:.4f} | {row['perturbed_matched_ranking']:.4f} | {row['perturbation_score_correlation']:.4f} |")
    lines.extend([
        "",
        "## Frozen causal gate",
        "",
        f"Strongest control: `{result['e1']['gate']['strongest_control']}`.",
        f"Passing identifiable tasks: `{result['e1']['gate']['passing_task_count']}` / `{len(TASKS)}`; required >=2.",
        f"Safety checks: `{json.dumps(result['e1']['gate']['safety_checks'], sort_keys=True)}`.",
        "",
        "No classifier, network training, conjugacy, E2, confirm, CRACKS, or expert data were opened.",
        "",
    ])
    return "\n".join(lines)


def run() -> dict[str, Any]:
    protocol, benchmark, receipt = freeze_inputs()
    if receipt["source_sha256"] != source_manifest()["sha256"]:
        raise ValueError("ANZA-EK source drift immediately after freeze")
    pre_run_path = FREEZE_ROOT / "pre_run_validator.json"
    if not pre_run_path.exists() or json.loads(pre_run_path.read_text()).get("research_status") != "ANZA_EK_E0_E1_PRE_RUN_PASS":
        raise ValueError("ANZA-EK pre-run validator has not authorized E1")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _json(RESULT_ROOT / "data_access_log.json", {
        "synthetic_patch_benchmark": True,
        "training": False,
        "classifier": False,
        "E2": False,
        "conjugacy": False,
        "confirm": False,
        "cracks": False,
        "expert": False,
    })
    e0 = run_e0(grid_size=int(protocol["grid_size_e0"]), K=int(protocol["K"]))
    figure_paths = save_e0_figures(RESULT_ROOT / "figures", K=int(protocol["K"]))
    if e0["status"] != "ANZA_EK_E0_PASS":
        status = "STOP_ANZA_EK_E0_MATHEMATICAL_VALIDATION_FAIL"
        result = {"status": status, "e0": e0, "e1": None, "protocol_sha256": receipt["protocol_sha256"], "benchmark_sha256": receipt["benchmark_sha256"], "source_sha256": receipt["source_sha256"], "figures": figure_paths, "training_performed": False, "E2_opened": False, "cracks_accessed": False, "expert_accessed": False}
        _json(RESULT_ROOT / "metrics.json", result)
        (RESULT_ROOT / "ANZA_EK_E0_E1_REPORT.md").write_text(_report(result))
        return result

    kernel_cache = {
        (method, orientation_index): generated_kernel_bank(
            method,
            orientation=orientation_index * np.pi / int(protocol["orientation_count"]),
            size=int(protocol["kernel_size"]),
            K=int(protocol["K"]),
            sigma=float(protocol["seed_sigma"]),
        )
        for method in METHODS
        for orientation_index in range(int(protocol["orientation_count"]))
    }
    raw_rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {method: {task: [] for task in TASKS} for method in METHODS}
    identifiability = {"pair_count": 0, "pixel_equal_count": 0, "minimum_l2_difference": float("inf")}
    for task in TASKS:
        for index in range(PAIRS_PER_TASK):
            pair = generate_pair(task, index)
            identifiability["pair_count"] += 1
            identifiability["pixel_equal_count"] += int(pair["pixel_equal"])
            identifiability["minimum_l2_difference"] = min(float(identifiability["minimum_l2_difference"]), float(pair["l2_difference"]))
            for method in METHODS:
                kernels = kernel_cache[(method, pair["orientation_index"])]
                positive_score, positive_corr, positive_features = _score_patch(pair["positive"], kernels)
                negative_score, negative_corr, negative_features = _score_patch(pair["negative"], kernels)
                positive_perturbed_score, _, _ = _score_patch(pair["positive_perturbed"], kernels)
                negative_perturbed_score, _, _ = _score_patch(pair["negative_perturbed"], kernels)
                row = {
                    "method": method,
                    "task": task,
                    "index": index,
                    "seed": pair["seed"],
                    "orientation_index": pair["orientation_index"],
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "positive_perturbed_score": positive_perturbed_score,
                    "negative_perturbed_score": negative_perturbed_score,
                    "matched_hit": int(positive_score > negative_score),
                    "positive_correlations": json.dumps(positive_corr.tolist()),
                    "negative_correlations": json.dumps(negative_corr.tolist()),
                    "positive_feature_vector": json.dumps(positive_features.tolist()),
                    "negative_feature_vector": json.dumps(negative_features.tolist()),
                }
                raw_rows.append(row)
                grouped[method][task].append(row)
    if identifiability["pixel_equal_count"] or identifiability["minimum_l2_difference"] <= 1e-6:
        raise ValueError("E1 contains geometry-identical positive/negative pairs")
    metrics = {method: {task: summarize_scores(grouped[method][task]) for task in TASKS} for method in METHODS}
    macro = {
        method: {
            key: float(np.mean([float(metrics[method][task][key]) for task in TASKS]))
            for key in ("matched_ranking", "tpr_at_fpr05", "perturbed_matched_ranking", "perturbation_score_correlation")
        }
        for method in METHODS
    }
    controls = METHODS[:-1]
    cat = METHODS[-1]
    strongest_control = max(controls, key=lambda method: macro[method]["matched_ranking"])
    task_gates = {}
    gain = float(protocol["gate"]["task_gain_tpr_or_ranking"])
    for task in TASKS:
        best_control_ranking = max(float(metrics[method][task]["matched_ranking"]) for method in controls)
        best_control_tpr = max(float(metrics[method][task]["tpr_at_fpr05"]) for method in controls)
        ranking_delta = float(metrics[cat][task]["matched_ranking"]) - best_control_ranking
        tpr_delta = float(metrics[cat][task]["tpr_at_fpr05"]) - best_control_tpr
        task_gates[task] = {"ranking_delta_vs_best_control": ranking_delta, "tpr_delta_vs_best_control": tpr_delta, "pass": bool(ranking_delta >= gain or tpr_delta >= gain)}
    passing_task_count = sum(int(value["pass"]) for value in task_gates.values())
    safety_checks = {
        "macro_clean_ranking": macro[cat]["matched_ranking"] - macro[strongest_control]["matched_ranking"] >= float(protocol["gate"]["macro_clean_ranking_noninferiority"]),
        "macro_perturbed_ranking": macro[cat]["perturbed_matched_ranking"] - macro[strongest_control]["perturbed_matched_ranking"] >= float(protocol["gate"]["macro_perturbed_ranking_noninferiority"]),
        "macro_perturbation_correlation": macro[cat]["perturbation_score_correlation"] - macro[strongest_control]["perturbation_score_correlation"] >= float(protocol["gate"]["macro_perturbation_correlation_noninferiority"]),
    }
    gate_pass = passing_task_count >= int(protocol["gate"]["minimum_passing_tasks"]) and all(safety_checks.values())
    status = "ANZA_EK_E1_MECHANISM_PASS" if gate_pass else "STOP_ERGODIC_ANOSOV_LOCAL_FEATURE_NO_MECHANISM"
    gate = {"pass": gate_pass, "strongest_control": strongest_control, "passing_task_count": passing_task_count, "task_gates": task_gates, "safety_checks": safety_checks}
    e1 = {"metrics": metrics, "macro": macro, "gate": gate, "identifiability": identifiability, "pair_count_per_task": PAIRS_PER_TASK, "method_count": len(METHODS), "learned_classifier": False}
    figure_paths.extend(_e1_figure(metrics, RESULT_ROOT / "figures"))
    result = {
        "status": status,
        "e0": e0,
        "e1": e1,
        "protocol_sha256": receipt["protocol_sha256"],
        "benchmark_sha256": receipt["benchmark_sha256"],
        "source_sha256": receipt["source_sha256"],
        "figures": figure_paths,
        "training_performed": False,
        "learned_classifier": False,
        "E2_opened": False,
        "conjugacy_opened": False,
        "confirm_created": False,
        "cracks_accessed": False,
        "expert_accessed": False,
    }
    _json(RESULT_ROOT / "e0_metrics.json", e0)
    _json(RESULT_ROOT / "e1_metrics.json", e1)
    _json(RESULT_ROOT / "metrics.json", result)
    _csv(RESULT_ROOT / "raw_per_pair.csv", raw_rows)
    (RESULT_ROOT / "ANZA_EK_E0_E1_REPORT.md").write_text(_report(result))
    _json(RESULT_ROOT / "TASK_STATE.json", {"status": status, "E2_authorized": gate_pass, "next_action": "Freeze a separate E2 protocol" if gate_pass else "STOP ANZA-EK; do not train this architecture", "training_performed": False, "cracks_accessed": False, "expert_accessed": False})
    return result
