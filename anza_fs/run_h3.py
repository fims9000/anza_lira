"""Bounded seed-41 F0--F3 ANZA-FS H3 experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hard_bench_v6 import SPLIT_SIZE
from .metrics import evaluate, select_matched_threshold, select_recall95_threshold
from .protocol import H3_ROOT, PREGRADIENT_ROOT, canonical_hash
from .training import one_batch_smoke, predict_variant, train_variant
from .validator import source_manifest


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT.parent / "_wip_backups" / "anza_lira" / "anza_fs_h3_checkpoints"


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def _ratio(value: float, reference: float) -> float | None:
    return float(value / reference) if reference > 0 else None


def _paired_bootstrap_false_bridge(first: list[dict[str, Any]], second: list[dict[str, Any]], *, seed: int = 41, resamples: int = 10000) -> dict[str, float]:
    if len(first) != len(second) or not first:
        raise ValueError("paired bootstrap requires aligned non-empty rows")
    first_values = np.asarray([row["false_bridge"] for row in first], dtype=np.float64)
    second_values = np.asarray([row["false_bridge"] for row in second], dtype=np.float64)
    delta = first_values - second_values
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        estimates[start : start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"mean_delta": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high), "resamples": resamples, "unit": "independent_scene"}


def _report(metrics: dict[str, Any]) -> str:
    lines = [
        "# ANZA-FS H3 report",
        "",
        "## Status",
        "",
        f"`{metrics['status']}`",
        "",
        "This is a frozen seed-41 synthetic development result on StressBench V6-HARD. It is not a confirm, CRACKS, multi-seed, H4, continuation, or expert result.",
        "",
        "| Variant | Threshold | Branch recall | False bridges | Negative events | FBR | Dice | Precision | Recall | clDice | Fragmentation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, value in metrics["variants"].items():
        row = value["primary"]["overall"]
        lines.append(
            f"| {variant} | {value['threshold']:.3f} | {row['branch_recall']:.4f} | {row['false_bridge_count']} | {row['negative_event_count']} | {row['false_bridge_rate']:.4f} | {row['dice']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['cldice']:.4f} | {row['fragmentation']:.4f} |"
        )
    lines.extend([
        "",
        "## Frozen gates",
        "",
        f"- F3 vs F1: `{json.dumps(metrics['comparisons']['F3_vs_F1'], sort_keys=True)}`",
        f"- F3 vs F2: `{json.dumps(metrics['comparisons']['F3_vs_F2'], sort_keys=True)}`",
        "",
        "Thresholds were selected only on calibration. Development was evaluated once after threshold freeze. Confirm, CRACKS, expert, H4, and parameter alternatives remained closed.",
        "",
    ])
    return "\n".join(lines)


def run(device: str = "cuda") -> dict[str, Any]:
    pregradient = json.loads((PREGRADIENT_ROOT / "validator.json").read_text())
    protocol = json.loads((PREGRADIENT_ROOT / "protocol.json").read_text())
    protocol_hash = canonical_hash(protocol)
    if pregradient.get("research_status") != "ANZA_FS_H3_PREGRADIENT_PASS" or not pregradient.get("training_authorized") or pregradient.get("protocol_sha256") != protocol_hash:
        raise ValueError("ANZA-FS H3 pre-gradient validator has not authorized training")
    if pregradient.get("code_sha256") != source_manifest()["sha256"]:
        raise ValueError("ANZA-FS H3 source changed after pre-gradient freeze")
    H3_ROOT.mkdir(parents=True, exist_ok=True)
    (H3_ROOT / "runs").mkdir(exist_ok=True)
    _json(H3_ROOT / "parent_pregradient.json", {
        "protocol_sha256": protocol_hash,
        "stressbench_sha256": pregradient["stressbench_sha256"],
        "code_sha256": pregradient["code_sha256"],
        "validator_status": pregradient["research_status"],
    })
    _json(H3_ROOT / "data_access_log.json", {
        "synthetic_train": True,
        "synthetic_calibration": "calibration[0:512]",
        "synthetic_development": "development[0:512]",
        "confirm": False,
        "test": False,
        "cracks": False,
        "H4": False,
        "expert": False,
    })
    smoke = one_batch_smoke(protocol, device=device)
    if not all(row["finite_gradients"] for row in smoke.values()):
        raise ValueError("ANZA-FS H3 CUDA vertical smoke failed")
    _json(H3_ROOT / "cuda_smoke.json", smoke)
    runs = {}
    for variant in protocol["matrix"]:
        runs[variant] = train_variant(
            variant,
            protocol=protocol,
            protocol_hash=protocol_hash,
            output_root=H3_ROOT / "runs",
            checkpoint_root=CHECKPOINTS,
            device=device,
        )

    probabilities: dict[str, list[np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    samples = None
    for variant in protocol["matrix"]:
        local_probabilities, local_samples, local_diagnostics = predict_variant(variant, Path(runs[variant]["checkpoint"]), device=device)
        probabilities[variant] = local_probabilities
        diagnostics[variant] = local_diagnostics
        if samples is None:
            samples = local_samples
    calibration_count = SPLIT_SIZE["calibration"]
    calibration_samples = samples[:calibration_count]
    development_samples = samples[calibration_count:]
    grid = [float(value) for value in protocol["threshold"]["grid"]]
    curves: dict[str, list[dict[str, Any]]] = {}
    selected: dict[str, dict[str, Any]] = {}
    for variant in protocol["matrix"]:
        curve = []
        for threshold in grid:
            summary, _ = evaluate(probabilities[variant][:calibration_count], calibration_samples, threshold)
            curve.append({"threshold": threshold, **summary["overall"]})
        curves[variant] = curve
        selected[variant] = select_recall95_threshold(curve, minimum_recall=float(protocol["gate"]["branch_recall_minimum"]))
    _json(H3_ROOT / "threshold_freeze.json", {"grid": grid, "selected": selected, "rule": protocol["threshold"]["primary_rule"]})

    variants: dict[str, Any] = {}
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    raw_rows: list[dict[str, Any]] = []
    for variant in protocol["matrix"]:
        threshold = float(selected[variant]["threshold"])
        primary, rows = evaluate(probabilities[variant][calibration_count:], development_samples, threshold)
        rows_by_variant[variant] = rows
        for row in rows:
            raw_rows.append({"variant": variant, "threshold": threshold, **row})
        variants[variant] = {
            "threshold": threshold,
            "calibration": selected[variant],
            "primary": primary,
            "diagnostics": diagnostics[variant],
            "run": runs[variant],
        }

    matched: dict[str, Any] = {}
    for comparator in ("F1_old_generic", "F2_free_foliation"):
        comparator_calibration = selected[comparator]
        f3_dice_threshold = select_matched_threshold(curves["F3_anza_fs"], "dice", float(comparator_calibration["dice"]))
        f3_precision_threshold = select_matched_threshold(curves["F3_anza_fs"], "precision", float(comparator_calibration["precision"]))
        comparator_development = variants[comparator]["primary"]
        f3_dice_development, _ = evaluate(probabilities["F3_anza_fs"][calibration_count:], development_samples, float(f3_dice_threshold["threshold"]))
        f3_precision_development, _ = evaluate(probabilities["F3_anza_fs"][calibration_count:], development_samples, float(f3_precision_threshold["threshold"]))
        matched[comparator] = {
            "fragmentation_at_matched_dice": {
                "comparator_threshold": variants[comparator]["threshold"],
                "F3_threshold": f3_dice_threshold["threshold"],
                "calibration_dice_target": comparator_calibration["dice"],
                "comparator_development_fragmentation": comparator_development["overall"]["fragmentation"],
                "F3_development_fragmentation": f3_dice_development["overall"]["fragmentation"],
                "ratio_F3_over_comparator": _ratio(f3_dice_development["overall"]["fragmentation"], comparator_development["overall"]["fragmentation"]),
            },
            "cldice_at_matched_precision": {
                "comparator_threshold": variants[comparator]["threshold"],
                "F3_threshold": f3_precision_threshold["threshold"],
                "calibration_precision_target": comparator_calibration["precision"],
                "comparator_development_cldice": comparator_development["overall"]["cldice"],
                "F3_development_cldice": f3_precision_development["overall"]["cldice"],
                "delta_F3_minus_comparator": f3_precision_development["overall"]["cldice"] - comparator_development["overall"]["cldice"],
            },
        }

    f1 = variants["F1_old_generic"]["primary"]["overall"]
    f2 = variants["F2_free_foliation"]["primary"]["overall"]
    f3 = variants["F3_anza_fs"]["primary"]["overall"]
    fbr_ratio_f1 = _ratio(f3["false_bridge_rate"], f1["false_bridge_rate"])
    fbr_ratio_f2 = _ratio(f3["false_bridge_rate"], f2["false_bridge_rate"])
    frag_ratio_f2 = matched["F2_free_foliation"]["fragmentation_at_matched_dice"]["ratio_F3_over_comparator"]
    dice_delta_f1 = f3["dice"] - f1["dice"]
    dice_delta_f2 = f3["dice"] - f2["dice"]
    f1_fbr_ok = fbr_ratio_f1 is not None and fbr_ratio_f1 <= float(protocol["gate"]["F3_vs_F1_fbr_ratio_max"])
    f2_fbr_ok = fbr_ratio_f2 is not None and fbr_ratio_f2 <= float(protocol["gate"]["F3_vs_F2_fbr_ratio_max"])
    f2_frag_ok = frag_ratio_f2 is not None and frag_ratio_f2 <= float(protocol["gate"]["F3_vs_F2_fragmentation_ratio_max"])
    dice_f1_ok = dice_delta_f1 >= float(protocol["gate"]["dice_noninferiority"])
    dice_f2_ok = dice_delta_f2 >= float(protocol["gate"]["dice_noninferiority"])
    practical_pass = bool(f1_fbr_ok and dice_f1_ok)
    hyperbolic_specific_pass = bool(practical_pass and dice_f2_ok and (f2_fbr_ok or f2_frag_ok))
    if not practical_pass:
        status = "STOP_ANZA_FS_NO_PRACTICAL_STRUCTURAL_GAIN"
    elif not hyperbolic_specific_pass:
        status = "ANZA_FS_PRACTICAL_GAIN_FOLIATION_NOT_HYPERBOLIC_SPECIFIC"
    else:
        status = "ANZA_FS_HYPERBOLIC_FOLIATION_PASS"
    comparisons = {
        "F3_vs_F1": {
            "fbr_ratio": fbr_ratio_f1,
            "dice_delta": dice_delta_f1,
            "fbr_gate": f1_fbr_ok,
            "dice_noninferiority": dice_f1_ok,
            "paired_fbr_delta_ci": _paired_bootstrap_false_bridge(rows_by_variant["F3_anza_fs"], rows_by_variant["F1_old_generic"]),
        },
        "F3_vs_F2": {
            "fbr_ratio": fbr_ratio_f2,
            "fragmentation_ratio_at_matched_dice": frag_ratio_f2,
            "dice_delta": dice_delta_f2,
            "fbr_gate": f2_fbr_ok,
            "fragmentation_gate": f2_frag_ok,
            "dice_noninferiority": dice_f2_ok,
            "paired_fbr_delta_ci": _paired_bootstrap_false_bridge(rows_by_variant["F3_anza_fs"], rows_by_variant["F2_free_foliation"]),
        },
    }
    metrics = {
        "status": status,
        "practical_gate_pass": practical_pass,
        "hyperbolic_specific_gate_pass": hyperbolic_specific_pass,
        "protocol_sha256": protocol_hash,
        "stressbench_sha256": pregradient["stressbench_sha256"],
        "seed": 41,
        "variants": variants,
        "matched_operating_points": matched,
        "comparisons": comparisons,
        "training_performed": True,
        "confirm_opened": False,
        "test_opened": False,
        "cracks_accessed": False,
        "H4_opened": False,
        "expert_accessed": False,
        "lambda_tuned": False,
        "M_tuned": False,
        "base_scale_tuned": False,
    }
    _json(H3_ROOT / "metrics.json", metrics)
    _json(H3_ROOT / "calibration_curves.json", curves)
    _json(H3_ROOT / "operator_diagnostics.json", diagnostics)
    _csv(H3_ROOT / "raw_per_sample.csv", raw_rows)
    (H3_ROOT / "ANZA_FS_H3_REPORT.md").write_text(_report(metrics))
    _json(H3_ROOT / "TASK_STATE.json", {
        "status": status,
        "practical_gate_pass": practical_pass,
        "hyperbolic_specific_gate_pass": hyperbolic_specific_pass,
        "next_action": "Run F1/F2/F3 seeds 42/43 only under a separately frozen protocol" if practical_pass else "STOP local ANZA architecture development; do not create another kernel family",
        "confirm_opened": False,
        "cracks_accessed": False,
        "H4_opened": False,
        "expert_accessed": False,
    })
    return metrics
